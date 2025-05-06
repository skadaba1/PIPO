#!/usr/bin/env python3
import argparse
import yaml
import os
import time
import psutil
import torch
import torch.nn.functional as F
import logging
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EsmForMaskedLM
from data.data_preparation import prepare_dataloaders
from lora_model import prepare_model, freeze_model

class Trainer:
    def __init__(self, model, ref_model, optimizer, scheduler,
                 dataloaders, cfg, tokenizer, device):
        self.cfg         = cfg
        self.device      = device
        self.model       = model.to(device)
        self.ref_model   = ref_model.to(device)
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.train_loader, self.val_loader, _ = dataloaders
        self.tokenizer   = tokenizer

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # logging setup
        log_path = cfg["logging"]["log_file"]
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        if not self.logger.handlers:
            self.logger.addHandler(fh)

        # track which epoch to start from
        self.start_epoch = 1

        # if there's a checkpoint on disk, load it
        ckpt_path = cfg.get("save", {}).get("checkpoint_path")
        if ckpt_path and os.path.isfile(ckpt_path):
            self.logger.info(f"Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            # resume next epoch
            self.start_epoch = ckpt.get("epoch", 1) + 1
            self.logger.info(f"Resumed at epoch {self.start_epoch}")

        self.model.train()
        self.ref_model.eval()

    def _log(self, metrics: dict, prefix: str):
        self.logger.info(f"{prefix}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    def validate(self):
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            batch = next(self.val_loader)
            out = process_batch(
                batch, self.model, self.ref_model,
                self.cfg["training"]["beta"],
                self.cfg["training"]["reg_weight"],
                self.cfg["training"]["preference_weight"],
                self.tokenizer,
                self.device
            )
        self.model.train()
        return out

    def save_checkpoint(self, epoch: int):
        ckpt_path = self.cfg.get("save", {}).get("checkpoint_path")
        if not ckpt_path:
            return
        os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
        }, ckpt_path)
        self.logger.info(f"Epoch {epoch} checkpoint saved to {ckpt_path}")

    def train(self):
        steps     = 0
        running_val = 0.0
        epochs    = self.cfg["training"]["epochs"]
        interval  = self.cfg["training"]["print_interval"]

        for epoch in range(self.start_epoch, epochs + 1):
            for batch in self.train_loader:
                steps += 1
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    tote, loss, prlp, drlp, accs, margins, *_ = process_batch(
                        batch, self.model, self.ref_model,
                        self.cfg["training"]["beta"],
                        self.cfg["training"]["reg_weight"],
                        self.cfg["training"]["preference_weight"],
                        self.tokenizer,
                        self.device
                    )
                self.scaler.scale(tote).backward()

                if steps % self.cfg["training"]["grad_accum"] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                if steps % interval == 0:
                    _, vloss, vprlp, vdrlp, vaccs, vmargins, *_ = self.validate()
                    running_val += vloss
                    avg_val = running_val / (steps // interval)
                    self.scheduler.step(avg_val)
                    self._log({
                        "loss": vloss, "pref_rel_lp": vprlp,
                        "dispref_rel_lp": vdrlp, "acc": vaccs,
                        "margin": vmargins
                    }, prefix="val")

                self._log({
                    "loss": loss.item(), "pref_rel_lp": prlp.item(),
                    "dispref_rel_lp": drlp.item(),
                    "acc": accs, "margin": margins
                }, prefix="train")

            # end of this epoch → save
            self.save_checkpoint(epoch)

def get_log_prob(logits, labels, attention_mask, tokenizer):
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1) * attention_mask
    seq_lens = (labels != tokenizer.pad_token_id).sum(dim=-1)
    return gathered.sum(dim=-1) / seq_lens.float()

def calculate_DPO_loss(
    model_pref_lp, model_disp_lp,
    ref_pref_lp, ref_disp_lp,
    strengths,
    beta=1.0, ipo=True,
    label_smoothing=0.0, reference_free=False
):
    beta = beta * strengths
    pref_rel = model_pref_lp - ref_pref_lp
    disp_rel = model_disp_lp - ref_disp_lp
    pi_lr = model_pref_lp - model_disp_lp
    ref_lr= ref_pref_lp - ref_disp_lp
    if reference_free:
        ref_lr = 0.0
    logits = pi_lr - ref_lr
    reward_acc    = (pref_rel > disp_rel).float().mean(dim=-1)
    reward_margin = (pref_rel - disp_rel).mean(dim=-1)

    if ipo:
        loss = ((logits - 1/(2 * beta))**2).mean()
    else:
        loss = (
            -F.logsigmoid(beta * logits)*(1-label_smoothing)
            -F.logsigmoid(-beta * logits)*label_smoothing
        ).mean()

    return loss, pref_rel.mean(dim=-1), disp_rel.mean(dim=-1), reward_acc, reward_margin, pi_lr, ref_lr

def compute_mlm_loss_batch(model, pref_ids, pref_mask, disp_ids, disp_mask, tokenizer, p_mask=0.15):
    pad_id  = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    total   = None
    for ids, mask in [(pref_ids, pref_mask), (disp_ids, disp_mask)]:
        labels = ids.clone()
        masked = ids.clone()
        batch_size, seq_len = ids.shape
        for i in range(batch_size):
            valid  = torch.where(ids[i] != pad_id)[0].cpu().numpy()
            num_m  = int(p_mask * len(valid))
            idx    = np.random.choice(valid, size=num_m, replace=False)
            masked[i, idx] = mask_id
            labels[i, [j for j in range(seq_len) if j not in idx]] = -100
        out = model(masked.to(model.device), attention_mask=mask.to(model.device), labels=labels.to(model.device))
        total = out.loss if total is None else total + out.loss
    return total

def process_batch(batch, model, ref_model, beta, reg_w, pref_w, tokenizer, device):
    p_ids    = batch['prefered_ids'].to(device)
    p_mask   = batch['prefered_mask'].to(device)
    d_ids    = batch['disprefered_ids'].to(device)
    d_mask   = batch['disprefered_mask'].to(device)
    strengths= batch['strengths'].to(device)

    m_pref_logits = model(p_ids, attention_mask=p_mask).logits
    m_disp_logits = model(d_ids, attention_mask=d_mask).logits
    r_pref_logits = ref_model(p_ids, attention_mask=p_mask).logits
    r_disp_logits = ref_model(d_ids, attention_mask=d_mask).logits

    m_pref_lp = get_log_prob(m_pref_logits, p_ids, p_mask, tokenizer)
    m_disp_lp = get_log_prob(m_disp_logits, d_ids, d_mask, tokenizer)
    r_pref_lp = get_log_prob(r_pref_logits, p_ids, p_mask, tokenizer)
    r_disp_lp = get_log_prob(r_disp_logits, d_ids, d_mask, tokenizer)

    dpo_loss, pref_rel_lp, disp_rel_lp, acc, margin, _, _ = calculate_DPO_loss(
        m_pref_lp, m_disp_lp, r_pref_lp, r_disp_lp,
        strengths,
        beta=beta,
        label_smoothing=pref_w
    )

    mlm_loss = compute_mlm_loss_batch(model, p_ids, p_mask, d_ids, d_mask, tokenizer)
    total_loss = dpo_loss + reg_w * mlm_loss

    return total_loss, dpo_loss, pref_rel_lp, disp_rel_lp, acc, margin