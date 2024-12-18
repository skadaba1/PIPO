from losses.dpo import calculate_DPO_loss, get_log_prob
from losses.mlm import compute_mlm_loss_batch
import torch
from tqdm import tqdm
import wandb

def process_batch(batch, model, ref_model, beta, reg_weight, preference_weight):

    prefered_ids = batch['prefered_ids']
    disprefered_ids = batch['disprefered_ids']
    prefered_mask = batch['prefered_mask']
    disprefered_mask = batch['disprefered_mask']
    strengths = batch['strengths']

    model_prefered_output = model(prefered_ids, attention_mask=prefered_mask).logits
    model_disprefered_output = model(disprefered_ids, attention_mask=disprefered_mask).logits

    ref_prefered_output = ref_model(prefered_ids, attention_mask=prefered_mask).logits
    ref_disprefered_output = ref_model(disprefered_ids, attention_mask=disprefered_mask).logits

    model_prefered_log_prob = get_log_prob(model_prefered_output, prefered_ids, prefered_mask)
    model_disprefered_log_prob = get_log_prob(model_disprefered_output, disprefered_ids, disprefered_mask)

    ref_prefered_log_prob = get_log_prob(ref_prefered_output, prefered_ids, prefered_mask)
    ref_disprefered_log_prob = get_log_prob(ref_disprefered_output, disprefered_ids, disprefered_mask)

    loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins, pi_logratios,ref_logratios  = calculate_DPO_loss(
        model_prefered_log_prob, model_disprefered_log_prob,
        ref_prefered_log_prob, ref_disprefered_log_prob, strengths,
        beta=beta, label_smoothing=preference_weight
    )
    mlm_model_loss = compute_mlm_loss_batch(model, prefered_ids, prefered_mask, disprefered_ids, disprefered_mask)
    mlm_ref_loss = 0.0 #compute_mlm_loss_batch(ref_model, prefered_ids, prefered_mask, disprefered_ids, disprefered_mask)
    total_loss = loss + reg_weight * mlm_model_loss
    return total_loss, mlm_model_loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins, pi_logratios, ref_logratios

def validate(model, val_dataloader, ref_model, beta, reg_weight, preference_weight):

    with torch.no_grad():
        batch = next(val_dataloader)
        total_loss, mlm_ref_loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins, pi_logratios, ref_logratios = process_batch(batch, model, ref_model, beta, reg_weight, preference_weight)

    return total_loss, mlm_ref_loss, prefered_relative_logprob , disprefered_relative_logprob, reward_accuracies, reward_margins


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, model, ref_model, optimizer, scheduler, train_dataloader, val_dataloader,
                 gradient_accumulation_steps=8, epochs=1, beta=0.1, reg_weight=1e-1, preference_weight=1e-1,
                 print_interval=100, local_rank=0):
        self.model = DDP(model.to(local_rank), device_ids=[local_rank], output_device=local_rank)
        self.ref_model = ref_model.to(local_rank)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs
        self.beta = beta
        self.reg_weight = reg_weight
        self.preference_weight = preference_weight
        self.print_interval = print_interval
        self.local_rank = local_rank

    def train(self):
        self.model.train()
        self.ref_model.eval()
        self.optimizer.zero_grad()

        for epoch in range(self.epochs):
            num_batches = len(self.train_dataloader)
            val_loss_running_avg = 0.0

            for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")):
                total_loss, mlm_ref_loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, \
                reward_margins, pi_logratios, ref_logratios = self.process_batch(batch)

                total_loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if dist.get_rank() == 0:  # Log only on the main process
                        wandb.log({
                            'train_loss': total_loss.item(),
                            'prefered_relative_logprob': prefered_relative_logprob,
                            'disprefered_relative_logprob': disprefered_relative_logprob,
                            'reward_accuracy': reward_accuracies,
                            'reward_margin': reward_margins,
                            'mlm_loss': mlm_ref_loss,
                        })

                if (batch_idx + 1) % (self.print_interval * self.gradient_accumulation_steps) == 0:
                    self.validate(val_loss_running_avg, epoch, batch_idx)

    def process_batch(self, batch):
        """Process a single batch."""
        total_loss, mlm_ref_loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, \
        reward_margins, pi_logratios, ref_logratios = process_batch(
            batch, self.model, self.ref_model, self.beta, self.reg_weight, self.preference_weight
        )
        return total_loss, mlm_ref_loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, \
               reward_margins, pi_logratios, ref_logratios

    def validate(self, val_loss_running_avg, epoch, batch_idx):
        """Run validation and log metrics."""
        self.model.eval()

        val_total_loss, val_mlm_ref_loss, val_prefered_relative_logprob, val_disprefered_relative_logprob, \
        val_reward_accuracies, val_reward_margins = validate(
            self.model, self.val_dataloader, self.ref_model, self.beta, self.reg_weight, self.preference_weight
        )

        val_loss_running_avg += val_total_loss

        if dist.get_rank() == 0:  # Log only on the main process
            wandb.log({
                'val_loss': val_total_loss,
                'val_prefered_relative_logprob': val_prefered_relative_logprob,
                'val_disprefered_relative_logprob': val_disprefered_relative_logprob,
                'val_reward_accuracy': val_reward_accuracies,
                'val_reward_margin': val_reward_margins,
                'val_mlm_loss': val_mlm_ref_loss,
            })

        self.model.train()

    def setup_distributed():
        """Set up the distributed process group."""
        dist.init_process_group(backend='nccl')

    def cleanup_distributed():
        """Clean up the distributed process group."""
        dist.destroy_process_group()

