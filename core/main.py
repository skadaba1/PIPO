
import torch
import torch.nn.functional as F
from transformers import AutoModel, EsmForMaskedLM
import esm
from transformers import AutoTokenizer
import os
from trainer import Trainer
from datasets import load_dataset
import numpy as np
import pandas as pd
import itertools
from functools import partial
from model.model_utils import prepare_model, freeze_model, CosineAnnealingWithMinLRScheduler
from dataset_utils import TMLibrary, collate_fn
import wandb


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    Trainer.setup_distributed()
    model_size = (30, '150M')
    model_checkpoint = f"facebook/esm2_t{model_size[0]}_{model_size[1]}_UR50D"
    EPS = 1e-6

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = prepare_model(EsmForMaskedLM.from_pretrained(model_checkpoint), rank=8, num_layers=model_size[0], dropout_rate=0.05).to(device)
    ref_model = freeze_model(EsmForMaskedLM.from_pretrained(model_checkpoint)).to(device) #prepare_model(EsmForMaskedLM.from_pretrained(model_checkpoint), rank=8, num_layers=model_size[0], dropout_rate=0.05).to(device)

    total_params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters in policy: {total_params(model)}")
    print(f"Total number of trainable parameters in ref_model: {total_params(ref_model)}")

    lr=5e-4
    batch_size = 8
    epochs = 1
    beta = 1e0 # reward weight
    max_length = 250
    reg_weight = 1e-1 # mlm weight
    gradient_accumulation_steps = 4
    preference_weight = 0.0 # label smoothing parameter
    print_interval = 10
    n_samples_train = 1000000
    n_samples_val = 150000

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)
    optimizer_ref = None #torch.optim.AdamW(filter(lambda p: p.requires_grad, ref_model.parameters()), lr=lr, weight_decay=1e-3)

    dataset = TMLibrary(file_path="/content/ab_library_train.csv", tokenizer=tokenizer, max_length=max_length, max_visitations=1e9, split='train', threshold=0.0, n_samples=n_samples_train)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn, device=device))

    val_dataset = TMLibrary(file_path="/content/ab_library_val.csv", tokenizer=tokenizer, max_length=max_length, max_visitations=1e9, split='val', threshold=0.0, n_samples=n_samples_val)
    val_dataloader = itertools.cycle(torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn, device=device)))

    num_training_steps = epochs * len(train_dataloader)
    num_warmup_steps = int(0.05 * num_training_steps)  # 10% warmup
    num_hold_steps = int(0.1 * num_training_steps)  # 10% hold
    min_lr = 1e-5  # Minimum learning rate you want to set

    scheduler = CosineAnnealingWithMinLRScheduler(optimizer, num_warmup_steps, num_hold_steps, num_training_steps, min_lr=min_lr)

    wandb.login()
    wandb.init()
    trainer = Trainer(model, ref_model, optimizer, scheduler, train_dataloader, val_dataloader,
                      gradient_accumulation_steps=8, epochs=1, beta=0.1, reg_weight=1e-1, preference_weight=1e-1,
                      print_interval=100, local_rank=local_rank)
    trainer.train()

    Trainer.cleanup_distributed()