import argparse
import yaml
import torch
from transformers import AutoTokenizer, EsmForMaskedLM
from data.data_preparation import prepare_dataloaders
from lora_model import prepare_model, freeze_model 
from utils import Trainer

# at the top of your training script
import psutil, os, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()


    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['checkpoint'])
    train_loader, val_loader, test_loader = prepare_dataloaders(
        tokenizer=tokenizer,
        max_length=cfg['training']['max_length'],
        batch_size=cfg['training']['batch_size'],
        device=device,
        strength_threshold=cfg['training'].get('strength_threshold', None),
        csv_file=cfg['data'].get('csv_file'),
        fasta_file=cfg['data'].get('fasta_file'),
        txt_file=cfg['data'].get('txt_file'),
        txt_sep=cfg['data'].get('txt_sep'),
        split_method=cfg['data'].get('split_method','random'),
        split_kwargs=cfg['data'].get('split_kwargs',{}),
        output_dir=cfg.get('output_dir','data/outputs'),
    )
    base = EsmForMaskedLM.from_pretrained(cfg['model']['checkpoint'])
    model = prepare_model(
        base,
        rank=int(cfg['model']['rank']),
        num_layers=int(cfg['model']['checkpoint'].split('_t')[1].split('_')[0]),
        dropout_rate=float(cfg['model']['dropout'])
    )

    ref_base  = EsmForMaskedLM.from_pretrained(cfg['model']['checkpoint'])
    ref_model = freeze_model(ref_base)

    lr = float(cfg['training']['lr'])
    wd = float(cfg['training']['weight_decay'])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=10,
        threshold=1e-2, min_lr=1e-5
    )

    trainer = Trainer(
        model, ref_model, optimizer, scheduler,
        (train_loader, val_loader, test_loader),
        cfg, tokenizer, device
    )
    trainer.train()


if __name__ == "__main__":
    main()