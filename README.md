# SMPO: Score-Based Preference Optimization for Protein Fitness Modeling 
Preference tuning protein language models with low N experimental datasets

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

For data partitioning using `graph-part`, `mmseqs2` and `emboss` are also required. These are best installed via conda:
```setup
conda install -c bioconda mmseqs2 emboss
```
If you are using Google Colab, the notebook `notebooks/baselines/tm_design_v0_1(LRH).ipynb` contains cells to install these dependencies.

## Training

### Jupyter Notebook Training

The primary training workflow is implemented in Jupyter notebooks. All training was done in Google Colab with an A100 GPU. 

**To train the model(s) in the paper:**

1.  **Setup Environment:**
    *   Ensure all dependencies from `requirements.txt` are installed.
    *   If using conda for `mmseqs2` and `emboss`, create and activate your conda environment.
    *   Mount Google Drive if using Colab and your data/models are stored there (as shown in the first cell of the notebook).

2.  **Data Preparation:**
    *   Due to the work being under peer review, the specific datasets used in the paper cannot be released at this time.
    *   **Required Data Schema:** Your input CSV files must contain at least two columns:
        *   A column containing the amino acid sequences (e.g., named `Sequence` or `aa_seq`). The sequences should be strings of standard one-letter amino acid codes.
        *   A column containing the corresponding numerical scores/labels (e.g., named `Pred_affinity` or `score`). These scores should be floating-point numbers.

3.  **Model Configuration and Training:**
    *   Execute the cells in `notebooks/baselines/tm_design_v0_1(LRH).ipynb` sequentially.
    *   The notebook contains the complete training pipeline including model setup, data loading, and training loops.

### Standalone Python Training

The repository provides several standalone Python scripts for training:

#### Main Training Script (`core/main.py`)

This script implements the core SMPO (Score-based Preference Optimization) training using DPO (Direct Preference Optimization) with optional MLM (Masked Language Modeling) regularization:

```bash
# Example usage (requires modification of file paths in the script)
cd core
python main.py
```

**Configuration:** Before running, you need to modify the file paths and hyperparameters in `core/main.py`:
- **Line 47:** Update `file_path="/content/ab_library_train.csv"` to your training data path
- **Line 51:** Update `file_path="/content/ab_library_val.csv"` to your validation data path

**Key hyperparameters you can adjust:**
- `lr`: Learning rate (default: 5e-4)
- `batch_size`: Training batch size (default: 8)
- `epochs`: Number of training epochs (default: 1)
- `beta`: DPO reward weight (default: 1.0)
- `reg_weight`: MLM regularization weight (default: 0.1)
- `preference_weight`: Label smoothing parameter (default: 0.0)
- `max_length`: Maximum sequence length (default: 250)

The script expects CSV files with columns matching the `TMLibrary` dataset class:
- `Sequence`: amino acid sequences
- `Pred_affinity`: numerical scores

#### Core Trainer Module (`core/trainer.py`)

The trainer module implements the distributed training logic with DDP (Distributed Data Parallel) support:

**Features:**
- **DPO Loss:** Implements preference-based optimization using preferred/non-preferred sequence pairs
- **MLM Regularization:** Optional masked language modeling to maintain protein language understanding
- **Distributed Training:** Supports multi-GPU training via PyTorch DDP
- **Validation:** Built-in validation loop with metric logging to Weights & Biases

**Key methods:**
- `train()`: Main training loop with gradient accumulation
- `process_batch()`: Processes preference pairs and computes DPO + MLM losses
- `validate()`: Validation with logging of reward accuracy and margins

#### Data Partitioning (`graphpart.py`)

Use this script to partition your dataset using homology-based or random splitting:

```bash
# For homology-based partitioning (requires graph-part, mmseqs2, emboss)
python graphpart.py
```

**Functions:**
- `homology_partition(file_path, sequences)`: Creates train/validation/test splits based on sequence similarity using graph partitioning
  - Uses needle alignment algorithm with 50% similarity threshold
  - Splits: 80% train, 15% test, 5% validation
  - Requires `graph-part` package and alignment tools

- `split_and_save_dataframe(df, train_ratio=0.8, val_ratio=0.05, test_ratio=0.15)`: Creates random splits
  - Shuffles data with fixed random seed (42) for reproducibility
  - Saves splits to separate CSV files
  - Option to filter unbounded values

**Usage Example:**
```python
import pandas as pd
from graphpart import split_and_save_dataframe

# For random splitting
df = pd.read_csv("your_data.csv")
train_df, val_df, test_df = split_and_save_dataframe(
    df, 
    train_file='train.csv', 
    val_file='val.csv', 
    test_file='test.csv'
)

# For homology-based splitting (requires sequences list)
sequences = df['Sequence'].tolist()
homology_partition("your_data.csv", sequences)
```

## Evaluation

### Sequence Evolution Evaluation (`evaluation/sequence_evolution.py`)

This script implements directed evolution simulation using the trained model to evolve protein sequences toward higher fitness scores:

```bash
cd evaluation
python sequence_evolution.py
```

**Setup Requirements:** Before running, you must set these global variables in the script:
- `model`: Your trained SMPO model
- `tokenizer`: ESM tokenizer from transformers
- `device`: CUDA device
- `params`: Linear regression parameters from fitness model
- `linear_model`: Fitted linear regression function

**Key Functions:**
- `evolve(seq, num_children=100, max_generations=20)`: Main evolution function
  - Takes a starting protein sequence
  - Generates mutant children via SNPs and masking
  - Selects best performers based on predicted fitness
  - Continues until positive fitness score achieved

- `snp(seq, max_length=50)`: Single nucleotide polymorphism simulation
  - Randomly decides whether to apply SNP or general masking
  - Uses 10% replacement ratio with mean 3 additional masks
  - Returns generated sequence and log probabilities

**Usage Example:**
```python
# After setting global variables
evolved_seq, final_score, generations = evolve(
    'EEEIRTTNPVATEQYGMSPYGLLGRLEA', 
    num_children=500, 
    max_generations=20
)
```

### Regression Analysis (`evaluation/regression/regression.py`)

For regression analysis and model fitting with cross-validation:

```bash
cd evaluation/regression
python regression.py
```

**Functions:**
- `fit_linear_model(probs, scores, model, initial_guess=None, plot=True)`: 
  - Fits linear or exponential models with K-fold cross validation (K=10)
  - Returns RMSE, relative RMSE, and R² scores for each fold
  - Plots actual vs predicted scores for the final fold
  - Computes average metrics across all folds

- `exponential_model(x, a, b)`: Exponential curve fitting function (a * exp(b * x))
- `linear_model(x, a, b)`: Linear regression function (a * x + b)

**Usage Example:**
```python
from regression import fit_linear_model, linear_model
import numpy as np

# Your sequence probabilities and experimental scores
probs = np.array([...])  # Model predicted probabilities
scores = np.array([...])  # Experimental fitness scores

# Fit linear model with cross-validation
fitted_model = fit_linear_model(probs, scores, linear_model)
```

### SAE (Sparse Autoencoder) Analysis (`evaluation/sae/`)

The SAE module provides tools for interpretability analysis of protein language model representations:

```bash
cd evaluation/sae
python analyze.py          # Main SAE training and analysis
python attention_inspection.py  # Attention mechanism inspection
```

#### Main Analysis (`analyze.py`)

**Key Functions:**
- `analyze_with_sae(model, tokenizer, dataset, target_layer, latent_dim=256)`:
  - Extracts activations from specified model layer
  - Trains sparse autoencoder on collected activations
  - Returns trained SAE and activation dataset

- `collect_activations(model, dataset_loader, activation_extractor, max_samples=10000)`:
  - Collects hidden state activations from model layers
  - Handles batched processing for memory efficiency
  - Reshapes activations for SAE training

- `analyze_feature_activations(sae, model, tokenizer, protein_sequence, target_layer)`:
  - Analyzes which SAE features activate for specific protein sequences
  - Returns feature activation map aligned with sequence positions

**Usage Example:**
```python
# Train SAE on model layer activations
trained_sae, activations = analyze_with_sae(
    model=your_model,
    tokenizer=tokenizer, 
    dataset=your_dataset,
    target_layer="esm.encoder.layer.15",  # Layer 15 of ESM-2
    latent_dim=512
)

# Analyze specific sequence
feature_map = analyze_feature_activations(
    trained_sae, model, tokenizer, 
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "esm.encoder.layer.15"
)
```

#### Attention Inspection (`attention_inspection.py`)

**Functions:**
- `get_relative_attention_scores(sequence, indexes=None)`: 
  - Computes attention scores for protein sequences
  - Can focus on specific sequence regions (e.g., binding sites)
  - Returns relative attention compared to baseline

- `compute_attention(model, x)`: Extracts and averages attention across all layers
- `plot_attention(attention_heatmap)`: Visualizes attention patterns as heatmaps

**Usage for Motif Analysis:**
```python
# Analyze attention for sequences with known motifs
sequences_with_motifs = [
    ("PROTEIN_SEQUENCE", [(start_idx, end_idx)]),  # (sequence, motif_positions)
    ...
]

# Compare attention distributions
__main__(sequences_with_motifs)  # Generates statistical comparison plots
```

### ClinVar Evaluation (`evaluation/clinvar/`)

Tools for evaluating model performance on clinically relevant variants:

```bash
cd evaluation/clinvar
python fetch_data.py       # Fetch protein data from UniProt
python perturb.py          # Analyze variant effects
```

**Functions:**
- `fetch_data(uniprot_id)`: Asynchronously fetches protein sequences and transmembrane domain annotations
- `fetch_fasta(session, uniprot_id)`: Downloads FASTA sequences from UniProt
- `fetch_json(session, uniprot_id)`: Retrieves structural annotations including transmembrane regions

### Motif Analysis (`evaluation/motifs/`)

Tools for analyzing the effect of protein motifs on model predictions:

```bash
cd evaluation/motifs
python prep_data.py        # Prepare motif datasets
python perturb_motif.py    # Analyze motif perturbations
```

**Functions:**
- `find_motifs(sequences, motif)`: Finds occurrences of specific motifs (e.g., 'GXXXG') in protein sequences
- `replace_randomly(string, target, replacement, num_to_replace)`: Randomly replaces motif occurrences
- `select_random_not_in_motif(sequence, motif, roi)`: Selects random positions outside of motif regions for control mutations

## Notebooks

The repository contains several Jupyter notebooks for comprehensive model training and evaluation:

- `notebooks/baselines/tm_design_v0_0.ipynb`: Basic baseline evaluation with traditional ML approaches
- `notebooks/baselines/tm_design_v0_1(LRH).ipynb`: Supervised Fine-Tuning (SFT) baseline with LoRA
- `notebooks/tm_design_v0_2.ipynb`: Advanced SMPO (Score-based Preference Optimization) implementation

### Comprehensive Notebook Instructions

#### Prerequisites for All Notebooks

**Environment Setup:**
All notebooks are designed to run in Google Colab with the following requirements:
- Google Colab Pro recommended (for A100 GPU access)
- Google Drive mounted for data storage and model checkpoints
- At least 25GB of available GPU memory for 150M parameter models

**Data Requirements:**
- CSV files with protein sequences and fitness scores
- Required columns: `Sequence`/`aa_seq` (amino acid sequences) and score columns
- Files should be uploaded to Google Drive and paths updated in notebook cells

#### Baseline Notebook 1: SFT Training (`tm_design_v0_1(LRH).ipynb`)

This notebook implements **Supervised Fine-Tuning (SFT)** of ESM-2 with LoRA for protein fitness prediction.

**Step-by-Step Execution:**

1. **Mount Google Drive & Install Dependencies**
   ```python
   # Cell 1: Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Cell 2: Install required packages
   !pip install transformers evaluate datasets requests pandas scikit-learn fair-esm wandb graph-part python-Levenshtein
   ```

2. **Install Bioinformatics Tools (for homology partitioning)**
   ```python
   # Cell 3: Install conda and bioinformatics tools
   import condacolab
   condacolab.install()
   # Restart runtime after this step
   
   # Cell 4: Install alignment tools
   !mamba install -y -c bioconda emboss mmseqs2
   ```

3. **Data Preparation & Partitioning**
   
   **Option A: Homology-based Partitioning (Recommended)**
   ```python
   # Update the file path to your data
   df = pd.read_csv('/content/your_data.csv')  # Replace with your file path
   sequences = df['AA_without_adapter'].tolist()  # Update column name as needed
   
   # Homology-based splitting using graph partitioning
   train_idx, test_idx, valid_idx = train_test_validation_split(
       sequences,
       alignment_mode='mmseqs2',  # or 'needle' for more precise alignment
       threads=8,
       threshold=0.5,  # 50% sequence similarity threshold
       test_size=0.15,  # 15% for testing
       valid_size=0.05,  # 5% for validation
       denominator="full",
       prefilter=True
   )
   ```
   
   **Option B: Random Partitioning**
   ```python
   # Simple random splitting function (included in notebook)
   train_df, val_df, test_df = split_and_save_dataframe(
       df, 
       train_ratio=0.8, 
       val_ratio=0.05, 
       test_ratio=0.15,
       train_file='tf_library_train.csv', 
       val_file='tf_library_val.csv', 
       test_file='tf_library_test.csv'
   )
   ```

4. **Model Configuration**
   ```python
   # Key hyperparameters (modify as needed)
   model_size = (30, '150M')  # ESM-2 30-layer 150M parameter model
   model_checkpoint = f"facebook/esm2_t{model_size[0]}_{model_size[1]}_UR50D"
   
   # LoRA configuration
   rank = 8  # LoRA rank (8 recommended)
   num_layers = 2  # Number of top layers to apply LoRA (2 = last 2 layers)
   dropout_rate = 0.05  # Dropout rate for regularization
   
   # Training hyperparameters
   lr = 5e-4  # Learning rate
   batch_size = 32  # Batch size
   epochs = 50  # Number of training epochs
   max_length = 250  # Maximum sequence length
   ```

5. **Model Preparation & Training**
   ```python
   # Prepare model with LoRA and regression head
   tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
   base_model = EsmModel.from_pretrained(model_checkpoint)
   model = prepare_model(base_model, rank=rank, num_layers=num_layers, dropout_rate=dropout_rate)
   
   # Setup training components
   optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
   scheduler = CosineAnnealingWithMinLRScheduler(optimizer, num_warmup_steps, num_hold_steps, num_training_steps, min_lr=1e-5)
   
   # Initialize Weights & Biases logging
   wandb.login()  # Enter your W&B API key when prompted
   wandb.init(project="protein-sft", name="esm2-lora-sft")
   ```

6. **Dataset Setup & Training Loop**
   ```python
   # Create datasets (update file paths)
   train_dataset = TMLibrary(
       file_path="/content/tf_library_train.csv",  # Update path
       tokenizer=tokenizer, 
       max_length=max_length, 
       score_key="Avg"  # Update to match your score column name
   )
   
   val_dataset = TMLibrary(
       file_path="/content/tf_library_val.csv",  # Update path
       tokenizer=tokenizer, 
       max_length=max_length, 
       score_key="Avg"
   )
   
   # Run training
   train(model, optimizer, scheduler, train_dataloader, val_dataloader, 
         gradient_accumulation_steps=8, epochs=epochs, print_interval=100)
   ```

7. **Model Evaluation & Cross-Validation**
   ```python
   # The notebook includes comprehensive evaluation with K-fold cross-validation
   # Results are automatically logged to W&B and saved to Google Drive
   # Expected R² score: ~0.55 for the SFT model
   ```

8. **Save Trained Model**
   ```python
   # Save model checkpoint
   torch.save(model.state_dict(), '/content/drive/MyDrive/sft_model_checkpoint.pth')
   ```

#### Advanced SMPO Notebook (`tm_design_v0_2.ipynb`)

This notebook implements **Score-based Preference Optimization (SMPO)** using Direct Preference Optimization (DPO) for improved protein fitness prediction.

**Step-by-Step Execution:**

1. **Environment Setup** (Same as SFT notebook)
   - Mount Google Drive
   - Install dependencies including transformers, wandb, graph-part
   - Install bioinformatics tools if using homology partitioning

2. **Data Preparation** (Enhanced for preference learning)
   ```python
   # SMPO requires preference pairs - sequences ranked by fitness scores
   # The TMLibrary class automatically creates preference pairs from scored sequences
   
   # Update file paths to your data
   train_dataset = TMLibrary(
       file_path="/content/tm_library_train_jhp.csv",  # Update path
       tokenizer=tokenizer, 
       max_length=max_length, 
       split='train', 
       frac=0.25,  # Use 25% of data (modify as needed)
       threshold=0.0  # Minimum score threshold
   )
   
   val_dataset = TMLibrary(
       file_path="/content/tm_library_val_jhp.csv",  # Update path
       tokenizer=tokenizer, 
       max_length=max_length, 
       split='val'
   )
   ```

3. **Model Configuration for SMPO**
   ```python
   # Load pre-trained SFT model as starting point
   model = prepare_model(EsmForMaskedLM.from_pretrained(model_checkpoint), 
                        rank=8, num_layers=model_size[0], dropout_rate=0.05)
   
   # Reference model (frozen copy of original model)
   ref_model = freeze_model(EsmForMaskedLM.from_pretrained(model_checkpoint))
   
   # SMPO-specific hyperparameters
   beta = 1.0  # DPO reward scaling parameter (higher = more aggressive preference learning)
   reg_weight = 0.5  # MLM regularization weight (0.1-0.5 recommended)
   preference_weight = 0.1  # Label smoothing for preference learning
   gradient_accumulation_steps = 8
   print_interval = 100
   ```

4. **DPO Training Configuration**
   ```python
   # Key SMPO training parameters
   lr = 5e-4  # Learning rate 
   batch_size = 8  # Smaller batch size due to preference pairs
   epochs = 1  # Usually 1 epoch is sufficient for DPO fine-tuning
   max_length = 250  # Maximum sequence length
   
   # Multi-layer perceptron for adaptive beta scaling (optional)
   mnn = torch.nn.Sequential(
       torch.nn.Linear(1, 16),
       torch.nn.ReLU(),
       torch.nn.Linear(16, 1),
       torch.nn.Sigmoid()
   ).to(device)
   ```

5. **Training Loop Execution**
   ```python
   # Initialize optimizers
   optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=lr, weight_decay=1e-3)
   scheduler = CosineAnnealingWithMinLRScheduler(optimizer, num_warmup_steps, num_hold_steps, 
                                               num_training_steps, min_lr=1e-5)
   
   # Initialize logging
   wandb.init(project="protein-smpo", name=f"esm2-smpo-{beta}beta-{reg_weight}reg")
   
   # Run SMPO training
   train(model, ref_model, mnn, optimizer, scheduler, train_dataloader, val_dataloader,
         gradient_accumulation_steps=gradient_accumulation_steps, 
         epochs=epochs, 
         beta=beta, 
         reg_weight=reg_weight, 
         preference_weight=preference_weight, 
         print_interval=print_interval)
   ```

6. **Model Evaluation & Analysis**
   ```python
   # The notebook includes comprehensive evaluation:
   # - Preference accuracy tracking
   # - Reward margin analysis  
   # - MLM loss monitoring
   # - K-fold cross-validation with R² scores
   # Expected R² improvement: ~0.55 (SFT) → 0.63 (SMPO)
   ```

7. **Sparse Autoencoder (SAE) Analysis** (Optional)
   ```python
   # Train SAE for interpretability analysis
   sae = SparseAutoencoder(input_dim=hidden_dim, latent_dim=256)
   trained_sae = train_sae(sae, activations, num_epochs=50, lr=5e-4)
   
   # Save SAE for later analysis
   torch.save(sae.state_dict(), '/content/drive/MyDrive/sae_SMPO_model.pth')
   ```

### Key Configuration Notes

**Data Format Requirements:**
- **For SFT:** CSV with `Sequence` and numerical score columns
- **For SMPO:** Same format, but TMLibrary class automatically creates preference pairs based on score rankings

**Memory Optimization:**
- Use gradient accumulation (8 steps recommended) for larger effective batch sizes
- Reduce `max_length` if experiencing OOM errors
- Consider using smaller model variants (ESM-2 8M or 35M) for prototyping

**Hyperparameter Tuning:**
- **Beta (β):** Controls preference learning strength (0.1-1.0, higher = more aggressive)
- **reg_weight:** MLM regularization strength (0.1-0.5, prevents catastrophic forgetting)
- **Learning rate:** 5e-4 works well, reduce to 1e-4 for more stable training
- **LoRA rank:** 8 provides good balance of efficiency and expressiveness

**Expected Training Times:**
- **SFT (50 epochs):** 2-4 hours on A100 GPU
- **SMPO (1 epoch):** 1-2 hours on A100 GPU
- **SAE training:** 30-60 minutes additional

**Troubleshooting:**
- **OOM errors:** Reduce batch size, max_length, or use gradient checkpointing
- **Diverging loss:** Reduce learning rate or beta parameter
- **Poor convergence:** Increase training epochs or adjust scheduler parameters

**Results Reproduction:**
- Use the exact hyperparameters specified in the notebooks
- Ensure proper data partitioning (homology-based recommended)
- Run multiple seeds and report average performance
- Expected results: SFT R² ≈ 0.55, SMPO R² ≈ 0.63

## Data Requirements

All scripts expect CSV data with the following schema:

| Column Name | Type | Description |
|-------------|------|-------------|
| `Sequence` or `aa_seq` | string | Amino acid sequence using standard one-letter codes |
| `Avg_induction_score` or `score` | float | Numerical score/label for the sequence |

## Pre-trained Models

You can download pre-trained model checkpoints from our shared Google Drive folder:

**[Download Models Here](https://drive.google.com/drive/folders/1ok_UtSTIlHkVRQlu1zfoCfergeq3BjMK?usp=sharing)**

### Model Naming Convention

The model files follow this naming pattern: `p4.150M.JHP.{training_config}.{date}.pth`

**Format breakdown:**
- `p4`: Project identifier
- `150M`: Model size (150 million parameters, corresponding to ESM-2 150M)
- `JHP`: Dataset identifier 
- `{training_config}`: Training configuration, which can include:
  - `SFT`: Supervised Fine-Tuning only
  - `25out100`, `30out100`, `40out100`, `50out100`, `75out100`: SMPO training using X% of the full dataset (e.g., `30out100` means 30% of data used)
  - `noMLM`: Training without Masked Language Modeling regularization
- `{date}`: Training date in MMDDYYYY format
- `.pth`: PyTorch model checkpoint extension

**Available Models:**
- `p4.150M.JHP.SFT.8r.46r2.05122025.pth`: Base SFT model (recommended starting point)
- `p4.150M.JHP.25out100.8r.62r2.05032025.pth`: SMPO trained on 25% of dataset
- `p4.150M.JHP.30out100.8r.62r2.05032025.pth`: SMPO trained on 30% of dataset  
- `p4.150M.JHP.40out100.8r.62r2.05032025.pth`: SMPO trained on 40% of dataset
- `p4.150M.JHP.50out100.8r.62r2.05032025.pth`: SMPO trained on 50% of dataset
- `p4.150M.JHP.75out100.8r.62r2.05032025.pth`: SMPO trained on 75% of dataset
- `p4.150M.JHP.noMLM.8r.62r2.05032025.pth`: SMPO trained without MLM regularization
- `sae_SFT_05132025.pth`: Sparse Autoencoder trained on SFT model activations
- `sae_SMPO_05132025.pth`: Sparse Autoencoder trained on SMPO model activations
- `sae_SMPO_noMLM_05132025.pth`: Sparse Autoencoder trained on SMPO (no MLM) model activations

**Model Details:**
- All models are fine-tuned from `facebook/esm2_t30_150M_UR50D`
- LoRA configuration: rank 8, applied to all 30 layers, dropout 0.05
- Training hyperparameters: learning rate 5e-4, batch size 32, 50 epochs
- The specific dataset used cannot be released due to peer review requirements

## Results

Our model achieves the following R-squared (R²) performance on predicting protein function (transmembrane protein induction score):

| Model Description             | Dataset           | R² Score | Notebook Reference                               |
| ----------------------------- |------------------- | -------- | ------------------------------------------------ |
| Baseline (ESM-2 embeddings + Regressor) | TM Design Lib v0.0 |   0.20   | `notebooks/baselines/tm_design_v0_0.ipynb` (Assumed) |
| SFT ESM-2 with LoRA (This Repo)  | TM Design Lib v0.1 (JHP) |   0.55   | `notebooks/baselines/tm_design_v0_1(LRH).ipynb`  |
| Full Model (ESM-2 SFT + SMPO) | TM Design Lib v0.1 (JHP) |   0.63   | `notebooks/baselines/tm_design_v0_2.ipynb`       |

The R² of 0.55 for the SFT model is based on the K-Fold cross-validation results presented in cell ID `PQ5ZDev85F79` of the `notebooks/baselines/tm_design_v0_1(LRH).ipynb`

## Contributing

This project is licensed under the MIT License. Please feel free to open issues or submit pull requests.

To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Create a new Pull Request.
