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

The repository contains several Jupyter notebooks:

- `notebooks/baselines/tm_design_v0_1(LRH).ipynb`: Main training and evaluation notebook
- `notebooks/tm_design_v0_2.ipynb`: Advanced model implementation

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
