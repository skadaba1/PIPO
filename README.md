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

This script implements the core PIPO training using Direct Preference Optimization (DPO):

```bash
# Example usage (requires modification of file paths in the script)
cd core
python main.py
```

**Note:** Before running, you need to modify the file paths in `core/main.py`:
- Update line 47: `file_path="/content/ab_library_train.csv"` to your training data path
- Update line 51: `file_path="/content/ab_library_val.csv"` to your validation data path

The script expects CSV files with columns matching the `TMLibrary` dataset class:
- `Sequence`: amino acid sequences
- `Pred_affinity`: numerical scores

#### Data Partitioning (`graphpart.py`)

Use this script to partition your dataset using homology-based or random splitting:

```bash
# For homology-based partitioning (requires graph-part, mmseqs2, emboss)
python graphpart.py
```

**Note:** The script contains functions for partitioning but requires modification to set your input file path and sequences. Key functions:
- `homology_partition()`: Creates train/validation/test splits based on sequence similarity
- `split_and_save_dataframe()`: Creates random splits

## Evaluation

### Sequence Evolution Evaluation

The repository includes a sequence evolution evaluation script:

```bash
cd evaluation
python sequence_evolution.py
```

**Requirements:** Before running, you must:
1. Train a model using one of the training methods above
2. Set the global variables in `sequence_evolution.py`:
   - `model`: Your trained model
   - `tokenizer`: ESM tokenizer
   - `device`: Training device
   - `params`: Linear model parameters
   - `linear_model`: Fitted linear regression model

### Regression Analysis

For regression analysis and model fitting:

```bash
cd evaluation/regression
python regression.py
```

This script provides functions for:
- `fit_linear_model()`: Fits linear models with K-fold cross validation
- `exponential_model()`: Exponential curve fitting

### SAE (Sparse Autoencoder) Analysis

The repository includes SAE analysis tools in `evaluation/sae/`:

```bash
cd evaluation/sae
python analyze.py          # Main analysis script
python attention_inspection.py  # Attention mechanism inspection
```

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

**Note:** Due to peer review requirements, the specific datasets used in the paper cannot be released. Users must provide their own data following the above schema.

## Pre-trained Models

You can download the SFT (Supervised Fine-Tuning) model checkpoint here:

- [p4.150M.JHP.SFT.8r.46r2.05122025.pth](https://docs.google.com/uc?export=download&id=YOUR_FILE_ID) (Please replace `YOUR_FILE_ID` with the actual shareable Google Drive link for the file `/content/drive/MyDrive/p4.150M.JHP.SFT.8r.46r2.05122025.pth` mentioned in the notebook. The specific dataset this model was trained on cannot be released due to peer review.)

This model was fine-tuned from `facebook/esm2_t30_150M_UR50D` using LoRA (rank 8, applied to all 30 layers) with a dropout of 0.05, on a private TM library dataset. Key hyperparameters included a learning rate of 5e-4, batch size 32, and 50 epochs.

## Results

Our model achieves the following R-squared (R²) performance on predicting protein function (transmembrane protein induction score):

| Model Description             | Dataset           | R² Score | Notebook Reference                               |
| ----------------------------- |------------------- | -------- | ------------------------------------------------ |
| Baseline (ESM-2 embeddings + Regressor) | TM Design Lib v0.0 |   0.20   | `notebooks/baselines/tm_design_v0_0.ipynb` (Assumed) |
| SFT ESM-2 with LoRA (This Repo)  | TM Design Lib v0.1 (JHP) |   0.55   | `notebooks/baselines/tm_design_v0_1(LRH).ipynb`  |
| Full Model (ESM-2 SFT + PIPO) | TM Design Lib v0.1 (JHP) |   0.63   | `notebooks/baselines/tm_design_v0_2.ipynb`       |

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
