# GyroNovo: Error-Guided Fragment Imputation with Mass-Aware Attention for De Novo Peptide Sequencing

[![Paper](https://img.shields.io/badge/arXiv-2609.30542-b31b1b.svg)](https://arxiv.org/abs/2609.30542)
[![Code](https://img.shields.io/badge/Code-Available-brightgreen.svg)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of **GyroNovo: Error-Guided Fragment Imputation with Mass-Aware Attention for De Novo Peptide Sequencing**.

📄 **Paper:** https://arxiv.org/abs/2609.30542

📊 **Dataset:** [NovoBench on Hugging Face](https://huggingface.co/datasets/jingbo02/NovoBench)


This repository provides the code for training, inference, evaluation, and reproducible experiments with GyroNovo, along with a LIPNovo baseline.

## 🔬 Overview

**GyroNovo** is a deep learning framework for de novo peptide sequencing from tandem mass spectrometry (MS/MS) data. It addresses the challenges of sparse, noisy, and incomplete spectra through **error-guided fragment imputation** and **mass-aware attention**.

Our framework introduces two key contributions:

- **Error-Guided Fragment Imputation:** Leverages amino-acid prediction errors from the decoder to guide the recovery of missing spectral fragments. The framework uses peptide-conditioned spectrum augmentation and an easy-to-hard training curriculum, with additional imputation and decoding objectives to improve robustness to incomplete and corrupted spectral evidence.
- **Mass-Aware Attention:** Incorporates pairwise mass differences between spectral peaks into self-attention through continuous m/z rotary positional encoding, enabling the model to capture mass relationships relevant to peptide fragmentation.

GyroNovo retains an encoder-imputer-decoder architecture at inference time, without requiring additional inputs or auxiliary search procedures.

## 📊 Results

Experiments on **NovoBench** demonstrate improvements over the state-of-the-art baseline of approximately:

- **+9 percentage points** in peptide-level precision.
- **+7 percentage points** in amino-acid-level precision.

For detailed experimental results, comparisons, and ablation studies, please refer to our [paper](https://arxiv.org/abs/2609.30542).

## 💻 Available Workflows

This repository supports training, prediction, and evaluation for both GyroNovo and the LIPNovo baseline.

| Workflow | Description | Configuration |
| --- | --- | --- |
| **GyroNovo** | Decoder-guided spectrum augmentation with an easy-to-hard curriculum and continuous m/z rotary encoding. | [`configs/GyroNovo.yaml`](configs/GyroNovo.yaml) |
| **lipnovo_base** | Original LIPNovo training baseline. | [`configs/lipnovo_base.yaml`](configs/lipnovo_base.yaml) |

**Checkpoint compatibility:** This release establishes a new architecture and training-from-scratch baseline. Checkpoints from earlier repository versions are not structurally compatible. Please use checkpoints trained with this version and their corresponding workflow configurations.

## ⚙️ Installation

Clone the repository and install the required dependencies using Conda.

From the repository root, run:

```bash
conda env create -f novobench.yaml
conda activate novobench
```

**Note:** The provided Conda environment contains pinned Linux and NVIDIA CUDA dependencies. It is not intended as a portable macOS or Windows environment file.

Weights & Biases logging is optional. To disable it, set `use_wandb: false` in the selected YAML configuration. To enable it, install `wandb` and configure your credentials.

## 📂 Dataset

GyroNovo uses the **NovoBench** benchmark for training and evaluation.

The dataset is publicly available on Hugging Face:

**[Download NovoBench Dataset](https://huggingface.co/datasets/jingbo02/NovoBench)**

For additional information about the benchmark and data preparation, refer to the [NovoBench GitHub repository](https://github.com/jingbo02/NovoBench).

### Data organization

Prepare the dataset as Parquet files using the following directory structure:

```text
dataset/
├── train.parquet
├── valid.parquet
└── test.parquet
```

- `train.parquet`: Training spectra.
- `valid.parquet`: Validation spectra.
- `test.parquet`: Test spectra used for prediction and evaluation.

Training requires the training and validation files. Prediction uses `test.parquet`, while the full experiment launcher requires all three files.

### Required data format

Each row in the Parquet files represents one MS/MS spectrum and must contain the following columns:

| Column | Description |
| --- | --- |
| `mz_array` | List of fragment peak mass-to-charge (m/z) values. |
| `intensity_array` | List of peak intensities aligned with `mz_array`. |
| `precursor_mz` | Precursor mass-to-charge value. |
| `precursor_charge` | Precursor charge state. |
| `modified_sequence` or `sequence` | Peptide sequence. `modified_sequence` takes precedence when both are present. |

**Important:**
- The current data loader expects a sequence column during prediction.
- Ground-truth peptide sequences are required for evaluation.
- Amino-acid and modification tokens should match the `lipnovo.residues` mapping defined in the chosen configuration.

## 🚀 Training

### Train GyroNovo

To train GyroNovo from scratch, run:

```bash
python run.py \
  --mode train \
  --data_path /path/to/dataset \
  --config_path configs/GyroNovo.yaml
```

Model hyperparameters, batch sizes, training duration, checkpoint selection, and output directories are controlled through the YAML configuration.

With the default GyroNovo configuration, training produces:

```text
outputs/GyroNovo/
├── checkpoints/
│   └── best_model_path.txt
└── train.log
```

Model checkpoint files are saved in `outputs/GyroNovo/checkpoints/`. The selected checkpoint path is recorded in `best_model_path.txt`.

### Train the LIPNovo baseline

To train the baseline using the same pipeline, run:

```bash
python run.py \
  --mode train \
  --data_path /path/to/dataset \
  --config_path configs/lipnovo_base.yaml
```

## 🔍 Prediction and Evaluation

### Generate peptide predictions

Run de novo sequencing on `test.parquet` using a trained GyroNovo checkpoint:

```bash
python run.py \
  --mode denovo \
  --data_path /path/to/dataset \
  --ckpt_path /path/to/checkpoint.ckpt \
  --denovo_output_path outputs/GyroNovo/predictions.csv \
  --config_path configs/GyroNovo.yaml
```

The predicted peptide sequences are saved to:

```text
outputs/GyroNovo/predictions.csv
```

For the LIPNovo baseline, use `configs/lipnovo_base.yaml` and a checkpoint trained with that configuration.

### Evaluate predictions

To evaluate predicted peptides against their ground-truth sequences, run:

```bash
python novobench/metrics/evaluate.py outputs/GyroNovo/predictions.csv
```

The evaluation script reports:

- Amino-acid-level metrics.
- Peptide-level metrics.
- Modification-level metrics.
- Precision–recall area under the curve (AUC).

The evaluator also adds a `pep_matches` column to the prediction CSV.

## 🔄 Reproduce Experiments

To simplify reproducibility, we provide an experiment launcher that automatically:

1. Trains GyroNovo and the LIPNovo baseline.
2. Selects a checkpoint for each workflow.
3. Generates peptide predictions on the test set.
4. Evaluates the predictions and records the results.

Run the complete experiment pipeline using:

```bash
DATA=/path/to/dataset bash run_exp.sh
```

The launcher requires NVIDIA GPUs and schedules experiments across the available GPU pool.

### Multi-GPU execution

For example, to run the two workflows on GPUs 0 and 1, using one GPU per experiment:

```bash
DATA=/path/to/dataset GPUS="0 1" NUM_GPUS=1 bash run_exp.sh
```

### Launcher configuration

The launcher supports the following environment variables:

| Variable | Description |
| --- | --- |
| `GPUS` | Physical GPU IDs, such as `"0 1"`. Automatically detected when omitted. |
| `NUM_GPUS` | Number of GPUs per experiment. Defaults to `1`. |
| `TRAIN_BATCH_SIZE` | Global training batch size across the experiment's GPUs. |
| `TRAIN_BATCH_SIZE_PER_GPU` | Per-GPU training batch size; alternative to `TRAIN_BATCH_SIZE`. |
| `PREDICT_BATCH_SIZE` | Batch size for single-GPU prediction. |
| `N_WORKERS` | Number of data-loader workers. |
| `DATASET_NAME` | Output directory label. Defaults to the dataset directory name. |

### Experiment outputs

| Output | Location |
| --- | --- |
| Model checkpoints and training logs | `outputs/<dataset-name>/GyroNovo/` and `outputs/<dataset-name>/lipnovo/base/` |
| Test predictions | `outputs/eval/<dataset-name>/<workflow>.csv` |
| Experiment logs and evaluation metrics | `outputs/run_exp_logs/<dataset-name>/<workflow>.log` |


## 📚 Citation

If you use GyroNovo in your research, please cite our paper:

```bibtex
@misc{elmekki2026gyronovo,
  title={GyroNovo: Error-Guided Fragment Imputation with
         Mass-Aware Attention for De Novo Peptide Sequencing},
  author={Abdellah El Mekki and
          Laks V. S. Lakshmanan and
          Muhammad Abdul-Mageed},
  year={2026},
  eprint={2609.30542},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2609.30542},
  url={https://arxiv.org/abs/2609.30542}
}
```

## 🙏 Acknowledgments

We thank the authors and contributors of [LIPNovo](https://github.com/usr922/LIPNovo) and [NovoBench](https://github.com/jingbo02/NovoBench) for making their work publicly available.

Our implementation draws inspiration from LIPNovo's training pipeline and NovoBench's benchmarking framework. We also include a LIPNovo baseline and use the [NovoBench dataset](https://huggingface.co/datasets/jingbo02/NovoBench) for evaluation.

Please consider citing the original publications when using these resources.

## 📄 License

This project is distributed under the [MIT License](LICENSE).

## 📬 Contact

For questions, feedback, or research inquiries, please refer to the contact information provided in the [paper](https://arxiv.org/abs/2609.30542) or open an issue in this repository.