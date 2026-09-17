# Speech-Based Depression Classification using WavLM

Speech-based depression classification comparing an MFCC + SVM baseline, a Log-Mel + CNN model, and a fine-tuned **WavLM** speech foundation model, evaluated on the **BioCAS2024 Depression Detection Grand Challenge** dataset.

---

## Overview

Depression can affect acoustic and temporal characteristics of speech. This project classifies speech into four target labels using three approaches:

- **MFCC + SVM** — classical ML baseline
- **Log-Mel Spectrogram + CNN** — deep learning baseline
- **WavLM + MLP** — pretrained speech representations (frozen and fine-tuned)

The final model fine-tunes the last two encoder layers of `microsoft/wavlm-base-plus`.

## Objectives

- Explore audio preprocessing, MFCC, and Mel Spectrogram features
- Build classical ML and CNN baselines
- Extract and fine-tune WavLM speech representations
- Handle class imbalance and prevent participant-level data leakage
- Generate predictions for unseen test participants

## Dataset

```text
bio-cas-2024-depression-detection-grand-challenge/
├── Train.csv / Test.csv / sample_submission.csv
├── train/  (S001.mp3, S002.mp3, ...)
├── test/   (S006.mp3, S018.mp3, ...)
├── train_json/
└── test_json/
```

Four target labels (0–3), imbalanced — Label 1 has few samples. Not included in this repo due to size; evaluation emphasizes **Macro F1** and **Balanced Accuracy** over raw accuracy.

## Methodology

**Preprocessing:** resample to 16 kHz → mono → normalize → STFT / Mel Spectrogram / MFCC → 5-second segments.

**1. MFCC + SVM** — 13 MFCCs summarized by mean + std (26-D vector) → StandardScaler → RBF SVM with class weighting.

**2. Log-Mel + CNN** — conv layers, batch norm, ReLU, max/adaptive pooling, FC layers, dropout; class-weighted cross-entropy loss.

**3. WavLM (frozen)** — `wavlm-base-plus` → mean-pooled 768-D embedding per 5s segment → MLP head (Linear → ReLU → Dropout ×2) trained on frozen embeddings.

**4. WavLM (fine-tuned)** — last 2 encoder layers unfrozen + classification head, encoder at a lower LR, head at a higher LR. Chosen for the small, imbalanced dataset; reduces trainable parameters vs. full fine-tuning.

## Data Leakage Prevention

Train/validation split is performed at the **participant level** before segmentation, so no participant's segments appear in both sets.

## Participant-Level Prediction

Each participant may have multiple 5s segments. Segment-level class probabilities are averaged to produce one prediction per participant.

## Evaluation Metrics

Macro F1, Balanced Accuracy, Accuracy, per-class Precision/Recall/F1, Confusion Matrix. Macro F1 and Balanced Accuracy are prioritized because of class imbalance.

## Results

| Model | Macro F1 | Balanced Accuracy |
|---|---|---|
| MFCC + SVM | 0.3529 | 0.3518 |
| Log-Mel + CNN | 0.4762 | 0.4286 |
| Frozen WavLM + MLP | 0.4833 | 0.5000 |
| **Fine-Tuned WavLM** | **0.6451** | **0.6310** |

**Fine-Tuned WavLM — per-class F1:** Label 0: 0.9231 · Label 1: 0.0000 · Label 2: 0.8571 · Label 3: 0.8000

Label 1 was not correctly classified, consistent with severe class imbalance and few available examples — Macro F1 should be read alongside per-class results.

## Test Inference

24 test participants, ~2.16 minutes inference time, predictions saved to `submission.csv`:

```csv
Participant,Label
S018,0
S068,2
S106,3
```

Ground-truth test labels aren't available, so test predictions aren't used to compute metrics.

## Project Notebooks

| Notebook | Contents |
|---|---|
| `01_Data_Exploration_Preprocessing.ipynb` | Dataset exploration, label distribution, audio loading, STFT/Mel/MFCC extraction |
| `02_Neural_Network_Transfer_Learning.ipynb` | SVM baseline, participant-level split, CNN architecture and training |
| `03_Evaluation_Analysis.ipynb` | CNN evaluation, classification report, confusion matrix, SVM vs CNN |
| `04_Pretrained_Speech_Model_WavLM.ipynb` | WavLM embeddings, frozen/fine-tuned models, evaluation, test inference, submission |

## Repository Structure

```text
speech-depression-detection/
├── notebooks/
│   ├── 01_Data_Exploration_Preprocessing.ipynb
│   ├── 02_Neural_Network_Transfer_Learning.ipynb
│   ├── 03_Evaluation_Analysis.ipynb
│   └── 04_Pretrained_Speech_Model_WavLM.ipynb
├── models/
│   └── WavLM_Depression_FineTuned_Best.pth
├── README.md
├── requirements.txt
└── .gitignore
```

## Trained Model

Fine-Tuned WavLM Depression Classifier, based on `microsoft/wavlm-base-plus` (~351 MB) — hosted on [https://huggingface.co/anshitasingh13/speech-depression](#) due to size. Requires the `WavLMFineTuner` architecture from this repo for inference.

## Technologies Used

Python · Scikit-learn (SVM) · PyTorch (CNN, Transfer Learning) · Librosa (STFT, Mel, MFCC) · Hugging Face Transformers (WavLM) · NumPy/Pandas · Matplotlib/Seaborn · Jupyter/Colab/VS Code

## Installation

```bash
git clone https://github.com/AnshitaSingh01/speech-depression-detection
cd speech-depression-detection
pip install -r requirements.txt
```

## Requirements

```text
numpy, pandas, scikit-learn, matplotlib, seaborn, librosa, soundfile, torch, transformers
```

Full list in `requirements.txt`.

## Usage

Run notebooks sequentially: `01 → 02 → 03 → 04`

1. **Data Exploration & Preprocessing** — dataset exploration and preprocessing
2. **Model Development** — baseline SVM and CNN experiments
3. **Evaluation & Analysis** — model evaluation and comparison
4. **WavLM & Fine-Tuning** — embedding extraction, fine-tuning, evaluation, test inference

## Dataset Setup

Dataset not included. Update the path in each notebook:

```python
DATASET_PATH = "path/to/bio-cas-2024-depression-detection-grand-challenge"
```

For Colab, mount the dataset from Google Drive.

## Limitations

- Small, imbalanced dataset (especially Label 1); limited validation population
- Validation metrics may vary across participant-level splits
- Minority class not correctly classified in the reported evaluation
- Not clinically validated; results are specific to this dataset/setup
- Test predictions can't be independently evaluated (no ground truth)

## Future Work

Data augmentation for minority classes · larger participant-level cross-validation · hyperparameter optimization · audio + transcript multimodal learning · ASR and linguistic features · wav2vec 2.0 / HuBERT / Whisper encoder · attention-based fusion · unsupervised anomaly detection · cross-dataset and zero-shot evaluation

## Author

**Anshita Singh** — B.Tech, Computer Science and Engineering. Interests: AI, Machine Learning, Deep Learning, Computer Vision, Speech AI.

## Disclaimer

For educational and research purposes only. Not clinically validated — not intended to diagnose depression or inform medical decisions. Results are experimental and specific to this dataset and setup.
