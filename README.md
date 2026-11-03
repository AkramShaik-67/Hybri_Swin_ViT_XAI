# Hybrid Swin Transformer and Vision Transformer Framework with Explainable AI for Deepfake Detection

## Overview

This project develops a hybrid deepfake image detector by combining
**Swin Transformer V2** and **Vision Transformer (ViT)** feature
representations. The fused features are passed to a shared
classification head to predict the image class. An **occlusion-based
Explainable AI (XAI)** method is also included to visualize image
regions that support or oppose a selected prediction.

The goal is to explore whether combining hierarchical, window-based
visual features from Swin Transformer with global token-based
representations from ViT can support deepfake detection, while providing
visual evidence that helps users inspect model decisions.

>  Model performance must be measured on the test set; no accuracy or other
> result is claimed in this README.

## Project Title

**Hybrid Swin Transformer and Vision Transformer Framework with
Explainable AI for Deepfake Detection**

## Dataset

This project uses the Kaggle **Deepfake** dataset:

- **Dataset page:**
  https://www.kaggle.com/datasets/shaikakram67/deepfake

The notebook downloads the dataset using `kagglehub` and expects this
directory structure:

```text
Dataset/
├── Train/
│   ├── Fake/
│   └── Real/
├── Validation/
│   ├── Fake/
│   └── Real/
└── Test/
    ├── Fake/
    └── Real/
```

The actual class-folder names are read from the dataset using
`torchvision.datasets.ImageFolder`. The notebook checks that class
ordering is consistent across the training, validation, and test sets.

## Methodology

### 1. Data loading and preprocessing

- Download the training, validation, and in-domain test dataset with
  `kagglehub`; load images from its `Train`, `Validation`, and `Test`
  class folders using `torchvision.datasets.ImageFolder`.
- Apply OpenCV bilateral filtering to RGB images before tensor conversion
  (enabled; diameter **9**, color sigma **100**, spatial sigma **100**).
  This filter is applied to training, validation, and test images.
- For training, use random resized cropping to **256 × 256** (scale
  0.9–1.0), random horizontal flips (0.5), rotations (±10°), color
  jitter (brightness/contrast 0.15, saturation 0.1, hue 0.03), and
  random erasing (probability 0.15).
- For validation and test, resize directly to **256 × 256** without
  random augmentation.
- Convert images to tensors and normalize using ImageNet mean
  `[0.485, 0.456, 0.406]` and standard deviation
  `[0.229, 0.224, 0.225]`. Random erasing is applied after normalization.
- Use batch size **16**. Check that class names and their ordering match
  across the train, validation, and in-domain test splits.

### 2. Hybrid feature extraction

The notebook creates pretrained backbones with `timm` and removes their
standalone classification heads so their feature sequences can be used:

- **Swin Transformer V2 Tiny** (`swinv2_tiny_window16_256`) extracts
  hierarchical, spatial/token features.
- **Vision Transformer Base** (`vit_base_patch16_224`, configured with
  `img_size=256`) extracts global patch-token features.

A bidirectional cross-attention fusion module projects both feature
streams to a shared **384-dimensional** space and uses **6 attention
heads**. Swin tokens query ViT keys and values, while ViT tokens query
Swin keys and values. Residual connections, dropout, and layer
normalization refine both streams. The module mean-pools each refined
stream, concatenates the two pooled vectors, and passes the fused
representation to the classifier. This is token-level cross-attention
fusion, rather than direct concatenation of the original backbone
embeddings.

### 3. Classification head

The fused feature vector is passed through a shared classifier
consisting of layer normalization, fully connected layers, GELU
activations, dropout, and a final output layer sized to the number of
dataset classes.

### 4. Training strategy

The notebook includes the following training components:

- Cross-entropy loss with label smoothing.
- AdamW optimizer with separate learning rates for backbone and
  classifier parameters.
- Warmup followed by cosine-style learning-rate scheduling.
- Automatic mixed precision when CUDA is available.
- Gradient accumulation and gradient clipping.
- Exponential Moving Average (EMA) model weights for validation.
- Best-validation-checkpoint tracking and training-history logging.

The configured maximum training duration is **50 epochs**. Actual
training time depends on the hardware, dataset size, and runtime
environment.

### 5. Evaluation

The notebook calculates and plots evaluation results, including:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Classification report
- Confusion matrix
- ROC curve / AUC analysis
- Training and validation loss/accuracy curves

Report the values produced by your own completed run. Results can vary
with the dataset version, class distribution, random seed, software
versions, and hardware.

### 6. Cross-dataset evaluation`n`nAfter training, the notebook evaluates the best validation checkpoint on`nthe test directories of the two external datasets listed above. It uses`nthe deterministic test transform and maps common folder labels (`Fake` /`n`Deepfake` to fake; `Real` / `Original` to real) into the training`nclass indices. The current cross-dataset cell reports image count, loss,`nand accuracy for each dataset. It does not fine-tune on those external`ntest sets. Record the outputs from your completed run; this README does`nnot claim specific cross-dataset scores.`n`nFor a valid zero-shot generalization evaluation, keep these external test`nsets out of training and model selection, verify the folder label mapping,`nand document any filtering or sampling performed.`n`n### 7. Single-image prediction

After training, the notebook demonstrates inference on an individual
image and displays the predicted class and its softmax confidence.

### 8. Explainable AI with occlusion

The XAI section masks small image patches and measures how the selected
class probability changes. The patch scores are combined into a heatmap
and visualized alongside the input image.

- Regions whose occlusion reduces the target-class probability are
  treated as evidence supporting that class.
- Regions whose occlusion increases the target-class probability can
  be interpreted as evidence opposing that class.
- The visualization is an approximate model-evidence explanation. It
  should not be interpreted as a definitive localization of
  manipulated pixels or as proof that the model uses valid forensic
  cues.

## Repository / Notebook Contents

The main project notebook is `hybrid-swin-vit-xai.ipynb`. It contains
cells for:

1.  Dataset download and path configuration
2.  Environment and device checks
3.  Dataset inspection and sample visualization
4.  Data augmentation and data loaders
5.  Swin Transformer and ViT feature extractors
6.  Feature fusion and classification
7.  Training, validation, and checkpoint tracking
8.  Test-set prediction and evaluation metrics
9.  Training curves and single-image inference
10. Occlusion-based XAI visualization

The notebook also saves a model checkpoint named `best_hybrid_model.pth`
when the corresponding save cell is executed.

## Requirements

A Python environment with the following libraries is needed:

- Python 3.10 or a compatible version supported by the installed
  PyTorch packages
- PyTorch
- torchvision
- timm
- kagglehub
- NumPy
- Matplotlib
- OpenCV (`opencv-python`)
- scikit-learn
- Pillow

Install the main dependencies with:

```bash
pip install torch torchvision timm kagglehub numpy matplotlib opencv-python scikit-learn pillow
```

For GPU training, install the PyTorch build compatible with your operating
system and CUDA environment by following the official
[PyTorch installation guide](https://pytorch.org/get-started/locally/).
A CUDA-capable GPU is recommended for training speed, but CPU execution
may be possible and will be slower.

## How to Run

### Option A: Kaggle Notebook

1. Open or create a notebook on [Kaggle](https://www.kaggle.com/code).
2. Add the Deepfake dataset to the notebook, or allow the notebook's
   `kagglehub.dataset_download("shaikakram67/deepfake")` cell to download it.
3. Enable GPU acceleration in notebook settings if available.
4. Run the dataset setup cell and verify that the paths point to
   `Dataset/Train`, `Dataset/Validation`, and `Dataset/Test`.
5. Run the notebook cells in order.
6. Review training and validation curves, then evaluate the model on the
   test split.
7. Attach both external datasets and run the cross-dataset evaluation cell.`n8. Run single-image prediction and the XAI cells to inspect an example.

### Option B: Google Colab

1. Open `hybrid-swin-vit-xai.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → GPU** if a GPU is available.
3. Run the dependency installation cell or install any missing packages.
4. Run the dataset-download cell. If you instead mount or upload the
   dataset manually, update the directory variables accordingly.
5. Run the notebook cells in order, from setup and data loading through
   training, evaluation, prediction, and XAI.

### Dataset Path Configuration

The notebook downloads the dataset using `kagglehub` and sets the paths
as follows:

```python
path = kagglehub.dataset_download("shaikakram67/deepfake")
train_dir = f"{path}/Dataset/Train"
val_dir   = f"{path}/Dataset/Validation"
test_dir  = f"{path}/Dataset/Test"
```

If you use a manually mounted dataset, replace these paths with the real
locations. Confirm that all three paths exist before starting training,
and that class names and class ordering match across all three splits.

### Runtime notes

- The two pretrained transformer backbones are computationally
  demanding. A GPU with sufficient memory is recommended.
- Training may take a long time, especially on CPU. Monitor the
  runtime and GPU memory before restarting a long-running cell.
- The first `kagglehub` download and pretrained model initialization
  may require an internet connection.
- If Kaggle access requires authentication in your environment,
  configure Kaggle access according to Kaggle's documentation.
- If a notebook runtime disconnects, verify whether the downloaded
  dataset and saved checkpoint still exist before resuming.

## Model and Training Configuration

The following settings are defined in the current notebook. The notebook
uses a **hybrid Swin Transformer V2 + Vision Transformer** architecture,
not a standalone ViT model.

| Setting                       | Value in notebook                                               |
| ----------------------------- | --------------------------------------------------------------- |
| Model architecture            | SwinV2 Tiny + ViT Base with bidirectional cross-attention       |
| Swin backbone                 | `swinv2_tiny_window16_256`                                      |
| ViT backbone                  | `vit_base_patch16_224`, configured for 256-pixel input          |
| Pretrained weights            | Enabled for both backbones                                      |
| Input image size              | 256 × 256                                                       |`n| Bilateral filtering            | Enabled; diameter 9, color/space sigma 100                      |`n| Cross-dataset evaluation       | Two external Kaggle test sets; no fine-tuning                   |
| Batch size                    | 16                                                              |
| Gradient accumulation         | 2 steps                                                         |
| Maximum epochs                | 50                                                              |
| Loss function                 | Cross-entropy with label smoothing (`0.08`)                     |
| Optimizer                     | AdamW                                                           |
| Backbone learning rate        | `2e-5`                                                          |
| Classifier-head learning rate | `2e-4`                                                          |
| Weight decay                  | `1e-4` for eligible parameters; biases/norm parameters excluded |
| Learning-rate scheduler       | Warmup followed by cosine-style schedule                        |
| Compute device                | CUDA GPU when available; otherwise CPU                          |
| Mixed precision               | Enabled when CUDA is available                                  |
| Gradient clipping             | Maximum gradient norm of `1.0`                                  |
| Model averaging               | Exponential Moving Average (EMA) used during validation         |

These are configuration values, not a guarantee of model performance.
Actual runtime and results depend on the dataset, hardware, and completed
training run.

## Evaluation Metrics

The notebook reports or visualizes the following evaluation outputs:

- **Accuracy:** Fraction of test images classified correctly.
- **Weighted precision:** Precision averaged across classes, weighted by
  the number of true samples in each class.
- **Weighted recall / sensitivity:** Recall averaged across classes.
- **Weighted F1-score:** Weighted F1 measure across classes.
- **Classification report:** Per-class precision, recall, F1-score, and
  support.
- **Confusion matrix:** Shows correct and incorrect predictions by class.
- **ROC curve / AUC:** Included where supported by the notebook's
  prediction/evaluation cells.
- **Training curves:** Training and validation loss and accuracy by epoch.

No particular accuracy is guaranteed. Report the metrics produced by your
actual completed run. Use the test split for final evaluation rather than
repeatedly tuning the model against it.

## Explainable Artificial Intelligence (XAI)

The notebook uses **occlusion sensitivity** to inspect predictions from
the hybrid model. It covers patches of the input image and measures how
the probability of the selected class changes when each patch is replaced
with the normalized-space mean-color value. Patch scores are combined into
an evidence map and displayed with the original image.

The current notebook's XAI function uses a patch size of 32 pixels and a
stride of 16 pixels by default. Positive scores indicate that occluding a
region reduced the selected class probability (evidence supporting that
class); negative scores indicate that occlusion increased the probability
(evidence opposing that class).

**Interpretation:** This is an approximate, image-level explanation of
model behavior. It is not a pixel-level ground-truth manipulation mask and
does not prove that highlighted areas contain forged artifacts.

## Model Checkpoint

The notebook saves the checkpoint as:

```text
best_hybrid_model.pth
```

The checkpoint contains the model state dictionary, class names, input
image size, best validation accuracy, and best validation loss. Ensure the
file exists in the current working directory (or use its full path) before
loading it. Recreate the same hybrid model architecture and class count
before loading the saved state dictionary.

## Reproducibility and Practical Notes

- Ensure class names and class ordering are consistent between the
  training, validation, and test folders.
- Keep training, validation, and test data separate to reduce data
  leakage. If images originate from the same source video or identity,
  consider splitting by source rather than randomly by image.
- Confirm that the checkpoint corresponds to the intended best-validation
  model and that the checkpoint-saving cell ran successfully.
- The first dataset download and pretrained model initialization require
  internet access.
- Transformer backbones can use substantial GPU memory. Reduce batch size
  if you encounter out-of-memory errors, and remember that this changes
  the configuration from the documented default.
- Performance can vary on unseen datasets, compression levels, and
  manipulation methods. Treat predictions as model estimates rather than
  definitive forensic proof.

## Repository Structure

```text
project/
├── README.md
├── hybrid-swin-vit-xai.ipynb
├── best_hybrid_model.pth   # generated after the checkpoint cell runs
└── Dataset/                # downloaded or mounted dataset (optional)
    ├── Train/
    ├── Validation/
    └── Test/
```

This is an example layout. The dataset and checkpoint may be stored
elsewhere and generally should not be committed to version control if they
are large or subject to dataset licensing terms.

## Output Artifacts

Depending on which notebook cells finish successfully, outputs include:

- Printed dataset counts and class names
- Training and validation loss/accuracy history
- Classification report and confusion matrix
- ROC/AUC plots
- Single-image prediction with confidence
- Occlusion-based XAI heatmap
- `best_hybrid_model.pth` checkpoint

Generated plots and files are available only if the relevant cells have
run successfully and the runtime preserves their outputs.

## Citation

If you use the dataset, cite or acknowledge its Kaggle page and its
creator according to the dataset's stated terms:

- Training and in-domain evaluation: https://www.kaggle.com/datasets/shaikakram67/deepfake
- Cross-dataset evaluation: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
- Cross-dataset evaluation: https://www.kaggle.com/datasets/shivamardeshna/real-and-fake-images-dataset-for-image-forensics



