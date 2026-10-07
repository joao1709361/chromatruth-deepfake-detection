# ChromaTruth: Deepfake Detection through Chromatic Reconstruction Inconsistencies

ChromaTruth is an experimental deep learning project for detecting facial image manipulations through **chromatic reconstruction inconsistencies**.

Instead of training a conventional Real vs. Fake classifier, the approach learns the chromatic structure of genuine faces. A U-Net is trained exclusively on real images to reconstruct chrominance from luminance, and reconstruction errors are then analyzed to identify manipulated images.

> **Best reported experimental AUC-ROC: ≈ 0.7627**

This project was developed as part of the **Neural Networks and Deep Learning** course of the BSc in Artificial Intelligence and Data Science at the University of Beira Interior.

---

## Overview

The central hypothesis is that genuine facial images contain natural relationships between **luminance** and **chrominance**, while digital manipulations may introduce subtle inconsistencies between them.

Images are converted from RGB to the **CIELAB color space**:

- **L** — luminance / structural information
- **a, b** — chrominance / color information

The model receives only the **L channel** and learns to reconstruct the **a and b channels**.

The reconstruction error is calculated as:

```text
E = ||ab_predicted - ab_original||₂
```

Higher or more irregular reconstruction errors may indicate chromatic inconsistencies associated with manipulation.

---

## Pipeline

```text
FFHQ real faces
      ↓
Train / test split
      ↓
Stable Diffusion inpainting
      ↓
Real / manipulated image pairs
      ↓
RGB → Lab conversion
      ↓
U-Net colorization model
      ↓
Chromatic reconstruction error
      ↓
Detection scores + heatmaps
      ↓
ROC / AUC evaluation
```

The complete workflow includes:

1. preparation of the real-face dataset;
2. generation of manipulated images using Stable Diffusion inpainting;
3. training of a U-Net using only genuine images;
4. calculation of chromatic reconstruction-error maps;
5. comparison of different detection scores;
6. generation of explainability heatmaps;
7. evaluation using ROC/AUC and threshold-based metrics.

---

## Dataset

The project uses the **Flickr-Faces-HQ (FFHQ)** dataset as the source of genuine facial images.

The original experimental setup considered:

```text
52,001 real images
41,600 training images
10,401 testing images
```

A fixed random seed (`42`) was used to make the split reproducible.

The FFHQ dataset is **not included in this repository**.

---

## Synthetic Manipulations

Manipulated images were generated using **Stable Diffusion Inpainting** through the Hugging Face Diffusers library.

A region of each selected face is masked and regenerated while attempting to preserve realistic appearance, lighting and surrounding context.

Main generation parameters used in the code include:

```text
Model: runwayml/stable-diffusion-inpainting
Inference steps: 12
Guidance scale: 5.5
Strength: 0.65
Seed: 123
```

During experimentation, different numbers of generated images were evaluated:

```text
3,000
6,000
12,000
```

The final reported experiment used **6,000 manipulated images**.

---

## Colorization Model

The reconstruction model is a lightweight **U-Net implemented in PyTorch**.

### Input

```text
L channel
1 × 256 × 256
```

### Output

```text
a and b channels
2 × 256 × 256
```

The architecture contains an encoder-decoder structure with:

- convolutional blocks;
- batch normalization;
- max pooling;
- transposed convolutions;
- skip connections;
- `tanh` output for normalized chrominance prediction.

The current training implementation uses:

- **Smooth L1 Loss**
- **AdamW**
- learning-rate scheduling;
- early stopping;
- automatic mixed precision when CUDA is available;
- automatic saving of the best checkpoint.

---

## Detection and Explainability

After training, the U-Net reconstructs the chrominance of real and manipulated images.

The predicted chrominance is compared with the original image to produce a pixel-level reconstruction-error map.

Several statistics were explored to transform these error maps into detection scores, including:

- high-percentile reconstruction error;
- mean of the highest-error pixels;
- error ratios;
- peak ratios;
- fused scores;
- a combined `score_mix`.

The final reported configuration used:

```text
PHIGH = 85
λ = 0.9
```

The project also generates **heatmaps** showing regions with stronger chromatic inconsistencies.

Experiments with masks restricted to facial regions were also performed, but they reduced performance. This suggested that useful chromatic information may also exist around hair, facial boundaries and nearby background regions.

---

## Results

The strongest reported experimental configuration achieved:

| Metric / Setting | Result |
|---|---:|
| AUC-ROC | **≈ 0.7627** |
| High percentile | **85** |
| Score weight λ | **0.9** |
| Manipulated images | **6,000** |

Some of the main experimental observations were:

- intermediate percentiles performed better than extremely high percentiles;
- restricting the analysis to facial masks reduced performance;
- manipulated images tended to produce stronger and less homogeneous chromatic reconstruction errors;
- increasing the manipulated set from 3,000 to 6,000 improved performance;
- increasing it further to 12,000 did not provide consistent additional gains.

Threshold-based evaluation is implemented separately and calculates:

```text
Accuracy
Precision
Recall
F1-score
F2-score
Confusion Matrix
```

---

## Repository Structure

```text
chromatruth-deepfake-detection/
│
├── README.md
├── .gitignore
├── TP2_ChromaTruth_pipeline.ipynb
│
├── src/
│   ├── build_pairs.py
│   ├── config.py
│   ├── detect_and_xai.py
│   ├── evaluate_threshold.py
│   ├── generate_fakes_inpainting.py
│   ├── logger_utils.py
│   ├── make_splits.py
│   ├── test_auc_coluna.py
│   └── train_colorizer.py
│
└── docs/
    └── Trabalho_Pratico2_RNDL.pdf
```

The dataset, generated images, model checkpoints and large experimental outputs are intentionally excluded from the repository.

---

## Jupyter Notebook

The full project pipeline can also be explored through:

```text
TP2_ChromaTruth_pipeline.ipynb
```

The notebook brings together the main stages of the project in a single document and is useful for understanding the overall experimental workflow.

---

## Running the Project

### Main dependencies

```text
Python
PyTorch
NumPy
OpenCV
Matplotlib
Pillow
Diffusers
Transformers
Accelerate
```

Install the main dependencies with:

```bash
pip install torch torchvision
pip install numpy opencv-python matplotlib pillow
pip install diffusers transformers accelerate
```

A CUDA-compatible GPU is strongly recommended for Stable Diffusion generation and model training.

### Expected data structure

The current scripts use the `src/` directory as the project root, so the expected local structure is:

```text
src/
├── data/
│   ├── real_all/
│   ├── real_train/
│   ├── real_test/
│   ├── fake_test/
│   └── masks/
│
└── outputs/
```

Place the original FFHQ images inside:

```text
src/data/real_all/
```

### Execution order

From the repository root:

```bash
python src/make_splits.py
python src/generate_fakes_inpainting.py
python src/train_colorizer.py
python src/build_pairs.py
python src/detect_and_xai.py
python src/test_auc_coluna.py
python src/evaluate_threshold.py
```

---

## Technologies

**Deep Learning:** PyTorch, U-Net  
**Generative AI:** Stable Diffusion, Hugging Face Diffusers  
**Computer Vision:** OpenCV, CIELAB color space, image processing  
**Evaluation:** ROC, AUC, Accuracy, Precision, Recall, F1/F2  
**Explainability:** chromatic reconstruction-error heatmaps  
**Programming:** Python, NumPy, Matplotlib

---

## Limitations

ChromaTruth is an **experimental academic project**, not a production-ready forensic detector.

The strongest reported scoring strategy was evaluated using corresponding real/manipulated image pairs. The reported AUC should therefore be interpreted in the context of this paired experimental setup rather than as the performance of a completely independent single-image detector.

The experiments also focus primarily on manipulations generated through Stable Diffusion inpainting and on FFHQ facial images.

Generalization to other datasets, face swaps, GAN-generated images, different diffusion models and video deepfakes remains to be evaluated.

---

## Future Work

Potential extensions include:

- development of a fully independent single-image detection score;
- evaluation on external deepfake benchmarks;
- testing additional manipulation techniques;
- comparison with supervised deepfake classifiers;
- combining chromatic features with texture or frequency-domain information;
- extension to video and temporal consistency analysis.

---

## Academic Context

**João Craveiro**  
BSc in Artificial Intelligence and Data Science  
University of Beira Interior  
Neural Networks and Deep Learning — 2025/2026

The complete academic report is available in:

```text
docs/Trabalho_Pratico2_RNDL.pdf
```

---

## Author

**João Craveiro**

[GitHub](https://github.com/joao1709361) · [LinkedIn](https://www.linkedin.com/in/joaodiogocraveiro/)
