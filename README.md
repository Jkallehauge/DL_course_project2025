Here is a comprehensive `README.md` draft based on your project report and the code snippet provided.

I have structured it to highlight the technical depth (Multi-Task Learning, Adaptive Loss) and the narrative of your results (The "Glioma Gap" and data cleaning).

-----

## Brain Tumor Classification: Transfer Learning vs. Multi-Task U-Net

  

This repository contains the implementation for the "Deep Learning in Visual Recognition" course project (2025). It investigates the classification of MRI scans into four categories (Meningioma, Glioma, Pituitary Adenoma, and No Tumor) using two distinct approaches: standard Transfer Learning and a custom Hierarchical Multi-Task Learning (MTL) architecture.

**Author:** Jesper F. Kallehauge  
**Institution:** Danish Centre for Particle Therapy, Aarhus University Hospital

-----

### 📖 Project Overview

Accurate tumor classification is critical for treatment planning. [cite_start]This project aims to diagnose tumor types solely from MRI images, simulating scenarios where biopsies are too risky[cite: 18].

The project explores:

1.  [cite_start]**Transfer Learning:** Benchmarking diverse backbones (Swin Transformer, ResNeXt, MobileNet, etc.)[cite: 112].
2.  **Multi-Task Learning:** A U-Net based architecture that simultaneously performs:
      * Image Reconstruction (Self-Supervision)
      * Binary Tumor Detection (Tumor vs. No Tumor)
      * [cite_start]Subtype Classification (Specific Tumor Type)[cite: 125].
3.  [cite_start]**Data Curation:** Analysis of latent space (UMAP) to identify and resolve dataset duplicates, improving accuracy from \~77% to 97%[cite: 270].

-----

### 🧠 Architectures

#### 1\. Transfer Learning (Warm-Start)

We finetuned several ImageNet-pretrained models to establish a baseline. [cite_start]The **Swin Transformer** achieved the highest validation accuracy, though **MobileNetV2** showed better generalization initially[cite: 183].

  * *Models tested:* VGG16, ResNeXt50, Swin-B, ViT, MobileNetV2/V3.

#### 2\. Hierarchical Multi-Task U-Net

We implemented a custom U-Net backbone that optimizes a joint objective function.

  * [cite_start]**Shared Encoder:** 4 downsampling blocks with increasing channel depth (64 → 1024)[cite: 148, 152].
  * [cite_start]**Reconstruction Decoder:** Restores the image to learn robust anatomical features ($L_{recon}$)[cite: 164].
  * **Dual Classification Heads:**
      * *Binary Head:* Detects presence of tumor ($L_{bin}$).
      * [cite_start]*Subtype Head:* Classifies the specific type ($L_{sub}$), masked to ignore healthy inputs[cite: 131].

-----

### ⚖️ Adaptive Loss Balancing (EMA)

A key feature of this implementation is the **Dynamic Loss Weighting**. [cite_start]Instead of static hyperparameters, we use Exponential Moving Averages (EMA) to normalize loss magnitudes, ensuring no single task dominates the gradient descent[cite: 135].

The loss calculation (found in `epoch_run`) is defined as:

$$L_{total} = \frac{L_{recon}}{\sigma_{recon}} + \frac{L_{cls}}{\sigma_{cls}}$$

Where $\sigma$ represents the running EMA of the respective loss.

-----

### 📊 Results & The "Glioma Gap"

| Dataset Version | Model | Test Accuracy |
| :--- | :--- | :--- |
| **Dataset 1 (Raw)** | Swin Transformer | \~77.0% |
| **Dataset 1 (Raw)** | MTL U-Net | \~77.7% |
| **Dataset 2 (Cleaned)** | **MTL U-Net** | **97.33%** |

[cite_start]**Key Finding:** Initial experiments showed a persistent failure to classify **Gliomas** (Recall ≈ 0.33)[cite: 186]. UMAP analysis revealed a distributional shift between validation and test samples for Gliomas. [cite_start]After removing 1,113 duplicate images found in the combined datasets, performance stabilized significantly[cite: 270].

*Figure: UMAP projection showing the shift in Glioma distribution (blue x vs blue o) before cleaning.*

-----

### 🛠️ Installation & Usage

1.  **Clone the repository**

    ```bash
    git clone https://github.com/Jkallehauge/DL_course_project2025.git
    cd DL_course_project2025
    ```

2.  **Install dependencies**

    ```bash
    pip install torch torchvision numpy matplotlib scikit-learn
    ```

3.  **Data Setup**

      * Dataset 1: [Kaggle Link 1](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data)
      * Dataset 2: [Kaggle Link 2](https://www.google.com/search?q=https://www.kaggle.com/datasets/rishantenis/brain-tumor-mri-dataset/data)
      * *Note: Ensure you run the cleaning script to remove duplicates.*

4.  **Running the Training**
    *(Update with your specific script names)*

    ```bash
    # Run transfer learning baseline
    python train_transfer.py --model swin_b

    # Run Multi-Task Learning
    python train_mtl.py --epochs 100 --use_amp
    ```

-----

### 📂 Repository Structure

```
├── data/                  # Data loaders and preprocessing scripts
├── models/
│   ├── transfer_nets.py   # Definitions for Swin, ResNext, etc.
│   └── unet_mtl.py        # Custom U-Net with multi-heads
├── utils/
│   ├── loss.py            # EMA Loss weighting logic
│   └── visualization.py   # UMAP and Grad-CAM plotting
├── notebooks/             # Jupyter notebooks for analysis
├── train.py               # Main training loop
└── README.md
```

-----

### 📝 References

1.  [cite_start]Danish Neuro-Oncology Group (DNOG) Guidelines[cite: 274, 296].
2.  Louis, D. N., et al. [cite_start]"The 2021 WHO Classification of Tumors of the Central Nervous System."[cite: 297].
3.  Ronneberger, O., et al. [cite_start]"U-Net: Convolutional Networks for Biomedical Image Segmentation."[cite: 299].
4.  Kendall, A., et al. [cite_start]"Multi-Task Learning Using Uncertainty to Weigh Losses."[cite: 300].
