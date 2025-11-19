UNetHierMTL – Multi-Task U-Net for Tumor Segmentation and Hierarchical Classification

This repository contains an implementation of UNetHierMTL, a multi-task 2D U-Net designed for:

Image reconstruction (Auto Encoder)

Binary tumor detection (tumor vs. no-tumor)

Subtype classification (glioma, meningioma, pituitary)

The model combines a standard encoder–decoder U-Net architecture with a shared latent representation extracted from the bottleneck. This latent vector feeds two classification heads while the decoder reconstructs the segmentation map.
