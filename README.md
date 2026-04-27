# VAE-GAN Captioning

## Overview

This project implements an image captioning system that combines **VAE-GAN-based visual feature learning** with **sequence generation** for natural language description.

The model first extracts latent visual representations using a VAE-GAN fusion mechanism, and then generates captions based on these learned features.

This project demonstrates a multimodal deep learning pipeline integrating computer vision and natural language processing.

## Quick Summary

- Task: Image Captioning  
- Model: VAE-GAN + Language Model  
- Dataset: CIFAR-100 (32×32 images)  
- Result: Test Loss ≈ 1.15, Perplexity ≈ 3.1  

## Model Design

This project implements a lightweight image captioning model that combines visual representations from a diffusion-based VAE and a GAN discriminator.

The learned visual features are used to guide a simple language model for caption generation.

Since the model was trained on a local machine, the design prioritizes efficiency and lightweight computation rather than state-of-the-art performance.

## Problem Statement

Image captioning requires understanding visual content and translating it into human-readable text. This is challenging because:

* Visual features must capture semantic meaning
* Generated captions must be coherent and relevant
* The model must bridge vision and language domains

This project explores using generative models (VAE + GAN) to improve feature representation.

## Methodology

The pipeline consists of:

1. **Feature Extraction**

   * Images are encoded into latent representations using a VAE-GAN fusion model

2. **Representation Learning**

   * The model learns robust feature embeddings capturing visual semantics

3. **Caption Generation**

   * A sequence model generates captions based on the latent features

## Technologies Used

* Python
* PyTorch
* NumPy
* Deep Learning
* Computer Vision
* Natural Language Processing

## Repository Structure

```text
vae-gan-captioning/
├── ckpt/                 # Model checkpoints
├── vae_gan_fusion.py     # Core model (VAE-GAN fusion)
├── predict.py            # Caption generation
├── eval_test.py          # Evaluation script
├── utils.py              # Helper functions
├── requirements.txt
└── README.md
```

## Requirements

Python 3.8+  
PyTorch  

## How to Run

### 1. Clone repository

```bash
git clone https://github.com/Emma-NYL/vae-gan-captioning.git
cd vae-gan-captioning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run caption generation

```bash
python predict.py
```

### 4. Evaluate model

```bash
python eval_test.py
```

## Example Output

Example (illustrative):

Input Image:
[A sample image here]

Generated Caption:
"A dog running across a grassy field."

(*Example output will be updated with actual results*)

## Example Results

Below are sample input images and generated captions:

- Input images are low-resolution (32×32) samples from the dataset  
- Generated captions are produced by the trained model  

Example captions:
- "a small plate"
- "a small plate"
- "a small plate"

These results show that the model is able to map visual features to text, but the generated captions are relatively simple and lack diversity.
<img width="917" height="276" alt="Screenshot 2026-04-26 at 10 37 53 PM" src="https://github.com/user-attachments/assets/311b46d5-37b5-4464-915b-e98d9d31e1ac" />

## Training Insights

### Loss Metrics
- Test Loss ≈ 1.15  
- Perplexity ≈ 3.1  

These results suggest that the model has learned a meaningful mapping between visual features and text.

### Observations
- The model tends to generate repetitive captions (e.g., "a small plate")  
- This indicates reliance on template-based supervision  
- Limited caption diversity due to simplified training data  

### Dataset Limitation
The model was trained on CIFAR-100, which has very low resolution (32×32).  
This results in loss of fine-grained visual details and makes high-quality caption generation difficult.

## My Contribution

- Designed and implemented the VAE-GAN fusion architecture for visual feature learning  
- Built the end-to-end image captioning pipeline  
- Developed prediction and evaluation modules  
- Organized the repository structure for reproducibility and external review 

## What I Learned

* How generative models (VAE, GAN) can enhance feature representation
* Integration of computer vision and NLP in multimodal tasks
* Structuring machine learning projects for external evaluation

## Future Work

* Add dataset description and training pipeline
* Improve caption quality using attention mechanisms
* Include evaluation metrics (BLEU, CIDEr)
* Add visualization and sample outputs

## Author

Naiyue Liang
GitHub: https://github.com/Emma-NYL
