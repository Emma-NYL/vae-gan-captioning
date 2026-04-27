# VAE-GAN Captioning

## Overview

This project implements an image captioning system that combines **VAE-GAN-based visual feature learning** with **sequence generation** for natural language description.

The model first extracts latent visual representations using a VAE-GAN fusion mechanism, and then generates captions based on these learned features.

This project demonstrates a multimodal deep learning pipeline integrating computer vision and natural language processing.

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
