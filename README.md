# VIZMed: Vision-Integrated Zero-Shot Medical AI  

## Overview  
VIZMed is a research project aimed at improving zero-shot learning for medical imaging by expanding upon the MedCLIP model. The goal is to enhance its ability to interpret medical images and reports using multimodal data. This project forms part of my thesis and explores how training on diverse datasets can improve generalization in medical vision-language models.  

## Objectives  
- Expand MedCLIP’s capabilities to improve zero-shot classification and retrieval.  
- Train the model on a combination of **CheXpert**, **MIMIC-CXR**, and **PadChest** datasets.  
- Improve multimodal understanding of medical images and textual reports.  
- Evaluate performance against existing benchmarks and propose enhancements.  

## Methodology  
- **Data Preprocessing**: Curate, clean, and prepare CheXpert, PadChest, and MIMIC-CXR datasets.  
- **Model Training**: Fine-tune MedCLIP using a combination of supervised and contrastive learning techniques.  
- **Evaluation**: Measure zero-shot performance using various medical classification and retrieval tasks.  
- **Optimization**: Experiment with architectural improvements and hyperparameter tuning.  

## Datasets  
- **CheXpert**: A large labeled dataset of chest X-rays.  
- **MIMIC-CXR**: A collection of chest radiographs with associated clinical reports.
- **PadChest**: A dataset containing a wide variety of radiology images and text reports.  

## Getting Started  
1. Clone this repository:  
   ```sh
   git clone https://github.com/ZaneBrackley/VIZMed.git
   cd VIZMed
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
3. Download and preprocess the datasets (instructions in data/README.md).
4. Start training using
   ```sh
   python train.py --config configs/train_config.yaml

## Future Work
- Extend zero-shot learning beyond chest X-rays to other medical modalities.
- Explore the integration of transformer-based architectures for improved text-image alignment.
- Assess real-world applicability through clinical validation studies.

## Acknowledgements
This work is built upon MedCLIP, leveraging its foundational approach to medical vision-language learning. The original codebase can be found [here](https://github.com/RyanWangZf/MedCLIP).
