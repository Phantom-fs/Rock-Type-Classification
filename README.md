# Rock-Type-Classification

This repository contains the implementation of deep learning models for geological image segmentation, specifically designed for rock type identification and classification. The models utilize transfer learning and fine-tuning of state-of-the-art architectures like ResNet, Inception, DenseNet, MobileNet, and EfficientNet to achieve high accuracy in classifying 19 distinct rock types. The study demonstrates the effectiveness of these models in enhancing geological image analysis and segmentation.

# Paper

### [Advancing geological image segmentation: Deep learning approaches for rock type identification and classification](https://www.sciencedirect.com/science/article/pii/S2590197424000399)

- Published in the Elsevier journal: **Applied Computing and Geosciences**

### Abstract:
<p align="justify">This study aims to tackle the obstacles linked with geological image segmentation by employing sophisticated deep learning techniques. Geological formations, characterized by diverse forms, sizes, textures, and colors, present a complex landscape for traditional image processing techniques. Drawing inspiration from recent advancements in image segmentation, particularly in medical imaging and object recognition, this research proposed a comprehensive methodology tailored to the specific requirements of geological image datasets. To establish the dataset, a minimum of 50 images per rock type was deemed essential, with the majority captured at the University of Las Palmas de Gran Canaria and during a field expedition to La Isla de La Palma, Spain. This dual-source approach ensures diversity in geological formations, enriching the dataset with a comprehensive range of visual characteristics. The study involves the identification of 19 distinct rock types, each documented with 50 samples, resulting in a comprehensive database containing 950 images. The methodology involves two crucial phases: initial preprocessing of the dataset, focusing on formatting and optimization, and subsequent application of deep learning models—ResNets, Inception V3, DenseNets, MobileNets V3, and EfficientNet V2 large. Preparing the dataset is crucial for improving both the quality and relevance, thereby to ensure the optimal performance of deep learning models, the dataset was preprocessed. Following this, transfer learning or more specifically fine-tuning is applied in the subsequent phase with ResNets, Inception V3, DenseNets, MobileNets V3, and EfficientNet V2 large, leveraging pre-trained models to enhance classification task performance. After fine-tuning eight deep learning models with optimal hyperparameters, including ResNet101, ResNet152, Inception-v3, DenseNet169, DenseNet201, MobileNet-v3-small, MobileNet-v3-large, and EfficientNet-v2-large, comprehensive evaluation revealed exceptional performance metrics. DenseNet201 and InceptionV3 attained the highest accuracy of 98.49% when tested on the original dataset, leading in precision, sensitivity, specificity, and F-score. Incorporating preprocessing steps further improved results, with all models exceeding 97.5% accuracy on the preprocessed dataset. In K-Fold cross-validation (k = 5), MobileNet V3 large excelled with the highest accuracy of 99.15%, followed by ResNet101 at 99.08%. Despite varying training times, models on the preprocessed dataset showed faster convergence without overfitting. Minimal misclassifications were observed, mainly among specific classes. Overall, the study’s methodologies yielded remarkable results, surpassing 99% accuracy on the preprocessed dataset and in K-Fold cross-validation, affirming the efficacy in advancing rock type understanding.</p>

### Links:

- **[DOI](https://doi.org/10.1016/j.acags.2024.100192)**
- **[Journal Homepage](https://www.sciencedirect.com/journal/applied-computing-and-geosciences)**
- **[Scopus](https://www.scopus.com/sourceid/21101075727)**

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [License](#license)

## Introduction

The goal of this project is to advance the geological image segmentation process by employing deep learning techniques. Rock types are identified using a dataset consisting of 19 distinct rock types, each with 50 samples. The repository contains code to preprocess the dataset, data splitting, finding optimal Hyperparameters, apply transfer learning to deep learning models for classification tasks, and K-fold cross-validation.

## Dataset

The dataset consists of **950 images** of 19 different rock types, captured primarily at the University of Las Palmas de Gran Canaria and during fieldwork in La Isla de La Palma, Spain. Each image represents one of the following rock types:

- Aphanitic Basalt
- Volcanic Tuff
- Andesite
- Volcanic Scoria
- Vesicular Basalt
- Granodiorite
- ... (complete list in the paper)

The dataset was expanded using techniques such as **image tiling** and **data augmentation**, resulting in a total of 3,800 images.

## Methodology

The workflow is divided into two phases:
1. **Preprocessing the dataset** (image tiling, normalization, resizing, etc.)
2. **Model training** using transfer learning, leveraging pre-trained models such as:
   - ResNet101, ResNet152
   - InceptionV3
   - DenseNet169, DenseNet201
   - MobileNet V3 Small, MobileNet V3 Large
   - EfficientNet V2 Large

Each model was fine-tuned and optimized using a grid search for hyperparameter tuning.

## Preprocessing

Preprocessing steps include:
- **Image tiling**: Dividing large images into smaller tiles to increase dataset size and diversity.
- **Normalization**: Standardizing pixel intensity to improve model convergence.
- **Resizing**: Adjusting image dimensions to match model input requirements (e.g., 224x224 or 299x299).

## Models

The repository contains code for fine-tuning and evaluating several deep learning models using the **PyTorch** framework. The following models are included:

- ResNet101, ResNet152
- Inception V3
- DenseNet169, DenseNet201
- MobileNet V3 (Small and Large)
- EfficientNet V2 Large


## Results

- **Best-performing model**: MobileNet V3 Large
- **Accuracy**: 99.15% in K-Fold cross-validation
- **Comprehensive metrics**: Precision, Sensitivity, Specificity, and F-Score are all above 98% on the preprocessed dataset.
- **K-Fold Cross-Validation**: All models achieved over 98% accuracy in 5-fold cross-validation.

For detailed model performance and evaluation metrics (precision, sensitivity, specificity, F1-score), refer to the results in the paper.


## Installation

### Usage

Clone the repository:

```bash
git clone https://github.com/Phantom-fs/Rock-Type-Classification.git
cd Rock-Type-Classification
```

### Prerequisites

Ensure that you have the following installed:

- Python >= 3.11.5
- PyTorch >= 2.1.1
- CUDA for GPU acceleration

### Install dependencies

[Requirement file](https://github.com/Phantom-fs/Rock-Type-Classification/blob/main/Code/requirements.txt)

```bash
cd Code
conda create --name <env> --file requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.