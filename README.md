<div align="center"></div>

# 🦷 Periodontal Disease Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-orange.svg)](https://www.kaggle.com/datasets/renataadarmanto/periodontal-disease)

*A machine learning project to predict periodontal disease using neural networks and advanced ML techniques.*

[📊 Dataset](#-dataset) • [🚀 Getting Started](#-getting-started) • [🎓 Learning Objectives](#-learning-objectives) • [📋 Roadmap](#-roadmap) • [✅ Progress](#-progress)

</div>

## 🎯 Project Overview

This project builds upon previous experience with neural networks (from the EV charging prediction exercise) to tackle a healthcare prediction problem. We'll use real-world periodontal disease data from Kaggle to develop and compare different machine learning approaches.

## 🎓 Learning Objectives

### Core ML Skills
- [ ] **Data Exploration & Analysis**: Learn to thoroughly understand healthcare datasets
- [ ] **Feature Engineering**: Discover how to create meaningful features from clinical data
- [ ] **Model Comparison**: Compare different algorithms (logistic regression vs neural networks)
- [ ] **Performance Evaluation**: Learn appropriate metrics for classification problems
- [ ] **Cross-validation**: Implement proper validation techniques to avoid overfitting

### Advanced Techniques
- [ ] **Hyperparameter Tuning**: Optimize neural network architecture and training parameters
- [ ] **Regularization**: Implement dropout, L1/L2 regularization to prevent overfitting
- [ ] **Class Imbalance**: Handle imbalanced datasets common in medical data
- [ ] **Feature Selection**: Identify the most important predictors of periodontal disease

### Domain Knowledge
- [ ] **Healthcare ML**: Understand challenges specific to medical prediction problems
- [ ] **Clinical Features**: Learn about periodontal disease indicators and measurements
- [ ] **Model Interpretation**: Make models interpretable for clinical decision-making

## 📋 Project Roadmap

### Phase 1: Data Foundation ✅ COMPLETED
- [x] **Dataset Acquisition**: Downloaded periodontal disease dataset from Kaggle
- [x] **Data Exploration**: Analyzed dataset structure, image properties, and categories
- [x] **Data Quality Assessment**: Identified image formats, dimensions, and class distribution

### Phase 2: Data Preprocessing ✅ COMPLETED
- [x] **Data Cleaning**: Handled image format variations (JPEG vs MPO)
- [x] **Dataset Class Creation**: Built custom PyTorch Dataset class for image loading
- [x] **Data Encoding**: Created labels for 3-class classification (MGI=0, OHG=1, PFI=2)
- [x] **Dataset Structure**: Successfully loaded all 1,376 images across three categories

### Phase 3: Data Splitting and Preprocessing ✅ COMPLETED
- [x] **Train/Test Split**: Successfully implemented train/test split using PyTorch's random_split
- [x] **Transform Pipeline**: Created separate transforms for training (with augmentation) and testing
- [x] **Data Preprocessing**: Implemented image resizing (224x224), normalization, and data augmentation
- [x] **Dataset Structure**: Ready for PyTorch DataLoader integration with proper train/test datasets

### Phase 4: Neural Network Development
- [ ] **CNN Architecture**: Design convolutional neural network for image classification
- [ ] **Training Implementation**: Implement training loop with proper validation
- [ ] **Hyperparameter Tuning**: Optimize learning rate, architecture, and regularization
- [ ] **Model Persistence**: Save and load trained models

### Phase 5: Advanced Techniques
- [ ] **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
- [ ] **Ensemble Methods**: Explore combining multiple models for better performance
- [ ] **Feature Importance**: Analyze which features contribute most to predictions
- [ ] **Model Interpretation**: Create visualizations and explanations for model decisions

### Phase 6: Evaluation & Analysis
- [ ] **Performance Comparison**: Compare all models using appropriate metrics
- [ ] **Error Analysis**: Understand where models fail and why
- [ ] **Clinical Relevance**: Assess practical applicability of the model
- [ ] **Documentation**: Create comprehensive analysis and recommendations

## 📊 Dataset

### Source
This project utilizes the **Periodontal Disease Dataset** from Kaggle, created by [jiaweihong2032](https://www.kaggle.com/datasets/jiaweihong2032/periodontal-data-for-testing).

### Dataset Information
- **📁 Structure**: Organized into 3 testing categories
- **🖼️ Content**: Clinical intraoral photographs for periodontal disease classification
- **📈 Size**: 1,376 images (1.43 GB)
- **🏷️ Classes**: 3 periodontal disease assessment indices
- **📊 Format**: RGB images (2784 × 1856 pixels)

### Dataset Categories
The dataset contains three types of periodontal disease assessment images:

1. **MGI (Modified Gingival Index)**: 383 images
   - Assessment of gum inflammation and health
   - Focus on gingival tissue condition

2. **OHG (Oral Hygiene Index)**: 619 images  
   - Overall oral hygiene assessment
   - Evaluation of dental cleanliness

3. **PFI (Plaque Formation Index)**: 374 images
   - Plaque accumulation measurement
   - Bacterial buildup assessment

### Dataset Access
```bash
# Download via Kaggle Hub
import kagglehub
path = kagglehub.dataset_download("jiaweihong2032/periodontal-data-for-testing")

# Or visit: https://www.kaggle.com/datasets/jiaweihong2032/periodontal-data-for-testing
```

### Directory Structure
```
datasets/images/
├── MGI-testing/MGI/     # 383 Modified Gingival Index images
├── OHG-testing/OHG/     # 619 Oral Hygiene Index images
└── PFI-testing/PFI/     # 374 Plaque Formation Index images
```

### Image Properties
- **Dimensions**: 2784 × 1856 pixels
- **Color Mode**: RGB (color images)
- **Formats**: JPEG, MPO
- **Content**: Frontal intraoral photographs showing anterior teeth and gums
- **Challenge**: Subtle visual differences between assessment categories

## ✅ Progress

### 🎉 **Completed Milestones**

#### **Phase 1: Data Foundation** ✅
- **Dataset Discovery**: Successfully identified and downloaded the periodontal disease dataset
- **Data Exploration**: Analyzed 1,376 clinical intraoral photographs
- **Category Identification**: Discovered 3 assessment indices (MGI, OHG, PFI)
- **Image Analysis**: Determined image properties (2784×1856 RGB, JPEG/MPO formats)
- **Visual Exploration**: Displayed and compared sample images from each category

#### **Key Discoveries** 🔍
- **Problem Type**: 3-class image classification (MGI vs OHG vs PFI)
- **Image Content**: Frontal intraoral photographs showing anterior teeth and gums
- **Class Distribution**: OHG (619), PFI (374), MGI (383) - slight class imbalance
- **Challenge Level**: Subtle visual differences between assessment categories
- **Technical Setup**: Successfully configured PyTorch, PIL, and matplotlib for image processing

#### **Phase 2: Data Preprocessing** ✅
- **Custom Dataset Class**: Built `PeriodontalDataset` class inheriting from `torch.utils.data.Dataset`
- **Image Loading**: Successfully loaded all 1,376 images with proper path handling
- **Label Encoding**: Implemented 3-class classification labels (MGI=0, OHG=1, PFI=2)
- **Data Structure**: Created lists of image paths and corresponding labels
- **Technical Skills**: Learned nested loops, path manipulation, and PyTorch dataset architecture

#### **Phase 3: Data Splitting and Preprocessing** ✅
- **Train/Test Split**: Implemented 80/20 split using PyTorch's `random_split` function
- **Transform Pipeline**: Created separate transforms for training (with augmentation) and testing
- **Image Preprocessing**: Implemented resizing to 224x224, normalization, and data augmentation
- **Data Augmentation**: Added random horizontal flips and rotations for training data
- **Technical Skills**: Learned PyTorch transforms, data splitting, and preprocessing pipelines

### 🔄 **Current Status**
- **Next Phase**: CNN Architecture Design and Training Implementation
- **Learning Focus**: Building convolutional neural networks for medical image classification
- **Technical Skills**: Ready to implement CNN architecture, training loops, and model evaluation

## 🛠️ Technical Stack

<table>
<tr>
<td align="center" width="20%">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40" height="40"/>
<br><b>Python 3.8+</b>
</td>
<td align="center" width="20%">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="40" height="40"/>
<br><b>PyTorch</b>
</td>
<td align="center" width="20%">
<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="40" height="40"/>
<br><b>Scikit-learn</b>
</td>
<td align="center" width="20%">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="40" height="40"/>
<br><b>Pandas</b>
</td>
<td align="center" width="20%">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="40" height="40"/>
<br><b>NumPy</b>
</td>
</tr>
</table>

- **🔬 PyTorch**: Neural network implementation and computer vision
- **📊 scikit-learn**: Baseline models and preprocessing
- **🐼 pandas**: Data manipulation and analysis
- **🔢 numpy**: Numerical computations
- **📈 matplotlib/seaborn**: Data visualization
- **📓 Jupyter Notebooks**: Interactive development and analysis

## 📊 Expected Outcomes

By the end of this project, you will have:

1. **Practical ML Pipeline**: A complete end-to-end machine learning workflow
2. **Healthcare ML Experience**: Understanding of medical prediction challenges
3. **Advanced Neural Networks**: Experience with regularization, hyperparameter tuning, and validation
4. **Model Comparison Skills**: Ability to evaluate and compare different ML approaches
5. **Portfolio Project**: A well-documented project demonstrating ML expertise

## 🎯 Success Metrics

- [ ] Model achieves reasonable accuracy on test data (>80% for binary classification)
- [ ] Proper validation prevents overfitting (training vs validation performance)
- [ ] Model is interpretable and clinically meaningful
- [ ] Code is well-documented and reproducible
- [ ] Clear understanding of model limitations and improvements

## 📝 Notes

This project emphasizes **learning over just getting results**. Each step includes:
- **Hints and guidance** rather than direct solutions
- **Questions to think about** before implementing
- **Reflection on decisions** and trade-offs
- **Exploration of alternatives** and improvements

## 🚀 Getting Started

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/Tuminha/predicting-periodontal-disease.git
cd predicting-periodontal-disease

# Install dependencies
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn
pip install scikit-learn pillow jupyter
pip install kaggle  # For dataset download
```

### Quick Start
1. **📥 Download Dataset**: Use Kaggle API to get the periodontal disease dataset
2. **🔍 Explore Data**: Start with data exploration in Jupyter notebooks
3. **🏗️ Build Models**: Follow the roadmap for systematic development
4. **📊 Evaluate**: Compare different approaches and analyze results

### Project Structure
```
periodontal_disease_predictor/
├── 📁 datasets/
│   └── 📁 images/           # Downloaded image data
├── 📁 notebooks/            # Jupyter notebooks for exploration
├── 📁 models/              # Saved model files
├── 📄 README.md            # This file
└── 📄 requirements.txt     # Python dependencies
```

---

<div align="center">

**🎓 Remember: The goal is to become great at ML through hands-on practice and deep understanding!**

[⬆️ Back to Top](#-periodontal-disease-predictor)

</div>
