<div align="center"></div>

# 🦷 Periodontal Disease Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-orange.svg)](https://www.kaggle.com/datasets/renataadarmanto/periodontal-disease)

*A machine learning project to predict periodontal disease using neural networks and advanced ML techniques.*

[📊 Dataset](#-dataset) • [🚀 Getting Started](#-getting-started) • [🎓 Learning Objectives](#-learning-objectives) • [📋 Roadmap](#-roadmap)

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

### Phase 1: Data Foundation
- [ ] **Dataset Acquisition**: Find and download appropriate periodontal disease dataset from Kaggle
- [ ] **Data Exploration**: Understand dataset structure, features, and target variables
- [ ] **Data Quality Assessment**: Identify missing values, outliers, and data quality issues

### Phase 2: Data Preprocessing
- [ ] **Data Cleaning**: Handle missing values and outliers
- [ ] **Feature Engineering**: Create new meaningful features from existing data
- [ ] **Data Encoding**: Convert categorical variables to numerical format
- [ ] **Feature Scaling**: Normalize/standardize features for neural network training

### Phase 3: Baseline Models
- [ ] **Simple Baseline**: Create a simple rule-based or statistical baseline
- [ ] **Logistic Regression**: Implement logistic regression as a linear baseline
- [ ] **Performance Benchmarking**: Establish performance baselines for comparison

### Phase 4: Neural Network Development
- [ ] **Network Architecture**: Design appropriate neural network structure
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
This project utilizes the **Periodontal Disease Dataset** from Kaggle, created by [Renata A. Darmanto](https://www.kaggle.com/datasets/renataadarmanto/periodontal-disease).

### Dataset Information
- **📁 Structure**: Organized into train/validation/test splits
- **🖼️ Content**: Medical images for periodontal disease classification
- **📈 Size**: ~220 files (17.9 MB)
- **🏷️ Classes**: Multiple periodontal disease categories
- **📊 Usage**: 117 downloads, 668 views (as of September 2024)

### Dataset Access
```bash
# Download via Kaggle API
kaggle datasets download renataadarmanto/periodontal-disease

# Or visit: https://www.kaggle.com/datasets/renataadarmanto/periodontal-disease
```

### Directory Structure
```
periodontal_disease/
├── train/          # Training images
├── val/            # Validation images  
└── test/           # Test images
```

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
