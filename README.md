# KRISHNOPREYA_MNIST_CODE-AND-REPORT
# MNIST Handwritten Digit Classification  
### Classical Machine Learning Approach

## Overview
This project presents a complete end-to-end pipeline for **handwritten digit classification** using the **MNIST dataset** and **classical machine learning techniques**.  
It covers data exploration, preprocessing, dimensionality reduction using PCA, multiple model implementations (including one from scratch), hyperparameter tuning, ensemble learning, and detailed error analysis.

The objective is to **compare traditional ML models** on a high-dimensional image dataset and analyze their performance, strengths, and weaknesses.

---

## Dataset
- **Name:** MNIST Handwritten Digits
- **Total samples:** 70,000
- **Image size:** 28 × 28 pixels
- **Features:** 784 pixel values
- **Classes:** Digits 0–9
- **Train/Test split:** 80% / 20% (stratified)

Dataset is loaded using `fetch_openml` with fallback support for CSV input.

---

## Project Workflow
Load Dataset
↓
Exploratory Data Analysis
↓
Preprocessing & Normalization
↓
PCA Dimensionality Reduction
↓
Model Training & Hyperparameter Tuning
↓
Evaluation & Error Analysis
↓
Ensemble Learning
↓
Final Report & Saved Outputs


---

## Exploratory Data Analysis
- Class distribution (bar chart and pie chart)
- Sample image from each digit class
- Average image per digit
- Missing value checks
- Dataset statistics

All plots are automatically saved for reproducibility.

---

## Preprocessing
- Pixel normalization to range **[0, 1]**
- Stratified train-test split
- Dimensionality reduction using **Principal Component Analysis (PCA)**

### PCA Details
- **95% variance retained**
- Dimensionality reduced from **784 → 154**
- Approx. **80% feature reduction**
- PCA variance and projection visualizations included

---

## Models Implemented

### 1. KNN (From Scratch)
- Fully custom implementation (no sklearn)
- Vectorized Euclidean distance computation
- Batch-wise prediction for memory efficiency
- Manual tuning of `k`

### 2. KNN (Scikit-learn)
- GridSearchCV tuning for:
  - Number of neighbors
  - Distance metric
  - Weighting strategy

### 3. Support Vector Machine (SVM)
- Linear and RBF kernels
- Grid search over `C` and `gamma`
- Best-performing individual model

### 4. Decision Tree
- Tuned for:
  - Maximum depth
  - Split criteria
  - Minimum samples per leaf

### 5. Ensemble Models
- Soft Voting Ensemble
- Hard Voting Ensemble
- Combines KNN, SVM, and Decision Tree models

---

## Model Performance Summary

| Model | Accuracy |
|------|---------|
| KNN (From Scratch) | 97.32% |
| KNN (Sklearn) | 97.53% |
| SVM | **98.59%** |
| Decision Tree | 85.09% |
| Soft Voting Ensemble | 98.19% |
| Hard Voting Ensemble | 98.00% |

**Best Individual Model:** SVM (RBF Kernel)

---

## Evaluation Metrics
For each model:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

Additional analysis includes:
- Visualization of misclassified samples
- Most frequent confusion pairs
- Per-digit error rate analysis

---

## PCA Effect Analysis
Model performance was evaluated with varying PCA components:
- 50 components
- 100 components
- 154 components (95% variance)

Results show diminishing accuracy gains beyond ~100 components, especially for tree-based models.

---

## Output Files

### Images

---

## Saved Models
The following trained components are serialized using `pickle`:
- Custom KNN model
- Sklearn KNN
- SVM
- Decision Tree
- Soft & Hard Voting Ensembles
- PCA transformer

This allows reuse without retraining.

---

## Technologies Used
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Graphviz
- Google Colab

---

## How to Run
1. Open the notebook in **Google Colab**
2. Run all cells sequentially
3. Outputs, plots, and trained models are generated automatically

---

## Key Takeaways
- Classical ML models can achieve **>98% accuracy** on MNIST with proper preprocessing
- PCA dramatically reduces dimensionality with minimal accuracy loss
- SVM performs best on high-dimensional image data
- Ensemble models do not always outperform the best single model

---

## Author
Developed as a complete classical machine learning study on MNIST, covering implementation, experimentation, and analysis.
