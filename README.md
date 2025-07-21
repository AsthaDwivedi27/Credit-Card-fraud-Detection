# Credit-Card-fraud-Detection




This project focuses on detecting fraudulent credit card transactions using machine learning techniques. It utilizes a highly imbalanced dataset and employs advanced resampling techniques like SMOTE to balance class distribution. Two models — Logistic Regression and Random Forest — are trained and evaluated for performance comparison.
---

## Features

- Data preprocessing (scaling, cleaning)
- SMOTE for handling class imbalance
- Model training using Logistic Regression and Random Forest
- Evaluation using Accuracy, Confusion Matrix, ROC-AUC, and more
- Basic visualizations using Seaborn and Matplotlib

---

## How to Run

1. Make sure you have Python installed.
2. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
````

3. Run the script:

```bash
python summer.py
```

> Note: Update the dataset path inside the script if needed.

---

## Dataset

The dataset is a credit card transaction log, with imbalanced labels (fraud vs legit).

---

## Output

The script prints:

* Classification reports
* Accuracy & ROC-AUC scores
* Confusion matrix
* Visual plots for data distribution and results

---


