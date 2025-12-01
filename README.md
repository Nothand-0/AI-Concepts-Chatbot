# Bais-audit-notebook
# Bias Audit and Mitigation in Machine Learning

## Overview
This repository contains a **fully self-contained Jupyter notebook** that demonstrates a bias audit for a synthetic hiring dataset. The notebook includes:

- Generation of a **synthetic dataset** with built-in gender bias.
- Training a **baseline logistic regression model**.
- Calculation of **fairness metrics**:
  - Disparate Impact (DI)
  - Equal Opportunity Difference (EOD)
  - Average Odds Difference (AOD)
- Implementation of a **post-processing bias mitigation technique** (threshold adjustment per group).
- Visualizations of **prediction rates** and **confusion matrices** before and after mitigation.
- **Statistical testing** (Chi-square test) to validate bias.

This notebook is designed for both technical and non-technical audiences to explore bias in machine learning systems.

---

## Features

1. **Synthetic Dataset Creation**
   - Simulates a hiring scenario with features: `gender`, `age`, `experience`.
   - Introduces controlled bias favoring males.

2. **Model Training**
   - StandardScaler preprocessing.
   - Logistic Regression classifier.
   - Baseline accuracy calculation.

3. **Fairness Metrics**
   - `compute_metrics` function calculates DI, EOD, and AOD per demographic group.

4. **Bias Mitigation**
   - Post-processing threshold adjustment to equalize outcomes across groups.
   - Comparison of fairness metrics before and after mitigation.

5. **Visualizations**
   - Bar plots of prediction rates per gender.
   - Confusion matrices per gender before and after mitigation.

6. **Statistical Testing**
   - Chi-square test for independence between predicted outcomes and gender.

---

## Requirements

- Python 3.7+
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - scipy

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
