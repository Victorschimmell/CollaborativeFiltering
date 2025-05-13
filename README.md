# Memory-Based Collaborative Filtering: A Comparative Study

This repository contains the code and experiments from my bachelor's thesis, which investigates key design choices in **memory-based collaborative filtering (CF)**. The study focuses on how different **similarity measures** and **neighborhood sizes** affect the performance of CF models, using the widely adopted **MovieLens** explicit-rating datasets.

## 📌 Motivation

Personalized recommendation systems play a crucial role in modern digital platforms by tailoring content to individual user preferences. While **collaborative filtering** remains one of the most established techniques, the literature often emphasizes predictive accuracy, leaving a gap in understanding how specific design decisions impact performance in practice.

## 🎯 Objectives

This project aims to:

- Evaluate the effect of different similarity functions (e.g., cosine, Pearson) on CF performance
- Analyze how neighborhood size (`k`) influences recommendation quality and coverage
- Compare memory-based CF to a lightweight model-based baseline (e.g., FunkSVD)
- Investigate the impact of rating sparsity on the robustness of similarity-based methods

## ❓ Research Questions

1. **RQ1**: To what extent do different similarity measures influence the accuracy of memory-based CF?
2. **RQ2**: How does neighborhood size (`k`) affect recommendation quality and coverage?
3. **RQ3**: How does memory-based CF compare to a lightweight model-based method in terms of accuracy and computational cost?
4. **RQ4**: What is the impact of rating sparsity on similarity robustness?

## 📁 Contents

- `data/`: Contains scripts for loading and processing MovieLens datasets
- `metrics/`: Evaluation metrics (precision, recall, MAE, RMSE, etc.)
- `models/`: Implementations of memory-based CF and FunkSVD
- `experiments/`: Scripts and notebooks for running experiments
- `results/`: Plots and evaluation outputs

## 🛠️ Technologies

- Python
- NumPy / Pandas
- Scikit-learn
- Matplotlib / Seaborn

## 📜 License

MIT License. See `LICENSE` file for details.

## 🙋‍♂️ Author

Victor [Your Last Name] – Bachelor of Science in Computer Science, University of Copenhagen
