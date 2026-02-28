# 🔴 FRAUDSENSE — AI Transaction Fraud Detector

> A production-grade, single-file ML web app that detects fraudulent transactions in real time. Features SMOTE oversampling for class imbalance, threshold-optimized models, SHAP-style attribution, ROC + Precision-Recall curves, confusion matrix, and a live animated transaction feed.

**Made by [Musharib](https://linktr.ee/Musharib_)**

[![GitHub](https://img.shields.io/badge/GitHub-musharibramzan950--bit-181717?style=flat&logo=github)](https://github.com/musharibramzan950-bit)
[![Linktree](https://img.shields.io/badge/Linktree-Musharib__-43E55E?style=flat&logo=linktree)](https://linktr.ee/Musharib_)

---

## ✨ Features

- **5 ML Models** — Gradient Boosting, Random Forest, Extra Trees, Neural Network (MLP), Logistic Regression
- **SMOTE Oversampling** — handles real-world class imbalance (5.5% fraud rate)
- **Optimal threshold tuning** — per-model F1-maximizing decision threshold
- **16 transaction features** — amount, velocity, distance, merchant, country, card age, and more
- **Fraud probability gauge** — 0–100% score with color-coded verdict
- **SHAP-style risk drivers** — per-prediction feature attribution bars
- **ROC Curve** — true/false positive tradeoff per model
- **Precision-Recall Curve** — especially useful for imbalanced datasets
- **Confusion Matrix** — TP/TN/FP/FN breakdown
- **Feature Importance** — which signals catch fraud best
- **Fraud by hour chart** — when fraud happens most
- **🔴 Live animated transaction feed** — real-time scrolling legit/fraud transactions
- **Auto-installs all dependencies**

---

## 🚀 Quick Start

```bash
python fraudsense.py
```

Opens at → `http://localhost:7070`

Zero config. Just run it.

---

## 📦 Dependencies

Auto-installed. Or manually:

```bash
pip install flask scikit-learn numpy pandas imbalanced-learn
```

**Python 3.8+** required.

---

## 🧠 ML Architecture

| Model | Algorithm | Notes |
|---|---|---|
| Gradient Boost | `GradientBoostingClassifier` | 300 trees, lr=0.07, depth=5 |
| Random Forest | `RandomForestClassifier` | 250 trees, balanced class weights |
| Extra Trees | `ExtraTreesClassifier` | 200 trees, balanced |
| Neural Net | `MLPClassifier` | 128→64→32→16, ReLU, Adam |
| Logistic Reg. | `LogisticRegression` + `RobustScaler` | L2, C=0.3, balanced |

### Key ML Techniques:
- **SMOTE** — Synthetic Minority Oversampling on training data only (no leakage)
- **RobustScaler** — outlier-resistant normalization for linear/NN models
- **Stratified split** — preserves fraud rate in train/test
- **Threshold optimization** — 80-point F1 sweep per model to find optimal cutoff
- **Evaluation**: AUC-ROC, Average Precision, F1, Precision, Recall, CV-AUC

---

## 🎯 Input Features (16)

| Feature | Description |
|---|---|
| Amount ($) | Transaction dollar amount |
| Hour | Hour of day (0–23) |
| Day of Week | Mon=0 … Sun=6 |
| Merchant Category | Grocery / Gas / ATM / Gambling etc |
| Country | US / UK / NG / RU / Unknown etc |
| Txns (1h) | Transactions in last 1 hour |
| Txns (24h) | Transactions in last 24 hours |
| Avg Amount 7d | Average spend over past week |
| Distance (km) | Distance from home |
| Card Age (days) | Age of card used |
| Failed PINs | Number of failed PIN attempts |
| Online Transaction | 0/1 flag |
| Foreign Transaction | 0/1 flag |
| Time Since Last (min) | Minutes since last transaction |
| Velocity Score | Internal velocity risk score (0–100) |
| Amount vs 7d Avg | Ratio of amount to 7-day average |

---

## 📊 Dashboard Panels

| Panel | Description |
|---|---|
| Transaction Form | 16-feature input for any transaction |
| Fraud Probability | % score + Low/Suspicious/High/FRAUD verdict |
| Risk Drivers | Which features pushed score up or down |
| Model Leaderboard | AUC, AP, F1, Precision, Recall, Threshold |
| ROC Curve | Per-model, interactive |
| Precision-Recall Curve | Better for imbalanced datasets |
| Feature Importance | Per-model horizontal bar chart |
| Confusion Matrix | TP/TN/FP/FN color grid |
| Fraud by Hour | Bar chart showing when fraud peaks |
| Live Feed | Animated real-time transaction stream |

---

## 📸 Terminal Output

```
════════════════════════════════════════════════════════
  🔴  FRAUDSENSE — AI Transaction Fraud Detector
  Made by Musharib | linktr.ee/Musharib_
════════════════════════════════════════════════════════
  ⚙️  Generating 8,000 transaction profiles…
  📊  Fraud rate: 5.5%  (440 fraudulent)
  ⚖️  Applying SMOTE oversampling…
  🧠  Training 5 ML models with optimal threshold tuning…

     Gradient Boost        AUC=0.9781  AP=0.9120  F1=0.8843
     Random Forest         AUC=0.9734  AP=0.9047  F1=0.8790
     Extra Trees           AUC=0.9690  AP=0.8980  F1=0.8710
     Neural Net            AUC=0.9612  AP=0.8840  F1=0.8600
     Logistic Reg.         AUC=0.9210  AP=0.8200  F1=0.8100

  ✅ Best: Gradient Boost  (Avg Precision = 0.9120)

  🌐  Running at → http://localhost:7070
════════════════════════════════════════════════════════
```

---

## 🛠️ Config

```bash
PORT=8080 python fraudsense.py
```

---

## 📁 Structure

```
fraudsense.py    ← entire app (single file)
README.md        ← this file
```

---

## 👤 Author

**Musharib Ramzan**

- 🌐 [Linktree](https://linktr.ee/Musharib_)
- 💻 [GitHub](https://github.com/musharibramzan950-bit)

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built with Python · Flask · scikit-learn · imbalanced-learn · Chart.js*
