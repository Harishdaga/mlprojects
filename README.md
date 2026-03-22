# 🎓 Student Performance Analysis & Prediction

> End-to-end data analysis and machine learning project predicting student exam performance based on demographic and social factors.

---

## 📌 Overview

This project analyses a student performance dataset to understand how factors like **gender, parental education, lunch type, and test preparation** influence exam scores in Maths, Reading, and Writing.

It covers the full data science workflow — from exploratory analysis to model training and a deployable web application.

---

## 📊 What's Inside

| File / Folder | Description |
|---|---|
| `notebook/EDA STUDENT PERFORMANE.ipynb` | Full exploratory data analysis with visualisations |
| `notebook/MODEL TRAINING.ipynb` | Model comparison and selection |
| `notebook/data/` | Student performance dataset |
| `src/` | Modular pipeline — data ingestion, transformation, model trainer |
| `app.py` / `application.py` | Flask web app for predictions |
| `artifacts/` | Saved model and preprocessor |
| `templates/` | HTML templates for the web app |

---

## 🔍 Project Workflow

1. **EDA** — Analysed score distributions, impact of parental education, lunch type, and test prep on performance
2. **Feature Engineering** — Encoded categorical variables, scaled numerical features using `ColumnTransformer`
3. **Model Training** — Compared multiple regressors: Linear Regression, Ridge, Lasso, K-Neighbours, Decision Tree, Random Forest, XGBoost, CatBoost, AdaBoost
4. **Best Model Selection** — Evaluated using R² score on test data
5. **Deployment** — Flask web app for real-time score prediction

---

## 🎯 Key Insights from EDA

- Students who completed test preparation scored significantly higher
- Parental level of education positively correlates with student performance
- Standard lunch vs free/reduced lunch shows notable score differences
- Female students scored higher in Reading & Writing; male students in Maths

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)

---

## 👤 Author

**Harish Daga** — Data Analyst
[harish.cv](https://harish.cv) · [LinkedIn](https://linkedin.com/in/harishdaga)
