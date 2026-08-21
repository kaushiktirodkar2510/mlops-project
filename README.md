# 🚗 Used Car Price Prediction

A machine learning application that predicts the estimated selling price of a used car based on its specifications and history.

The project uses a Random Forest Regression model with preprocessing for numerical and categorical features. A Streamlit web application provides an interactive interface for users to enter vehicle details and receive a predicted price.

---

## 📌 Project Overview

Used car prices depend on several factors such as manufacturing year, mileage, engine capacity, fuel type, brand, transmission, ownership history, service history, accidents, and insurance status.

This project uses machine learning to learn the relationship between these features and the selling price of used cars.

The project includes:

- Data preprocessing
- Numerical and categorical feature handling
- One-hot encoding
- Missing-value imputation
- Random Forest Regression
- Model evaluation
- Model serialization using Joblib
- Prediction module
- Streamlit web application
- Git/GitHub version control

---

## 🎯 Objectives

The main objectives of this project are:

1. Build a machine learning model for used car price prediction.
2. Handle numerical and categorical data appropriately.
3. Evaluate the performance of the trained model.
4. Save the trained model for reuse.
5. Create a reusable prediction module.
6. Develop an interactive Streamlit application.
7. Maintain the project using Git and GitHub.

---

## 🗂️ Project Structure

```text
mlops-project/
│
├── data/
│   └── used_cars.csv
│
├── models/
│   └── car_price_model.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
