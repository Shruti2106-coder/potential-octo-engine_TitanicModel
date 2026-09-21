# Titanic Survival Prediction 🚢

A machine learning project that predicts whether a passenger survived the Titanic disaster using classification algorithms and feature engineering.

## 📌 Project Overview

The goal of this project is to build a machine learning model that predicts passenger survival based on information such as:

* Passenger class
* Gender
* Age
* Number of siblings/spouses
* Number of parents/children
* Fare
* Port of embarkation
* Family size
* Passenger title
* Ticket group size
* Cabin information

The project uses feature engineering and multiple machine learning classification models to improve prediction performance.

## 🎯 Objective

Predict the `Survived` class for passengers in the Titanic test dataset.

* `0` = Did not survive
* `1` = Survived

## 📂 Dataset

The project is based on the Kaggle Titanic dataset.

The training dataset contains passenger information along with the target variable `Survived`.

The test dataset contains passenger information without the target variable.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Machine Learning
* Classification
* Feature Engineering

## 🧠 Feature Engineering

The following additional features are created:

* `FamilySize`
* `IsAlone`
* `Title`
* `TicketGroupSize`
* `FarePerPerson`
* `CabinDeck`
* `FamilyCategory`

These features provide additional information about the passenger and their travel group.

## 🤖 Machine Learning Models

The project evaluates multiple classification algorithms:

1. Random Forest
2. Extra Trees
3. Support Vector Machine (SVM)
4. Logistic Regression

Five-fold Stratified Cross-Vali

