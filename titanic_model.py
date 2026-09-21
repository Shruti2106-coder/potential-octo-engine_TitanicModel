# ==========================================
# TITANIC SURVIVAL PREDICTION
# ==========================================

# Import libraries
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ==========================================
# 1. LOAD DATA
# ==========================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Training data shape:", train.shape)
print("Test data shape:", test.shape)

print("\nFirst 5 rows of training data:")
print(train.head())


# ==========================================
# 2. BASIC DATA INFORMATION
# ==========================================

print("\nTraining Data Information:")
print(train.info())

print("\nMissing values in training data:")
print(train.isnull().sum())

print("\nMissing values in test data:")
print(test.isnull().sum())

print("\nSurvival distribution:")
print(train["Survived"].value_counts())

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

def create_features(df):
    df = df.copy()

    # 1. Calculate family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # 2. Check whether passenger was travelling alone
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # 3. Extract title from passenger's name
    df["Title"] = df["Name"].str.extract(
        r",\s*([^.]*)\.",
        expand=False
    )

    # 4. Group uncommon titles as "Rare"
    title_counts = df["Title"].value_counts()

    rare_titles = title_counts[title_counts < 10].index

    df["Title"] = df["Title"].replace(
        rare_titles,
        "Rare"
    )

    # 5. Standardize some titles
    df["Title"] = df["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })

    # 6. Calculate fare per family member
    df["FarePerPerson"] = (
        df["Fare"] / df["FamilySize"]
    )
        # 7. Extract cabin deck
    df["CabinDeck"] = df["Cabin"].fillna("U").str[0]

    # 8. Count how many passengers share the same ticket
    ticket_counts = df["Ticket"].value_counts()
    df["TicketGroupSize"] = df["Ticket"].map(ticket_counts)

    return df

# Apply feature engineering to both datasets

train = create_features(train)
test = create_features(test)

# Display the newly created features

print("\nNewly Created Features:")
print(
    train[
        [
            "FamilySize",
            "IsAlone",
            "Title",
            "FarePerPerson"
        ]
    ].head(10)
)


# ==========================================
# 4. SELECT FEATURES
# ==========================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "Title",
    "FarePerPerson",
    "CabinDeck",
    "TicketGroupSize"
]

# Features for training
X = train[features].copy()

# Target variable
y = train["Survived"].copy()

# Features for test data
X_test = test[features].copy()

print("\nSelected Features:")
print(X.columns.tolist())

print("\nShape of X:", X.shape)
print("Shape of X_test:", X_test.shape)

# ==========================================
# 5. HANDLE MISSING VALUES
# ==========================================

# Numerical columns
numerical_columns = [
    "Age",
    "Fare",
    "FarePerPerson"
]

# Fill numerical missing values with median
for column in numerical_columns:

    median_value = X[column].median()

    X[column] = X[column].fillna(median_value)

    X_test[column] = X_test[column].fillna(median_value)


# Categorical columns
categorical_columns = [
    "Sex",
    "Embarked",
    "Title",
    "CabinDeck"
]

# Fill categorical missing values with most common value
for column in categorical_columns:

    mode_value = X[column].mode()[0]

    X[column] = X[column].fillna(mode_value)

    X_test[column] = X_test[column].fillna(mode_value)


# Check remaining missing values

print("\nMissing values in training features:")
print(X.isnull().sum())

print("\nMissing values in test features:")
print(X_test.isnull().sum())

# ==========================================
# 6. ENCODE CATEGORICAL FEATURES
# ==========================================

# Convert categorical columns into numerical columns

X = pd.get_dummies(
    X,
    columns=["Sex", "Embarked", "Title", "CabinDeck"],
    drop_first=True
)

X_test = pd.get_dummies(
    X_test,
    columns=["Sex", "Embarked", "Title", "CabinDeck"],
    drop_first=True
)

# Make sure training and test data have exactly
# the same columns
X_test = X_test.reindex(
    columns=X.columns,
    fill_value=0
)


print("\nEncoded Training Data:")
print(X.head())

print("\nEncoded Training Shape:")
print(X.shape)

print("\nEncoded Test Shape:")
print(X_test.shape)

# ==========================================
# 7. TRAIN / VALIDATION SPLIT
# ==========================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nValidation Data Shape:")
print(X_valid.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nValidation Target Shape:")
print(y_valid.shape)

# ==========================================
# 8. TRAIN MULTIPLE ML MODELS
# ==========================================

# Create the models

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# Dictionary to store accuracy results

results = {}


# Train and evaluate every model

for name, model in models.items():

    print("\n======================================")
    print(name)
    print("======================================")

    # Train model
    model.fit(X_train, y_train)

    # Predict validation data
    predictions = model.predict(X_valid)

    # Calculate accuracy
    accuracy = accuracy_score(
        y_valid,
        predictions
    )

    # Store result
    results[name] = accuracy

    print(
        "Validation Accuracy:",
        round(accuracy, 4)
    )

    # Detailed performance
    print("\nClassification Report:")
    print(
        classification_report(
            y_valid,
            predictions
        )
    )
    
    # ==========================================
# 9. CROSS-VALIDATION
# ==========================================

print("\n======================================")
print("5-FOLD CROSS-VALIDATION")
print("======================================")

cv_results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    cv_results[name] = scores

    print(f"\n{name}")
    print("Fold Scores:", scores)
    print("Mean Accuracy:", round(scores.mean(), 4))
    print("Standard Deviation:", round(scores.std(), 4))
    
    # ==========================================
# 10. COMPARE CROSS-VALIDATION RESULTS
# ==========================================

print("\n======================================")
print("MODEL COMPARISON")
print("======================================")

for name, scores in cv_results.items():
    print(
        f"{name}: "
        f"Mean Accuracy = {scores.mean():.4f}, "
        f"Std = {scores.std():.4f}"
    )

# Find the model with the highest mean accuracy
best_model_name = max(
    cv_results,
    key=lambda name: cv_results[name].mean()
)

print("\nBest Model Based on Cross-Validation:")
print(best_model_name)

print(
    "Best Mean CV Accuracy:",
    round(cv_results[best_model_name].mean(), 4)
)

# ==========================================
# 11. RANDOM FOREST HYPERPARAMETER TUNING
# ==========================================

print("\n======================================")
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("======================================")

# Create Random Forest model
rf = RandomForestClassifier(
    random_state=42
)

# Parameters we want to test
param_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [5, 7, 9, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# Start tuning
grid_search.fit(X, y)

# Best parameters
print("\nBest Parameters:")
print(grid_search.best_params_)

# Best cross-validation score
print("\nBest Cross-Validation Accuracy:")
print(round(grid_search.best_score_, 4))

# ==========================================
# 12. EVALUATE TUNED RANDOM FOREST
# ==========================================

print("\n======================================")
print("TUNED RANDOM FOREST EVALUATION")
print("======================================")

# Get the best model found by GridSearchCV
best_rf = grid_search.best_estimator_

# Train best model on training portion
best_rf.fit(X_train, y_train)

# Predict validation data
tuned_predictions = best_rf.predict(X_valid)

# Calculate validation accuracy
tuned_accuracy = accuracy_score(
    y_valid,
    tuned_predictions
)

print(
    "\nTuned Random Forest Validation Accuracy:",
    round(tuned_accuracy, 4)
)

print("\nClassification Report:")
print(
    classification_report(
        y_valid,
        tuned_predictions
    )
)

# ==========================================
# 14. LEAKAGE-SAFE PREPROCESSING PIPELINE
# ==========================================

print("\n======================================")
print("LEAKAGE-SAFE RANDOM FOREST PIPELINE")
print("======================================")

# Numerical features
numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "FarePerPerson",
    "TicketGroupSize"
]

# Categorical features
categorical_features = [
    "Sex",
    "Embarked",
    "Title",
    "CabinDeck"
]

# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            drop="first"
        ))
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Random Forest model
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=800,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        )
    ]
)

# Cross-validation
pipeline_scores = cross_val_score(
    rf_pipeline,
    train[features],
    y,
    cv=5,
    scoring="accuracy"
)

print("\nFold Scores:")
print(pipeline_scores)

print(
    "\nMean Accuracy:",
    round(pipeline_scores.mean(), 4)
)

print(
    "Standard Deviation:",
    round(pipeline_scores.std(), 4)
)

# ==========================================
# 15. FINAL MODEL COMPARISON
# ==========================================

print("\n======================================")
print("FINAL MODEL COMPARISON")
print("======================================")

# Logistic Regression pipeline
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

# Gradient Boosting pipeline
gradient_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        )
    ]
)

# Models to compare
final_models = {
    "Logistic Regression": logistic_pipeline,
    "Random Forest": rf_pipeline,
    "Gradient Boosting": gradient_pipeline
}

final_scores = {}

# Evaluate each model
for name, model in final_models.items():

    scores = cross_val_score(
        model,
        train[features],
        y,
        cv=5,
        scoring="accuracy"
    )

    final_scores[name] = scores.mean()

    print(f"\n{name}")
    print("Fold Scores:", scores)
    print("Mean Accuracy:", round(scores.mean(), 4))
    print("Standard Deviation:", round(scores.std(), 4))

# Find best model
best_final_model_name = max(
    final_scores,
    key=final_scores.get
)

print("\n======================================")
print("FINAL MODEL")
print("======================================")

print("Selected Model:", best_final_model_name)
print(
    "Mean CV Accuracy:",
    round(final_scores[best_final_model_name], 4)
)

# ==========================================
# 16. TRAIN FINAL MODEL
# ==========================================

print("\n======================================")
print("TRAINING FINAL MODEL")
print("======================================")

# Get the selected model
final_model = final_models[best_final_model_name]

# Train on ALL training data
final_model.fit(
    train[features],
    y
)

print("Final model trained successfully.")


# ==========================================
# 17. PREDICT TEST DATA
# ==========================================

print("\n======================================")
print("CREATING TEST PREDICTIONS")
print("======================================")

# Predict survival for test passengers
test_predictions = final_model.predict(
    test[features]
)

print("Test predictions created.")
print("Number of predictions:", len(test_predictions))


# ==========================================
# 18. CREATE KAGGLE SUBMISSION
# ==========================================

print("\n======================================")
print("CREATING SUBMISSION FILE")
print("======================================")

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions.astype(int)
})

# Save submission file
submission.to_csv(
    "submission.csv",
    index=False
)

print("\nSubmission file created successfully!")
print("File name: submission.csv")

print("\nFirst 10 predictions:")
print(submission.head(10))

print("\nSubmission shape:")
print(submission.shape)

print("\nSurvival prediction distribution:")
print(submission["Survived"].value_counts())

# ==========================================
# 19. VERIFY SUBMISSION FILE
# ==========================================

print("\n======================================")
print("VERIFYING SUBMISSION FILE")
print("======================================")

# Load the saved submission
check_submission = pd.read_csv("submission.csv")

print("\nSubmission columns:")
print(check_submission.columns.tolist())

print("\nSubmission shape:")
print(check_submission.shape)

print("\nFirst 10 rows:")
print(check_submission.head(10))

print("\nMissing values:")
print(check_submission.isnull().sum())

print("\nUnique values in Survived:")
print(check_submission["Survived"].unique())

# Final checks
if list(check_submission.columns) == ["PassengerId", "Survived"]:
    print("\n✓ Column names are correct.")

if check_submission.shape[0] == len(test):
    print("✓ Correct number of passengers.")

if check_submission["Survived"].isnull().sum() == 0:
    print("✓ No missing predictions.")

if set(check_submission["Survived"].unique()).issubset({0, 1}):
    print("✓ Predictions contain only 0 and 1.")

print("\nSubmission verification completed!")