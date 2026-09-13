# Student Performance Prediction

An end-to-end Machine Learning regression project that predicts a student's **Math Score** based on demographic and academic background features, deployed via a **Flask** web application.

Deployed link : https://student-performance-prediction-o47s.onrender.com

---

## Project Flow

```
Student Data Input
       │
       ▼
1. Data Ingestion      → artifacts/train.csv, test.csv
       │
       ▼
2. Data Transformation → artifacts/preprocessor.pkl
   ├── Categorical → OneHotEncoder
   └── Numerical  → StandardScaler
       │
       ▼
3. Model Training      → artifacts/model.pkl
   (7 models compared, best selected by R²)
       │
       ▼
4. Flask Web App
   User enters details → Preprocessor → Model → Predicted Math Score
```

---

## Project Structure

```
student-performance-prediction/
│
├── data/
│   └── stud.csv                      # Raw dataset (1000 rows, 8 columns)
│
├── notebooks/
│   ├── 1_EDA.ipynb                   # Exploratory Data Analysis
│   └── 2_Model_Training.ipynb        # Interactive model training & evaluation
│
├── src/
│   ├── __init__.py
│   ├── exception.py                  # Custom exception handler
│   ├── logger.py                     # Centralized logging
│   ├── utils.py                      # save/load object, evaluate_models
│   │
│   ├── components/
│   │   ├── data_ingestion.py         # Reads raw CSV, creates train/test split
│   │   ├── data_transformation.py    # ColumnTransformer pipeline
│   │   └── model_trainer.py          # Trains & selects best model
│   │
│   └── pipeline/
│       └── prediction_pipeline.py    # PredictPipeline + CustomData classes
│
├── artifacts/                        # Auto-generated during training
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── templates/
│   └── index.html                    # Glassmorphism UI form
│
├── app.py                            # Flask application
├── requirements.txt
├── setup.py
└── README.md
```

---

## Dataset

- **Source**: [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Records**: 1,000 students
- **Target**: `math_score` (continuous — regression problem)

| Feature | Type | Description |
|---|---|---|
| gender | Categorical | male / female |
| race_ethnicity | Categorical | group A–E |
| parental_level_of_education | Categorical | high school to master's degree |
| lunch | Categorical | standard / free-reduced |
| test_preparation_course | Categorical | none / completed |
| reading_score | Numerical | 0–100 |
| writing_score | Numerical | 0–100 |

---

## Preprocessing

**ColumnTransformer** applies two transformations in parallel:

```
           X (features)
               │
    ┌──────────┴──────────┐
    ▼                     ▼
Categorical           Numerical
(5 columns)           (2 columns)
    │                     │
OneHotEncoder       StandardScaler
    │                     │
    └──────────┬──────────┘
               ▼
        Transformed X
```

- **OneHotEncoder** converts categorical strings to binary columns (e.g. `gender_male`, `gender_female`)
- **StandardScaler** standardizes numerical features → mean=0, std=1

**Important distinction:**
- Training: `preprocessor.fit_transform(X_train)` — learns AND applies transformation
- Inference: `preprocessor.transform(X_new)` — applies ONLY, using learned parameters

---

## Models Evaluated

| Model | Notes |
|---|---|
| Linear Regression | Baseline; strong performer on this dataset |
| Ridge | L2 regularized linear regression |
| Lasso | L1 regularized linear regression |
| K-Neighbors Regressor | Distance-based |
| Decision Tree | Single tree |
| Random Forest | Ensemble of trees |
| Gradient Boosting | Sequential error correction |
| XGBRegressor | Optimized gradient boosting |
| CatBoosting Regressor | Handles categoricals natively |
| AdaBoost Regressor | Adaptive boosting |

**Best model selected automatically** by highest test R² score.

### Evaluation Metrics

| Metric | Formula | Goal |
|---|---|---|
| MAE | Mean Absolute Error | Lower is better |
| RMSE | √MSE — penalizes large errors more | Lower is better |
| R² | Proportion of variance explained | Higher is better (1.0 = perfect) |

---

## Setup & Run

### 1. Clone or open the project

```bash
cd student-performance-prediction
```

### 2. Install dependencies (editable install)

```bash
pip install -r requirements.txt
```

### 3. Run the training pipeline

```bash
python3 -m src.components.data_ingestion
```

This generates:
- `artifacts/preprocessor.pkl`
- `artifacts/model.pkl`
- `artifacts/train.csv` & `artifacts/test.csv`

### 4. Launch the Flask app

```bash
python3 app.py
```

Open your browser at → [http://localhost:5000](http://localhost:5000)

---

## Interview Q&A

### Why is this a regression and not classification problem?
> The target `math_score` is a continuous numerical value (e.g. 72.43), not a discrete category. Regression predicts continuous outputs; classification predicts discrete labels.

### Why do we save both the preprocessor and the model?
> At inference time, new inputs must go through the **exact same transformations** learned from training data. Saving `preprocessor.pkl` ensures consistency — we call `transform()` (not `fit_transform()`) at prediction time.

### Why use ColumnTransformer?
> Our dataset has mixed types — categorical and numerical columns need different transformations. `ColumnTransformer` applies them simultaneously in a clean, reproducible pipeline that can be saved and reloaded.

### What is StandardScaler doing?
> It rescales each numerical feature to have mean ≈ 0 and standard deviation = 1. This prevents features with larger ranges from dominating models like KNN or regularized linear models.

### What is OneHotEncoder doing?
> Converts categorical text values into binary columns. ML models work with numbers, not strings — OHE creates a column per category with 0/1 values.

### Resume Summary
> **Problem**: Predict students' math scores based on demographic, educational, and prior assessment information.
> **EDA**: Analyzed score distributions and relationships between student characteristics and performance.
> **Preprocessing**: Applied One-Hot Encoding to categorical features and StandardScaler to numerical features using ColumnTransformer.
> **Modeling**: Compared multiple regression algorithms (Linear Regression, Random Forest, XGBoost, CatBoost, etc.) using MAE, RMSE and R².
> **Deployment**: Saved the preprocessing pipeline and trained model; served predictions through a Flask web application.

---

## Results

Best Model R² Score: **~0.88** on held-out test set.
