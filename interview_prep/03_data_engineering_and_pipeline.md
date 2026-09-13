# Module 3: Data Engineering & Production Pipeline Architecture

## 1. Modular Software Design Pattern
Unlike typical academic projects that execute everything in a single Jupyter Notebook (`.ipynb`), this project is engineered using production-grade **Object-Oriented Programming (OOP)** and separation of concerns:

```
src/
├── exception.py               # Custom traceback-aware exception hierarchy
├── logger.py                  # Thread-safe rotating timestamped logger
├── utils.py                   # Reusable I/O utilities (save_object, load_object, evaluate_models)
├── components/                # Pipeline stages (Extract, Transform, Train)
│   ├── data_ingestion.py      # Data extraction and train/test partition
│   ├── data_transformation.py # Sklearn ColumnTransformer & preprocessor persistence
│   └── model_trainer.py       # Multi-model benchmarking & artifact export
└── pipeline/                  # Serving / Inference layer
    └── prediction_pipeline.py # CustomData schema validation & PredictPipeline
```

---

## 2. Component 1: Data Ingestion (`data_ingestion.py`)

### Responsibilities:
1. Load source data from `data/stud.csv`.
2. Ensure reproducible output directory structures (`artifacts/`) using `os.makedirs(..., exist_ok=True)`.
3. Save an untouched snapshot of raw data (`artifacts/data.csv`) for data versioning/auditing.
4. Perform an **80/20 train/test split** with a fixed seed:
   ```python
   train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
   ```
5. Export `artifacts/train.csv` (800 rows) and `artifacts/test.csv` (200 rows).

### Design Pattern: Dataclasses for Configuration
Instead of hardcoding filepaths inside methods, a dedicated `@dataclass` encapsulates configuration:
```python
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
```
*Why this matters in interviews*: Separating configuration from execution logic adheres to the **Single Responsibility Principle (SRP)** and allows runtime overrides (e.g., pointing to S3 or GCS buckets in cloud deployments).

---

## 3. Component 2: Data Transformation (`data_transformation.py`)

### The Preprocessing Engine: `ColumnTransformer`
Tabular datasets contain mixed types requiring divergent mathematical preprocessing. `ColumnTransformer` coordinates parallel transformations:

```
                      Input Features
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    Numerical Columns              Categorical Columns
   ['writing_score',               ['gender', 'race_ethnicity',
    'reading_score']                'parental_level_of_education',
            │                       'lunch', 'test_preparation_course']
            ▼                               │
      StandardScaler()                      ▼
   (Zero mean, unit variance)        OneHotEncoder()
            │                     (Binary dummy variables)
            └───────────────┬───────────────┘
                            ▼
               Fused Transformed Matrix (19 features)
```

### Transformation Math & Justification:
1. **StandardScaler**:
   $$z = \frac{x - \mu}{\sigma}$$
   - *Why*: Centers writing and reading scores so their variance equals 1. Prevents scale bias in distance-based models (KNN) and gradient descent optimization in linear models.
2. **OneHotEncoder**:
   - *Why*: Encodes nominal strings into separate binary indicator columns. Prevents the false ordinal hierarchy that `LabelEncoder` would introduce (e.g., assigning 1, 2, 3 to race groups implying that Group 3 > Group 1).

### Preventing Data Leakage (The Golden Rule of ML Interviews)
```python
# TRAINING SET: Learn parameters (mean, std, category levels) AND transform
input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

# TESTING SET: ONLY transform using the parameters learned from training
input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
```
> [!IMPORTANT]
> **Data Leakage Answer**: If you run `fit_transform()` on the entire dataset or on the test set, your scaler calculates the test set's mean ($\mu_{test}$) and variance ($\sigma^2_{test}$). In the real world, future data does not exist yet. Fitting on test data produces over-optimistic test scores that fail in production.

### Matrix Concatenation:
The transformed feature array and the target array are bound into contiguous numpy arrays using `np.c_`:
```python
train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
```

### Preprocessor Serialization:
The fitted transformer is serialized to disk as `artifacts/preprocessor.pkl`.

---

## 4. Enterprise Infrastructure: Logging & Exception Handling

### Custom Traceback Exception (`src/exception.py`)
Standard Python exceptions print only generic messages. The project overrides `Exception` to extract deep runtime introspection:
```python
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"
```
*Value*: Instant debugging in production log files without attaching an interactive debugger.

### Centralized Logging (`src/logger.py`)
- Automatically initializes timestamped log files in a `logs/` directory.
- Captures timestamps, execution line numbers, module names, and severity levels (`INFO`, `WARNING`, `ERROR`).
