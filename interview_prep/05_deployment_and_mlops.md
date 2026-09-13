# Module 5: Production Deployment, Serving & MLOps

## 1. End-to-End Inference Lifecycle
When a user visits the web portal and requests a prediction, the request follows a strictly orchestrated pathway:

```
[ Web Browser ]
      │  HTTP POST (JSON/Form: gender, lunch, reading_score, etc.)
      ▼
[ Flask Controller (app.py) ]
      │  Maps payload to CustomData instance
      ▼
[ CustomData (prediction_pipeline.py) ]
      │  Constructs 1-row Pandas DataFrame matching training schema
      ▼
[ PredictPipeline.predict() ]
      ├── 1. Load artifacts/preprocessor.pkl via load_object()
      ├── 2. data_scaled = preprocessor.transform(features)  <-- ZERO LEAKAGE
      ├── 3. Load artifacts/model.pkl via load_object()
      └── 4. raw_prediction = model.predict(data_scaled)
      │
      ▼
[ Flask Response ]
      │  Formats result: round(raw_prediction[0], 2)
      ▼
[ HTML Template Render (templates/index.html) ]
```

---

## 2. Serving Components Deep Dive

### The `CustomData` Adapter
A common failure in ML deployment is **schema mismatch** (e.g., frontend sending `'readingScore'` while model expects `'reading_score'`, or incorrect column order).
The `CustomData` class enforces strict contract validation:
```python
class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        return pd.DataFrame({
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score]
        })
```

---

## 3. Serialization: Pickle vs. Dill vs. Joblib

| Library | Strengths | Weaknesses | Why used here |
|---|---|---|---|
| **`pickle`** | Python standard library; zero extra dependencies. | Struggles with lambdas, dynamic functions, closures. | Used in `src/utils.py` for standard Scikit-Learn transformers and linear/tree models. |
| **`dill`** | Extends `pickle`; serializes almost any Python bytecode, functions, closures. | Slightly slower; security risks if unpickling untrusted files. | Added in `requirements.txt` to support complex custom transformations if needed. |
| **`joblib`** | Optimized for numpy arrays and large scikit-learn models using disk memory mapping. | External library; file format less portable across non-Python environments. | Industry standard alternative for multi-gigabyte models. |

> [!CAUTION]
> **Production Security Note**: Serialized pickle files execute arbitrary code during unpickling (`__reduce__`). In enterprise systems, models must be loaded from authenticated, encrypted object stores (AWS S3 with IAM roles, GCP Cloud Storage) or converted to open formats like **ONNX**.

---

## 4. Packaging & Reproducibility (`setup.py`)
- The project includes `setup.py` with `find_packages()`.
- Adding `-e .` in `requirements.txt` installs the entire repository in **editable development mode**.
- *Interview Advantage*: This allows any script across subdirectories to import cleanly (`from src.logger import logging`) without hacky `sys.path.append(os.path.abspath('..'))` statements.

---

## 5. Scaling to Enterprise Production & MLOps

If an interviewer asks: *"How would you take this project from a local prototype to a production system serving 100,000 requests/day at Google/Amazon?"* Provide this roadmap:

### 1. Web Server Productionization (WSGI/ASGI)
- **Local**: `app.run(debug=True, port=8080)` uses Flask's single-threaded built-in dev server.
- **Production**: Wrap with **Gunicorn** or **Uvicorn** behind an **NGINX** reverse proxy:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:8080 app:app
  ```
- Or migrate to **FastAPI** with Pydantic typing for automatic Swagger OpenAPI documentation, async handling, and input validation.

### 2. Containerization (Docker)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
```

### 3. Drift Monitoring & Model Maintenance
1. **Data Drift (Covariate Shift)**:
   - *Problem*: Future student cohorts have significantly lower reading/writing preparation or different demographic balances.
   - *Detection*: Run **Kolmogorov-Smirnov (KS) tests** or calculate **Population Stability Index (PSI)** on weekly input batches vs. training baseline.
2. **Concept Drift**:
   - *Problem*: Exam grading rubrics change, meaning a reading score of 70 no longer correlates with math 70.
   - *Detection*: Track rolling **MAE** against ground-truth exam scores released at end-of-term.
3. **Automated CI/CD Retraining**:
   - GitHub Actions / Airflow DAG triggers `data_ingestion.py` -> `data_transformation.py` -> `model_trainer.py`. If new model test $R^2 > \text{current production } R^2$, auto-promote model artifact to registry (MLflow / S3).
