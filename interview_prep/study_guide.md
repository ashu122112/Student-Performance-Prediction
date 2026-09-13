# Student Performance Prediction — Master Study Curriculum

Welcome to the comprehensive interview preparation suite for the **Student Performance Prediction** end-to-end Machine Learning project. This curriculum is structured to prepare you for Data Science (DS), Machine Learning Engineer (MLE), and Data Analyst (DA) interviews across top-tier tech companies.

---

## 🗺️ Interview Study Roadmap

Use the specialized modules below to master each facet of the system:

1. **[Module 1: Project Overview & Business Impact](file:///Users/ashutoshsingh/project_ML/interview_prep/01_project_overview_and_business_impact.md)**
   - Problem framing, business objective, stakeholder mapping (counselors, school boards, students).
   - Translating ML metrics ($R^2$, MAE) into business ROI (early interventions, pass-rate lift).
   - Ethical considerations and bias monitoring in academic ML.

2. **[Module 2: EDA & Statistical Insights](file:///Users/ashutoshsingh/project_ML/interview_prep/02_eda_and_statistical_insights.md)**
   - Univariate analysis (normal distribution, skewness, outlier checks).
   - Bivariate & multivariate analysis: The 11.2-point lunch subsidy gap, test prep efficacy, parental education trends.
   - Hypothesis testing: Independent two-sample t-tests and one-way ANOVA.
   - Multicollinearity between reading and writing scores ($\rho \approx 0.95$).

3. **[Module 3: Data Engineering & Production Pipeline](file:///Users/ashutoshsingh/project_ML/interview_prep/03_data_engineering_and_pipeline.md)**
   - Modular architecture: Separation of concerns between Ingestion, Transformation, and Training.
   - `ColumnTransformer` mechanics: `StandardScaler` for numerical, `OneHotEncoder` for categorical.
   - Prevention of **Data Leakage** (`fit_transform` on train vs. `transform` on test/inference).
   - Enterprise logging and custom exception handling with runtime stack trace introspection.

4. **[Module 4: Model Selection & Algorithmic Evaluation](file:///Users/ashutoshsingh/project_ML/interview_prep/04_model_selection_and_evaluation.md)**
   - Systematic benchmarking of 7 algorithms (Linear, Ridge, Lasso, KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, CatBoost).
   - **The Critical Interview Question**: Why Linear Regression / Ridge won ($R^2 \approx 0.88$) over tree ensembles (Occam's razor, intrinsic linearity, low sample size $N=1000$).
   - Comprehensive metric breakdown: $R^2$, Adjusted $R^2$, MAE, MSE, and RMSE.

5. **[Module 5: Production Deployment & MLOps](file:///Users/ashutoshsingh/project_ML/interview_prep/05_deployment_and_mlops.md)**
   - Real-time inference lifecycle: Flask controller, `CustomData` schema adapter, `PredictPipeline`.
   - Serialization comparison: `pickle` vs. `dill` vs. `joblib` vs. `ONNX`.
   - Production readiness: WSGI/Gunicorn, Docker containerization, and data/concept drift monitoring.

---

## ⚡ High-Priority Files for Last-Minute Prep

- 🚀 **[Quick Revision Cheat Sheet](file:///Users/ashutoshsingh/project_ML/interview_prep/quick_revision.md)**:
  *Read this 10 minutes before your interview!* Contains the 60-second elevator pitch, vital numbers, architecture diagram, metric formulas, and rapid-fire flashcards.
- 🎯 **[Comprehensive Interview Questions & Answers](file:///Users/ashutoshsingh/project_ML/interview_prep/interview_questions.md)**:
  19 in-depth questions and answers categorized across Behavioral, EDA/Stats, Feature Engineering, ML Modeling, Python Internals, and System Design.
