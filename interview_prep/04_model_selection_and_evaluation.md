# Module 4: Model Selection, Evaluation & Algorithmic Deep Dive

## 1. Algorithmic Candidates Evaluated
In `src/components/model_trainer.py` and `notebooks/2_Model_Training.ipynb`, 7+ diverse regression families were systematically trained and compared:

| Algorithm Family | Specific Models Evaluated | Core Mechanism |
|---|---|---|
| **Linear Models** | Linear Regression, Ridge ($L_2$), Lasso ($L_1$) | Solves for optimal hyperplane coefficients $\hat{y} = X\beta$ minimizing squared residuals. |
| **Instance / Distance** | K-Neighbors Regressor (KNN) | Averages the target values of the $k$ nearest Euclidean neighbors. |
| **Non-Parametric Trees** | Decision Tree Regressor | Recursively splits feature space to minimize MSE in terminal leaf nodes. |
| **Bagging Ensemble** | Random Forest Regressor | Constructs $B$ independent deep trees on bootstrap samples with feature subspace sampling; averages predictions. |
| **Boosting Ensembles** | AdaBoost, Gradient Boosting, XGBoost, CatBoost | Sequentially fits shallow trees to the negative gradient (residuals) of the loss function. |

---

## 2. Benchmark Results & Performance Ranking

On the held-out test dataset ($200$ test observations), models achieved the following performance:

| Rank | Model Name | Test $R^2$ Score | Test RMSE | Test MAE | Complexity / Latency |
|---|---|---|---|---|---|
| **1** | **Linear Regression** | **~0.8804** | **~5.39** | **~4.21** | Extremely low ($<1\text{ ms}$) |
| 2 | Ridge Regression | ~0.8805 | ~5.39 | ~4.21 | Extremely low |
| 3 | CatBoost Regressor | ~0.8516 | ~5.72 | ~4.46 | Medium |
| 4 | Random Forest | ~0.8532 | ~5.70 | ~4.59 | Higher memory footprint |
| 5 | Gradient Boosting | ~0.8490 | ~5.77 | ~4.51 | Medium |
| 6 | XGBoost Regressor | ~0.8278 | ~6.16 | ~4.88 | Low inference, higher train time |
| 7 | Decision Tree | ~0.7300 | ~7.90 | ~6.20 | Prone to extreme variance |

*(Note: Threshold check in `model_trainer.py`: `if best_model_score < 0.6: raise CustomException(...)` ensures no degraded model is ever pushed to production).*

---

## 3. The Core Interview Question: "Why did Linear Regression beat or match complex Tree Ensembles?"

Interviewers at FAANG and top data teams ask this question to separate script-runners from true machine learning scientists.

### Reason 1: High Intrinsic Linearity of the Feature Space
Math performance is fundamentally tied to overall cognitive and academic foundation skills. The correlation between `reading_score`, `writing_score`, and `math_score` is $>0.80$. A linear hyperplane cleanly maps this relationship:
$$\text{math\_score} = \beta_0 + \beta_1(\text{reading}) + \beta_2(\text{writing}) + \sum \beta_i(\text{categorical}_i)$$
Adding non-linear axis-aligned splits (Decision Trees) creates jagged step-functions that approximate an already smooth linear plane less efficiently.

### Reason 2: Sample Size Constraint ($N = 1,000$)
- Tree ensembles like Random Forest and XGBoost have hundreds or thousands of hyperparameters and split decisions. With only 800 training rows, complex trees tend to slightly overfit localized noise in the training set (training $R^2 \approx 0.98$, test $R^2 \approx 0.85$).
- Linear Regression has very high bias and low variance; with only $\sim 19$ parameters to estimate, it cannot overfit easily on 800 rows.

### Reason 3: Occam's Razor & Enterprise Pragmatism
In machine learning engineering, **Occam's Razor** dictates: *When two models have comparable performance, choose the simpler model.*
- **Interpretability**: Linear regression coefficients can be directly audited and explained to school boards (e.g., "Holding reading constant, completing test prep adds +2.1 points to math").
- **Compute Efficiency**: Linear dot-product $\mathbf{w}^T \mathbf{x}$ executes in microseconds and requires almost zero server memory compared to serializing 500 gradient-boosted trees.
- **Fairness & Compliance**: Regulated domains (education, credit, health) demand model transparency to avoid unintended discriminatory biases.

---

## 4. Evaluation Metrics Deep Dive

### 1. Coefficient of Determination ($R^2$ Score)
$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
- **Intuition**: Quantifies the percentage of variance in `math_score` captured by the model compared to a baseline model that always predicts the mean $\bar{y}$.
- **Score of ~0.88**: Means **88%** of the variability in math scores is explained by our demographic and academic features. Only 12% remains unexplained residual variance.

### 2. Adjusted $R^2$
$$R^2_{\text{adj}} = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$$
- Where $n$ is sample size and $p$ is the number of predictors.
- *Why it matters*: Standard $R^2$ monotonically increases whenever a new feature is added, even if that feature is completely random noise. Adjusted $R^2$ penalizes redundant features.

### 3. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Intuition**: The average absolute point error. Our MAE is $\approx 4.2$ points. On a 100-point scale, an average error of 4.2 points is easily understood by non-technical academic stakeholders.

### 4. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
- **Intuition**: Because errors are squared before averaging, RMSE penalizes large outlier errors disproportionately compared to MAE.
- Our $\text{RMSE} \approx 5.39$. The ratio $\frac{\text{RMSE}}{\text{MAE}} \approx \frac{5.39}{4.21} \approx 1.28$ confirms there are very few extreme prediction blunders.

---

## 5. Hyperparameter Tuning & Cross-Validation Strategy
- In the baseline script, `evaluate_models` runs default parameters.
- In `notebooks/2_Model_Training.ipynb` and production extensions:
  - **K-Fold Cross-Validation** ($k=5$) prevents split bias.
  - **GridSearchCV / RandomizedSearchCV** optimizes parameters:
    - *Random Forest*: `n_estimators`, `max_depth`, `min_samples_split`.
    - *XGBoost*: `learning_rate` ($\eta$), `max_depth`, `subsample`, `colsample_bytree`.
    - *Ridge/Lasso*: Regularization strength $\alpha$.
