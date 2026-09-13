# Module 2: Exploratory Data Analysis (EDA) & Statistical Insights

## 1. EDA Methodology & Data Integrity
Before applying algorithms, rigorous data validation was performed in `notebooks/1_EDA.ipynb`:
- **Sample Size**: $N = 1,000$ rows, 8 columns.
- **Missing Value Audit**: `df.isna().sum()` yielded `0` across all 8 columns. No imputation was required.
- **Duplicate Audit**: `df.duplicated().sum()` yielded `0` records.
- **Data Types**: 5 `object` (string categorical) columns, 3 `int64` numerical columns.

---

## 2. Univariate Analysis: Distributions & Central Tendencies

| Metric | Math Score | Reading Score | Writing Score |
|---|---|---|---|
| **Mean** | 66.09 | 69.17 | 68.05 |
| **Standard Deviation** | 15.16 | 14.60 | 15.20 |
| **Median (50%)** | 66.00 | 70.00 | 69.00 |
| **Min / Max** | 0.0 / 100.0 | 17.0 / 100.0 | 10.0 / 100.0 |
| **IQR (Q3 - Q1)** | $77.0 - 57.0 = 20.0$ | $79.0 - 59.0 = 20.0$ | $79.0 - 57.75 = 21.25$ |

### Distribution Characteristics:
- All three exam scores exhibit **near-Gaussian (normal) bell curve distributions**.
- **Slight Negative Skewness (Left-tailed)**: Most students cluster around the 60–80 score range, with an extended tail toward lower scores (e.g., math score has a minimum of 0).
- **Outlier Detection**: Using the $1.5 \times \text{IQR}$ rule, only a few extreme low scores exist in the lower tail. They represent authentic student failures rather than data entry corruptions, and were thus retained to preserve real-world variance.

---

## 3. Bivariate & Multivariate Statistical Findings

### Insight 1: Collinearity Between Skills (The Strongest Predictors)
- Pearson correlation coefficient:
  - $\rho(\text{reading}, \text{writing}) \approx 0.954$ (Extremely high collinearity)
  - $\rho(\text{math}, \text{reading}) \approx 0.817$
  - $\rho(\text{math}, \text{writing}) \approx 0.802$
- **Interview Takeaway**: Reading and writing scores are the dominant drivers of the math score prediction. The high correlation between reading and writing indicates shared verbal comprehension abilities.

### Insight 2: Socioeconomic Factor — Lunch Type
- **Standard Lunch**: Mean Math Score $\approx 70.06$
- **Free/Reduced Lunch**: Mean Math Score $\approx 58.86$
- **Delta**: $\approx 11.2\text{ points}$ difference!
- **Statistical Significance**: A two-sample independent t-test yields $p < 0.001$, demonstrating that nutritional access and socioeconomic stability strongly correlate with standardized test outcomes.

### Insight 3: Test Preparation Course
- **Completed**: Mean Math Score $\approx 69.70$
- **None**: Mean Math Score $\approx 64.08$
- **Delta**: $+5.62\text{ points}$ advantage in math. (The boost is even larger in writing: $+9.9\text{ points}$).
- **Interview Takeaway**: Test prep is an actionable lever for schools — unlike socioeconomic status, test preparation can be directly provided as a proactive intervention.

### Insight 4: Parental Level of Education
- Ordered hierarchy of performance:
  $$\text{Master's Degree} > \text{Bachelor's Degree} > \text{Associate's Degree} > \text{Some College} > \text{High School} > \text{Some High School}$$
- Students whose parents held a Master’s degree scored an average of $\approx 73.6$ in math, compared to $\approx 62.1$ for parents with only high school education.
- One-way ANOVA test reveals statistically significant between-group variance ($p < 0.01$).

### Insight 5: Gender-Specific Performance Trends
- **Math Score**: Males scored higher on average ($\approx 68.7$ vs. $\approx 63.7$ for females).
- **Reading & Writing Scores**: Females outperformed males ($\text{Female Writing Mean} \approx 72.5$ vs. $\text{Male Writing Mean} \approx 63.3$).
- **Overall Total/Average Score**: When combined ($\text{Math} + \text{Reading} + \text{Writing}$), female students had a slightly higher composite average.

---

## 4. Feature Engineering Exploration
During exploratory analysis:
- Created composite features:
  - `total_score = math_score + reading_score + writing_score`
  - `average_score = total_score / 3`
- *Caveat for Modeling*: `total_score` and `average_score` **cannot** be used as predictors for `math_score` because they mathematically contain `math_score`. Doing so would introduce catastrophic target leakage ($R^2 = 1.0$, invalid in real-world testing). Only `reading_score` and `writing_score` were kept as numerical predictors.
