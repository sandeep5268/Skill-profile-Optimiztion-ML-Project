# Skill Profile Optimization for Interview Success

> A machine learning project that predicts a candidate's **resume score** based on their skill profile — helping job seekers understand what actually influences interview call rates.

---

## What This Project Is About

Getting interview calls is not just about luck. It depends on things like how many skills you have, whether you did internships, your educational background, the projects you've built, and whether you have a GitHub portfolio. But how much does each of these actually matter?

This project tries to answer that question using real data.

We built a desktop application (using Python and Tkinter) that takes a candidate's profile — their degree, years of experience, number of projects, certifications, skill count, internship experience, and GitHub presence — and predicts a **resume score** (out of 100). That resume score is then linked to how many interview calls a candidate is likely to receive.

The whole idea is to give candidates a data-backed way to understand their profile's strengths and weaknesses before they actually start applying.

---

## Project Highlights

- **End-to-end ML pipeline** — from raw CSV data to trained models to live predictions
- **Multiple regression models** compared side by side (Lasso, ElasticNet, Extra Trees, Stacking)
- **Role-based login system** — Admin trains models, User predicts on new data
- **MySQL-backed authentication** — credentials are stored safely, not hardcoded
- **Model persistence** — trained models are saved as `.pkl` files and reloaded automatically so you don't retrain every time
- **Visual EDA** — six charts generated automatically to understand the dataset
- **Actual vs Predicted plots** for every model, saved to the `results/` folder

---

## Folder Structure

```
Skill profile optimization for interview success/
│
├── main.py                          # Main application — GUI + ML logic
├── Database.txt                     # MySQL schema (run this to set up the DB)
│
├── Dataset/
│   ├── resume_skills_vs_interview_calls.csv   # Training dataset (200 records)
│   └── testdata.csv                           # Sample test data for predictions
│
├── models/                          # Saved trained models and encoders
│   ├── degree_encoder.pkl
│   ├── internship_encoder.pkl
│   ├── github_portfolio_encoder.pkl
│   ├── standard_scaler.pkl
│   ├── lasso_regressor_alpha_1.0.pkl
│   ├── elasticnet_regressor_alpha_1.0_l1_0.5.pkl
│   ├── ridge_regressor_alpha.pkl
│   ├── theilsen_regressor.pkl
│   ├── extra_trees_regressor.pkl
│   └── stacking_regressor.pkl
│
├── results/                         # Auto-generated result charts
│   ├── Lasso_Regressor_actual_vs_predicted.png
│   ├── Elastic_Net_Regressor_actual_vs_predicted.png
│   ├── Extra_Trees_Regressor_actual_vs_predicted.png
│   ├── Stacking_Regressor_actual_vs_predicted.png
│   └── regression_model_performance_comparison.png
│
├── BG_image/                        # Background images for the GUI
│   ├── background.jpg
│   └── bakground2.jpeg
│
└── resume_eda_plots.png             # EDA chart saved here
```

---

## Dataset Overview

The dataset contains **200 candidate records**, each with 10 columns:

| Column | Type | Description |
|---|---|---|
| `candidate_id` | String | Unique identifier (e.g., C001) — dropped during training |
| `degree` | Categorical | BTech, MTech, MCA, BSc |
| `years_experience` | Integer | 0 to 8 years |
| `projects_count` | Integer | Number of projects completed |
| `certifications` | Integer | Number of certifications |
| `skills_count` | Integer | Total skills listed on the resume |
| `internship` | Categorical | Yes / No |
| `github_portfolio` | Categorical | Yes / No |
| `resume_score` | Integer | Score out of 100 (**target variable**) |
| `interview_calls` | Integer | Number of interview calls received |

Average resume score in the dataset is **~90.6 / 100**, with candidates receiving **~5.8 interview calls** on average.

---

## How the Application Works

The application has two types of users — **Admin** and **User** — each with their own set of actions.

### Admin Flow (Train the Models)

1. **Upload Dataset** — Load the training CSV from the Dataset folder
2. **Preprocess Dataset** — Encodes categorical columns, scales numeric features, saves encoders
3. **Show EDA** — Generates 6 exploratory charts and saves them as PNG
4. **Train-Test Split** — Splits data 80/20 with a fixed random seed for reproducibility
5. **Train individual models** — Click any button to train that specific model
6. **Model Comparison** — Bar chart comparing all trained models by MAE, RMSE, and R²

### User Flow (Predict New Candidates)

1. **Upload Test Data** — Load a new CSV with candidate profiles
2. **Predict Resume Scores** — Runs the saved Stacking Regressor on the test data
3. **View Results** — Predictions are displayed in the output box and appended to the loaded dataframe

---

## Machine Learning Models Used

| Model | Why It Was Used |
|---|---|
| **Lasso Regressor** | Adds L1 penalty — good at removing irrelevant features by zeroing out their weights |
| **ElasticNet Regressor** | Combines L1 and L2 — balances feature selection and coefficient shrinkage |
| **Extra Trees Regressor** | Ensemble of randomized decision trees — handles non-linear patterns well |
| **Stacking Regressor** | Uses Ridge + TheilSen as base learners and Ridge as meta-learner — combines model strengths for better generalization |

Each model is evaluated using **MAE**, **MSE**, **RMSE**, and **R² Score**.

The **Stacking Regressor** is used specifically for final predictions on new test data, as it typically gives the best performance by combining the outputs of multiple learners.

---

## Preprocessing Pipeline

The preprocessing logic handles both **training** and **inference** cleanly:

1. **Drop `candidate_id`** — it carries no predictive value
2. **Label Encode** categorical columns (`degree`, `internship`, `github_portfolio`) — each encoder is saved to disk after fitting and reloaded during prediction
3. **Standard Scaling** — applied to all numeric features; the scaler is also saved and reloaded
4. **Target variable** — `resume_score` is separated before scaling (it's already dropped from X)

This ensures there is **no data leakage** — the test/prediction data always uses encoders and scalers fitted only on training data.

---

## EDA — What the Charts Show

When you click "Show EDA", 6 charts are generated and saved:

1. **Distribution of Resume Score** — how scores are spread across candidates
2. **Degree vs Resume Score** — does your degree type actually matter?
3. **Experience vs Resume Score** — does more experience mean higher score?
4. **Internship Distribution** — how many candidates had internship experience?
5. **Skills Count vs Resume Score (by GitHub)** — combining skills and GitHub presence
6. **Interview Calls vs Resume Score** — the direct link between score and calls received

---

## Database Setup

This project uses **MySQL** for user authentication. Run the SQL in `Database.txt` to set up the schema:

```sql
CREATE DATABASE IF NOT EXISTS sparse_db;
USE sparse_db;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role ENUM('Admin', 'User') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

The application connects using `pymysql`. Update the credentials in `main.py` if needed:

```python
def connect_db():
    return pymysql.connect(host='localhost', user='root', password='your_password', database='sparse_db')
```

---

## Requirements

Make sure you have the following installed:

```
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
Pillow
pymysql
tkinter  (comes built-in with Python on Windows)
```

Install all at once:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib Pillow pymysql
```

---

## How to Run

```bash
# 1. Clone / download the project
# 2. Set up the MySQL database using Database.txt
# 3. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib Pillow pymysql

# 4. Run the app
python main.py
```

---

## Results Summary

All model comparison charts are stored in the `results/` folder after training. The bar chart at the end normalizes MAE and RMSE (by inverting them) so that **higher bars always mean better performance** — making it easy to compare at a glance.

---

## Key Design Decisions

- **Model caching** — if a `.pkl` file already exists, the model is loaded instead of retrained. This saves time during repeated runs.
- **Predictions are clipped** to `[0, 100]` — since resume scores are bounded, any model predictions that go outside this range are clipped rather than thrown out.
- **Role-based access** — Admin controls the training pipeline; Users only get access to prediction functions. This simulates how a real deployment would work.
- **Separate test data path** — The `testdata.csv` in the Dataset folder mirrors the training schema but without `resume_score`, simulating a real use case where you receive new candidate data.

