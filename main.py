
import tkinter
from tkinter import filedialog
import pymysql
# GUI
from tkinter import messagebox, Text, END, Label, Scrollbar
from tkinter import filedialog
import tkinter as tk
# GUI Image Background
from PIL import Image, ImageTk
from tkinter import messagebox, filedialog
# Core
import os
from pathlib import Path
import joblib

# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & splitting
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import (
    Ridge,
    ElasticNet,
    Lasso,
    TheilSenRegressor
)
from sklearn.ensemble import (
    ExtraTreesRegressor,
    StackingRegressor
)

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



main = tkinter.Tk()
main.configure(bg='#f0f8ff')  # Light background
main.title("Resume Skills vs Interview Calls") 
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()

# Set window size to full screen
main.geometry(f"{screen_width}x{screen_height}")

global dataset, X, y
global X_train, X_test, y_train, y_test

MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset)+"\n\n")


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_data(df, is_train=True):
    global X, y

    df = df.copy()

    # -----------------------------
    # Drop ID Column
    # -----------------------------
    if 'candidate_id' in df.columns:
        df.drop(columns=['candidate_id'], inplace=True)

    # -----------------------------
    # Encode Categorical Features
    # -----------------------------
    categorical_cols = ['degree', 'internship', 'github_portfolio']

    for col in categorical_cols:
        encoder_path = os.path.join(MODEL_DIR, f"{col}_encoder.pkl")

        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            joblib.dump(le, encoder_path)
        else:
            le = joblib.load(encoder_path)
            df[col] = le.transform(df[col].astype(str))

    # -----------------------------
    # Feature Matrix
    # -----------------------------
    X = df.drop(columns=['resume_score'])

    # -----------------------------
    # Standard Scaling
    # -----------------------------
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler_path = os.path.join(MODEL_DIR, 'standard_scaler.pkl')

    if is_train:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    # -----------------------------
    # Target
    # -----------------------------
    y = df['resume_score']

    text.insert(
        END,
        f"Preprocessing complete.\nFeatures: {X.shape[1]}, Records: {X.shape[0]}\n\n"
    )
    text.see(END)

    return X, y


def perform_eda(df):
    global dataset

    text.insert(END, "Starting EDA...\n")
    text.see(END)

    plt.figure(figsize=(18, 12))

    # -----------------------------
    # Plot 1: Target Distribution
    # -----------------------------
    plt.subplot(2, 3, 1)
    sns.histplot(df['resume_score'], kde=True)
    plt.title('Distribution of Resume Score')

    # -----------------------------
    # Plot 2: Degree vs Resume Score
    # -----------------------------
    plt.subplot(2, 3, 2)
    sns.boxplot(x='degree', y='resume_score', data=df)
    plt.title('Degree vs Resume Score')

    # -----------------------------
    # Plot 3: Years of Experience vs Resume Score
    # -----------------------------
    plt.subplot(2, 3, 3)
    sns.scatterplot(x='years_experience', y='resume_score', data=df)
    plt.title('Experience vs Resume Score')

    # -----------------------------
    # Plot 4: Internship Distribution
    # -----------------------------
    plt.subplot(2, 3, 4)
    sns.countplot(x='internship', data=df)
    plt.title('Internship Distribution')

    # -----------------------------
    # Plot 5: Skills Count vs Resume Score
    # -----------------------------
    plt.subplot(2, 3, 5)
    sns.scatterplot(
        x='skills_count',
        y='resume_score',
        hue='github_portfolio',
        data=df
    )
    plt.title('Skills Count vs Resume Score')

    # -----------------------------
    # Plot 6: Interview Calls vs Resume Score
    # -----------------------------
    plt.subplot(2, 3, 6)
    sns.boxplot(x='interview_calls', y='resume_score', data=df)
    plt.title('Interview Calls vs Resume Score')

    plt.tight_layout()
    plt.savefig('resume_eda_plots.png')
    text.insert(END, "EDA plots saved as 'resume_eda_plots.png'\n")
    text.see(END)

    plt.show()
    text.insert(END, "EDA completed successfully.\n\n")
    text.see(END)


def split_train_test(X, y, test_size=0.2, random_state=42):
    global X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    text.insert(END, f"X_train: {X_train.shape}, X_test: {X_test.shape}\n")
    text.insert(END, f"y_train: {y_train.shape}, y_test: {y_test.shape}\n\n")
    text.see(END)



regression_metrics_df = pd.DataFrame(columns=['Algorithm', 'MAE', 'MSE', 'RMSE', 'R2'])

def Calculate_Regression_Metrics(algorithm, y_pred, y_test):
    
    global regression_metrics_df 

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    regression_metrics_df.loc[len(regression_metrics_df)] = [algorithm, mae, mse, rmse, r2]

    text.insert(END, f"{algorithm} Mean Absolute Error (MAE): {mae:.4f}\n")
    text.insert(END, f"{algorithm} Mean Squared Error (MSE): {mse:.4f}\n")
    text.insert(END, f"{algorithm} Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    text.insert(END, f"{algorithm} R² Score: {r2:.4f}\n\n")
    text.see(END)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Resume Score")
    plt.ylabel("Predicted Resume Score")
    plt.title(f"{algorithm} - Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig(f"results/{algorithm.replace(' ', '_')}_actual_vs_predicted.png")
    text.insert(END, f"{algorithm} Actual vs Predicted plot saved.\n\n")
    text.see(END)
    plt.show()



def train_elasticnet_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    alpha=1.0,
    l1_ratio=0.5,
    max_iter=2000
):
    
    model_path = os.path.join(
        MODEL_DIR,
        f'elasticnet_regressor_alpha_{alpha}_l1_{l1_ratio}.pkl'
    )

    if os.path.exists(model_path):
        text.insert(END, "Loading Elastic Net Regressor...\n")
        text.see(END)
        model = joblib.load(model_path)
    else:
        text.insert(END, "Training Elastic Net Regressor...\n")
        text.see(END)
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, f"Model saved to {model_path}\n\n")
        text.see(END)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # Evaluate
    # -----------------------------
    Calculate_Regression_Metrics(
        "Elastic Net Regressor",
        y_pred,
        y_test
    )

    return model


def train_extra_trees_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
):
    
    model_path = os.path.join(
        MODEL_DIR,
        'extra_trees_regressor.pkl'
    )

    if os.path.exists(model_path):
        text.insert(END, "Loading Extra Trees Regressor...\n")
        text.see(END)
        model = joblib.load(model_path)
    else:
        text.insert(END, "Training Extra Trees Regressor...\n")
        text.see(END)
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, f"Model saved to {model_path}\n\n")
        text.see(END)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # Evaluate
    # -----------------------------
    Calculate_Regression_Metrics(
        "Extra Trees Regressor",
        y_pred,
        y_test
    )

    return model


def train_lasso_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    alpha=1.0,
    max_iter=1000
):

    model_path = os.path.join(
        MODEL_DIR,
        f'lasso_regressor_alpha_{alpha}.pkl'
    )

    if os.path.exists(model_path):
        text.insert(END, "Loading Lasso Regressor...\n")
        text.see(END)
        model = joblib.load(model_path)
    else:
        text.insert(END, "Training Lasso Regressor...\n")
        text.see(END)
        model = Lasso(
            alpha=alpha,
            max_iter=max_iter,
            random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, f"Model saved to {model_path}\n\n")
        text.see(END)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred = model.predict(X_test)

    # Clip predictions to valid range (0–100 for resume score)
    y_pred = np.clip(y_pred, 0, 100)

    # -----------------------------
    # Evaluate
    # -----------------------------
    Calculate_Regression_Metrics(
        "Lasso Regressor",
        y_pred,
        y_test
    )

    return model


def train_stacking_regressor(
    X_train,
    y_train,
    X_test,
    y_test
):

    
    model_path = os.path.join(MODEL_DIR, 'stacking_regressor.pkl')

    if os.path.exists(model_path):
        text.insert(END, "Loading Stacking Regressor...\n")
        text.see(END)
        model = joblib.load(model_path)
    else:
        text.insert(END, "Training Stacking Regressor...\n")
        text.see(END)

        # -----------------------------
        # Base Learners
        # -----------------------------
        base_estimators = [
            ('ridge', Ridge(alpha=1.0)),
            ('theilsen', TheilSenRegressor(random_state=42)),
        ]

        # -----------------------------
        # Meta Learner
        # -----------------------------
        meta_model = Ridge(alpha=0.5)

        # -----------------------------
        # Stacking Model
        # -----------------------------
        model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1,
            passthrough=False
        )

        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, f"Model saved to {model_path}\n\n")
        text.see(END)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # Evaluate
    # -----------------------------
    Calculate_Regression_Metrics(
        "Stacking Regressor",
        y_pred,
        y_test
    )

    return model


def plot_regression_model_performance_tkinter():
    if regression_metrics_df.empty:
        messagebox.showerror("Error", "No regression results available. Train models first.")
        return

    text.delete('1.0', END)
    text.insert(END, "Plotting Regression Model Performance Comparison (Higher = Better)...\n")
    text.see(END)

    # Copy dataframe to avoid modifying original
    df_plot = regression_metrics_df.copy()

    # Convert error metrics so that higher is better
    df_plot['MAE'] = 1 / df_plot['MAE']
    df_plot['RMSE'] = 1 / df_plot['RMSE']

    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 7))

    df_melt = df_plot.melt(
        id_vars='Algorithm',
        value_vars=['MAE', 'RMSE', 'R2'],
        var_name='Metric',
        value_name='Score'
    )

    ax = sns.barplot(
        x='Algorithm',
        y='Score',
        hue='Metric',
        data=df_melt
    )

    plt.xticks(rotation=45, ha='right')
    plt.title("Regression Model Performance Comparison (Higher = Better)")
    plt.ylabel("Score")

    # Dynamic limits
    plt.ylim(0, df_melt['Score'].max() * 1.15)

    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.3f}",
            (p.get_x() + p.get_width() / 2., height),
            ha='center',
            va='bottom',
            fontsize=9,
            xytext=(0, 3),
            textcoords='offset points'
        )

    plt.legend(title="Metric")
    plt.tight_layout()

    save_path = "results/regression_model_performance_comparison.png"
    plt.savefig(save_path)
    plt.show()

    text.insert(END, f"Plot saved as '{save_path}'\n\n")
    text.see(END)



def upload_testdata():
  
    global testdata
    filename = filedialog.askopenfilename(initialdir="Dataset", title="Select Test CSV")
    
    if filename:
        testdata = pd.read_csv(filename)
        text.delete('1.0', END)
        text.insert(END, f"Test dataset loaded from {filename}\n")
        text.insert(END, f"Records: {testdata.shape[0]}, Columns: {testdata.shape[1]}\n\n")
        text.insert(END, str(testdata) + "\n\n")
        text.see(END)
    else:
        text.insert(END, "No file selected.\n\n")
        text.see(END)

def predict_testdata():
    
    global testdata, X_test_new, y_pred, model

    if 'testdata' not in globals() or testdata is None:
        text.insert(END, "Please upload test data first!\n\n")
        text.see(END)
        return

    testdata_copy = testdata.copy()

    degree_encoder = joblib.load(os.path.join(MODEL_DIR, 'degree_encoder.pkl'))
    internship_encoder = joblib.load(os.path.join(MODEL_DIR, 'internship_encoder.pkl'))
    github_encoder = joblib.load(os.path.join(MODEL_DIR, 'github_portfolio_encoder.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'standard_scaler.pkl'))

    testdata_copy['degree'] = degree_encoder.transform(testdata_copy['degree'].astype(str))
    testdata_copy['internship'] = internship_encoder.transform(testdata_copy['internship'].astype(str))
    testdata_copy['github_portfolio'] = github_encoder.transform(testdata_copy['github_portfolio'].astype(str))

    X_test_new = testdata_copy.drop(columns=['resume_score', 'candidate_id'], errors='ignore')

    numeric_cols = X_test_new.select_dtypes(include=['int64', 'float64']).columns
    X_test_new[numeric_cols] = scaler.transform(X_test_new[numeric_cols])

    text.insert(END, "Test data preprocessing completed successfully.\n\n")
    text.see(END)

    model = joblib.load(os.path.join(MODEL_DIR, 'stacking_regressor.pkl'))
    text.insert(END, "Stacking Regressor loaded for prediction.\n\n")
    text.see(END)

    y_pred = model.predict(X_test_new)
    y_pred = np.clip(y_pred, 0, 100)
    testdata['predicted_resume_score'] = y_pred

    text.insert(END, "Predictions added to test data (clipped to 0-100 range).\n\n")
    text.insert(END, str(testdata) + "\n\n")
    text.see(END)


# Set Background Image
def setBackground():
    global bg_photo
    image_path = r"BG_image\bakground2.jpeg" # Update with correct image path
    bg_image = Image.open(image_path)
    bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
    #bg_image = bg_image.resize((900, 600), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = Label(main, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)

setBackground()

def connect_db():
    return pymysql.connect(host='localhost', user='root', password='sandeep@ruthika', database='sparse_db')

# Signup Functionality
def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)"
                cursor.execute(query, (username, password, role))
                conn.commit()
                conn.close()
                messagebox.showinfo("Success", f"{role} Signup Successful!")
                signup_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x300")
    signup_window.title(f"{role} Signup")

    Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)

    Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# Login Functionality
def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "SELECT * FROM users WHERE username=%s AND password=%s AND role=%s"
                cursor.execute(query, (username, password, role))
                result = cursor.fetchone()
                conn.close()
                if result:
                    messagebox.showinfo("Success", f"{role} Login Successful!")
                    login_window.destroy()
                    if role == "Admin":
                        show_admin_buttons()
                    elif role == "User":
                        show_user_buttons()
                else:
                    messagebox.showerror("Error", "Invalid Credentials!")
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)


# Clear buttons function
def clear_buttons():
    for widget in main.place_slaves():
        if isinstance(widget, tkinter.Button):
            widget.destroy()

# -----------------------------
# Admin Button Functions
# -----------------------------
def show_admin_buttons():
    font1 = ('times', 13, 'bold')
    clear_buttons()

    tk.Button(main, text="Upload Dataset", 
              command=uploadDataset, 
              font=font1, bg="Lightblue").place(x=80, y=150)

    tk.Button(main, text="Preprocess Dataset", 
              command=lambda: preprocess_data(dataset), 
              font=font1, bg="Lightblue").place(x=300, y=150)

    tk.Button(main, text="Show EDA Image", 
              command=lambda: perform_eda(dataset), 
              font=font1, bg="Lightblue").place(x=550, y=150)

    tk.Button(main, text="Train-Test-Split", 
              command=lambda: split_train_test(X, y), 
              font=font1, bg="Lightblue").place(x=750, y=150)

    tk.Button(main, text="ElasticNet Regression", 
              command=lambda: train_elasticnet_regressor(X_train, y_train, X_test, y_test),
              font=font1, bg="Lightblue").place(x=100, y=220)

    tk.Button(main, text="Extra Trees Regression", 
              command=lambda: train_extra_trees_regressor(X_train, y_train, X_test, y_test),
              font=font1, bg="Lightblue").place(x=300, y=220)

    tk.Button(main, text="Lasso  Regressor", 
              command=lambda: train_lasso_regressor(X_train, y_train, X_test, y_test),
              font=font1, bg="Lightblue").place(x=550, y=220)

    tk.Button(main, text="Stacking Regressor", 
              command=lambda: train_stacking_regressor(X_train, y_train, X_test, y_test),
              font=font1, bg="Lightblue").place(x=750, y=220)

    tk.Button(main, text="Regression Model Comparison",
                command=plot_regression_model_performance_tkinter,
                font=font1, bg="Lightblue").place(x=1000, y=220)

    tk.Button(main, text="Logout", 
              command=show_login_screen, 
              font=font1, bg="red").place(x=1400, y=600)



# -----------------------------
# User Button Functions
# -----------------------------
def show_user_buttons():
    font1 = ('times', 13, 'bold')
    clear_buttons()

    tk.Button(main, text="Upload Test Data", 
              command=upload_testdata,  
              font=font1, bg="gray").place(x=300, y=200)

    tk.Button(main, text="Predict Resume Scores", 
              command=predict_testdata, 
              font=font1, bg="gray").place(x=550, y=200)

    tk.Button(main, text="Regression Model Comparison",
                command=plot_regression_model_performance_tkinter,
                font=font1, bg="gray").place(x=900, y=200)

    tk.Button(main, text="Exit", 
              command=close, 
              font=font1, bg="gray").place(x=1300, y=200)

    tk.Button(main, text="Logout", 
              command=show_login_screen, 
              font=font1, bg="red").place(x=1400, y=600)


def show_login_screen():
    clear_buttons()
    font1 = ('times', 14, 'bold')

    tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=1, bg='gray').place(x=100, y=100)
    tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=1, bg='gray').place(x=400, y=100)
    tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='Lightblue').place(x=700, y=100)
    tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=1, bg='Lightblue').place(x=1000, y=100)

def close():
    main.destroy()


font = ('times', 16, 'bold')
title = Label(
    main,
    text="Skill-Profile Optimization for Interview Success using Experience and Project Feature Engineering",
    bg='#0A2647', 
    fg='#FFFFFF',
    font=font,
    height=3,
    width=120
)
title.pack(pady=10)


                     
font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=420)
text.config(font=font1) 


# Admin and User Buttons
font1 = ('times', 14, 'bold')


tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=1, bg='gray').place(x=100, y=100)

tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=1, bg='gray').place(x=400, y=100)


admin_button = tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='Lightblue')
admin_button.place(x=700, y=100)

user_button = tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=1, bg='Lightblue')
user_button.place(x=1000, y=100)

main.mainloop()
