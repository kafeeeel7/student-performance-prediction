"""
Student Performance Prediction
------------------------------
Pipeline:
    1. Load data
    2. Clean data (duplicates, missing values)
    3. Exploratory data analysis (saved as images/)
    4. Train & compare ML models
         - Classification: will the student PASS (final_score >= 50)?
         - Regression:     predict the exact final_score
    5. Evaluate with accuracy / precision / recall / F1 / MAE / RMSE / R2

Run:
    python generate_data.py
    python student_performance.py
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    f1_score, mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_PATH = Path("data/students.csv")
IMG_DIR = Path("images")
IMG_DIR.mkdir(exist_ok=True)
PASS_MARK = 50
RANDOM_STATE = 42


# ----------------------------------------------------------------- 1. LOAD
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------- 2. CLEAN
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- DATA CLEANING ---")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows")

    print("Missing values per column (before imputation):")
    print(df.isna().sum()[df.isna().sum() > 0].to_string())

    # Impute here (not only inside the model pipeline) so EDA plots are clean too.
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    assert df.isna().sum().sum() == 0
    df["result"] = np.where(df["final_score"] >= PASS_MARK, "Pass", "Fail")
    print(f"Clean dataset: {df.shape[0]} rows | Pass rate: {(df['result'] == 'Pass').mean():.1%}")
    return df


# --------------------------------------------------------------------- 3. EDA
def run_eda(df: pd.DataFrame) -> None:
    print("\n--- EXPLORATORY DATA ANALYSIS ---")
    print(df.describe().round(2).T[["mean", "std", "min", "max"]].to_string())

    # Score distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(df["final_score"], bins=30, kde=True, color="#4C72B0")
    plt.axvline(PASS_MARK, color="red", linestyle="--", label=f"Pass mark ({PASS_MARK})")
    plt.title("Distribution of Final Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "01_score_distribution.png", dpi=150)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "02_correlation_heatmap.png", dpi=150)
    plt.close()

    # Study hours & attendance vs score
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.scatterplot(data=df, x="study_hours_per_week", y="final_score",
                    hue="result", alpha=0.6, ax=ax[0])
    ax[0].set_title("Study Hours vs Final Score")
    sns.scatterplot(data=df, x="attendance_pct", y="final_score",
                    hue="result", alpha=0.6, ax=ax[1])
    ax[1].set_title("Attendance vs Final Score")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "03_study_attendance_vs_score.png", dpi=150)
    plt.close()

    # Categorical effects
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col in zip(ax, ["parental_support", "internet_access", "extracurricular"]):
        sns.boxplot(data=df, x=col, y="final_score", ax=a)
        a.set_title(f"Final Score by {col.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "04_categorical_effects.png", dpi=150)
    plt.close()
    print("Saved EDA charts to images/")


# ------------------------------------------------------------ preprocessing
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


# ------------------------------------------------------- 4a. CLASSIFICATION
def run_classification(df: pd.DataFrame) -> None:
    print("\n--- CLASSIFICATION: Pass / Fail ---")
    X = df.drop(columns=["final_score", "result"])
    y = (df["result"] == "Pass").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    rows, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("prep", make_preprocessor(X)), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "5-Fold CV Acc": cv.mean(),
        })
        fitted[name] = pipe

    results = pd.DataFrame(rows).set_index("Model").round(3)
    print(results.to_string())
    results.to_csv("classification_results.csv")

    best_name = results["F1"].idxmax()
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, best.predict(X_test), target_names=["Fail", "Pass"]))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        best, X_test, y_test, display_labels=["Fail", "Pass"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix ({best_name})")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "05_confusion_matrix.png", dpi=150)
    plt.close()

    # Model comparison
    results[["Accuracy", "F1"]].plot(kind="bar", figsize=(7, 4), rot=0, colormap="Set2")
    plt.ylim(0.5, 1.0)
    plt.title("Classification Model Comparison")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "06_model_comparison.png", dpi=150)
    plt.close()

    # Feature importance from the Random Forest
    rf = fitted["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    names = [n.split("__")[1] for n in names]
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
    imp.sort_values().tail(10).plot(kind="barh", figsize=(7, 4.5), color="#55A868")
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "07_feature_importance.png", dpi=150)
    plt.close()
    print("\nTop factors influencing pass/fail:")
    print(imp.sort_values(ascending=False).head(5).round(3).to_string())


# ----------------------------------------------------------- 4b. REGRESSION
def run_regression(df: pd.DataFrame) -> None:
    print("\n--- REGRESSION: predict final score ---")
    X = df.drop(columns=["final_score", "result"])
    y = df["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }
    rows, preds = [], {}
    for name, model in models.items():
        pipe = Pipeline([("prep", make_preprocessor(X)), ("model", model)])
        pipe.fit(X_train, y_train)
        p = pipe.predict(X_test)
        preds[name] = p
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, p),
            "RMSE": np.sqrt(mean_squared_error(y_test, p)),
            "R2": r2_score(y_test, p),
        })
    results = pd.DataFrame(rows).set_index("Model").round(3)
    print(results.to_string())
    results.to_csv("regression_results.csv")

    best = results["R2"].idxmax()
    plt.figure(figsize=(5.5, 5))
    plt.scatter(y_test, preds[best], alpha=0.5, color="#C44E52")
    lims = [y_test.min(), y_test.max()]
    plt.plot(lims, lims, "k--", label="Perfect prediction")
    plt.xlabel("Actual score")
    plt.ylabel("Predicted score")
    plt.title(f"Actual vs Predicted ({best})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "08_actual_vs_predicted.png", dpi=150)
    plt.close()


def main() -> None:
    df = load_data(DATA_PATH)
    df = clean_data(df)
    run_eda(df)
    run_classification(df)
    run_regression(df)
    print("\nDone. Charts saved in images/, metrics saved as CSV files.")


if __name__ == "__main__":
    main()
