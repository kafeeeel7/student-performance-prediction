"""
Generates a synthetic student dataset (data/students.csv).

Why synthetic? It keeps the project self-contained and reproducible.
The data is deliberately a little messy (missing values, duplicate rows)
so the data-cleaning step in the main script has real work to do.

To use a real dataset instead (e.g. the UCI "Student Performance" dataset),
see the README section "Using your own data".
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1000


def build_dataset(n: int = N) -> pd.DataFrame:
    study_hours = np.clip(RNG.normal(14, 6, n), 0, 40)
    attendance = np.clip(RNG.normal(78, 14, n), 30, 100)
    previous_grade = np.clip(RNG.normal(62, 15, n), 15, 100)
    assignments = np.clip(RNG.normal(75, 18, n), 0, 100)
    sleep_hours = np.clip(RNG.normal(6.8, 1.2, n), 3, 10)
    absences = np.clip(RNG.poisson(5, n) + (100 - attendance) // 10, 0, 40).astype(int)

    gender = RNG.choice(["Male", "Female"], n)
    internet = RNG.choice(["Yes", "No"], n, p=[0.82, 0.18])
    parental_support = RNG.choice(["Low", "Medium", "High"], n, p=[0.2, 0.5, 0.3])
    extracurricular = RNG.choice(["Yes", "No"], n, p=[0.45, 0.55])

    support_bonus = pd.Series(parental_support).map({"Low": -3, "Medium": 0, "High": 3}).values
    internet_bonus = np.where(internet == "Yes", 2, -2)

    final_score = (
        0.30 * previous_grade
        + 0.80 * study_hours
        + 0.18 * attendance
        + 0.12 * assignments
        + 1.0 * (sleep_hours - 7)
        - 0.35 * absences
        + support_bonus
        + internet_bonus
        + RNG.normal(0, 4, n)
        + 5
    )
    final_score = np.clip(final_score, 0, 100).round(1)

    df = pd.DataFrame({
        "gender": gender,
        "study_hours_per_week": study_hours.round(1),
        "attendance_pct": attendance.round(1),
        "previous_grade": previous_grade.round(1),
        "assignments_completed_pct": assignments.round(1),
        "sleep_hours": sleep_hours.round(1),
        "absences": absences,
        "internet_access": internet,
        "parental_support": parental_support,
        "extracurricular": extracurricular,
        "final_score": final_score,
    })

    # --- make it realistically messy ---
    for col in ["attendance_pct", "sleep_hours", "parental_support", "study_hours_per_week"]:
        idx = RNG.choice(df.index, size=int(0.04 * n), replace=False)
        df.loc[idx, col] = np.nan
    df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)  # duplicates
    return df


if __name__ == "__main__":
    data = build_dataset()
    data.to_csv("data/students.csv", index=False)
    print(f"Saved data/students.csv  ->  {data.shape[0]} rows, {data.shape[1]} columns")
