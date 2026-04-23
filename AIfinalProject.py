import os
import json
import joblib
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # saves plots instead of opening GUI windows
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import entropy
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score

)

# ==============================
# CONFIG
# ==============================
CSV_PATH = "activity_data_8.csv"
OUTPUT_DIR = "model_results_rf_mlp"

WINDOW_SIZE = 50
STEP_SIZE = 25
MISSING_THRESHOLD = 0.75
TEST_SIZE = 0.20
RANDOM_STATE = 42

EXERCISES = ["pushup", "squat", "lunge", "situp"]
RAW_SENSOR_COLUMNS = ["Ax", "Ay", "Az", "A_mag", "Gx", "Gy", "Gz", "G_mag"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# PREPROCESSING CHECKS
# ==============================
def check_required_columns(df: pd.DataFrame):
    required = set(["label"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def drop_raw_columns_by_completeness(df: pd.DataFrame, threshold: float = 0.75):
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

    if cols_to_drop:
        print(f"Dropping raw columns with >{int(threshold*100)}% missing:", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    return df, cols_to_drop


def get_available_sensor_columns(df: pd.DataFrame):
    return [c for c in RAW_SENSOR_COLUMNS if c in df.columns]


# ==============================
# FEATURE ENGINEERING
# ==============================
def safe_entropy(x, bins=20):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist + 1e-12
    return float(entropy(hist))


def safe_stat(func, arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    try:
        return float(func(arr))
    except Exception:
        return np.nan


def extract_features(window: pd.DataFrame, feature_cols: list[str]):
    feats = []
    names = []

    for col in feature_cols:
        data = window[col].values

        feature_values = {
            f"{col}_mean": safe_stat(np.mean, data),
            f"{col}_std": safe_stat(np.std, data),
            f"{col}_min": safe_stat(np.min, data),
            f"{col}_max": safe_stat(np.max, data),
            f"{col}_median": safe_stat(np.median, data),
            f"{col}_q25": safe_stat(lambda x: np.percentile(x, 25), data),
            f"{col}_q75": safe_stat(lambda x: np.percentile(x, 75), data),
            f"{col}_range": safe_stat(lambda x: np.max(x) - np.min(x), data),
            f"{col}_abs_mean": safe_stat(lambda x: np.mean(np.abs(x)), data),
            f"{col}_energy": safe_stat(lambda x: np.sum(x**2), data),
            f"{col}_entropy": safe_entropy(data),
        }

        for k, v in feature_values.items():
            names.append(k)
            feats.append(v)

    return feats, names


# ==============================
# WINDOWING
# ==============================
def create_windows(df: pd.DataFrame, feature_cols: list[str]):
    X = []
    y = []
    feature_names = None

    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = df.iloc[start:end]

        if window["label"].nunique() == 1:
            feats, names = extract_features(window, feature_cols)
            if feature_names is None:
                feature_names = names

            X.append(feats)
            y.append(window["label"].iloc[0])

    if feature_names is None:
        feature_names = []

    return np.array(X, dtype=float), np.array(y), feature_names


def drop_engineered_features_by_completeness(X, feature_names, threshold=0.75):
    if X.size == 0:
        return X, feature_names, []

    missing_ratio = np.mean(np.isnan(X), axis=0)
    keep_idx = np.where(missing_ratio <= threshold)[0]
    dropped_idx = np.where(missing_ratio > threshold)[0]

    dropped_features = [feature_names[i] for i in dropped_idx]
    kept_features = [feature_names[i] for i in keep_idx]

    X = X[:, keep_idx]

    if dropped_features:
        print(f"Dropping engineered features with >{int(threshold*100)}% missing:")
        print(dropped_features)

    return X, kept_features, dropped_features


def fill_remaining_missing(X_train, X_test):
    train_medians = np.nanmedian(X_train, axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)

    inds_train = np.where(np.isnan(X_train))
    X_train[inds_train] = np.take(train_medians, inds_train[1])

    inds_test = np.where(np.isnan(X_test))
    X_test[inds_test] = np.take(train_medians, inds_test[1])

    return X_train, X_test


# ==============================
# PLOTS
# ==============================
def plot_feature_correlation(X_train, feature_names, exercise):
    if len(feature_names) == 0:
        return

    df_feat = pd.DataFrame(X_train, columns=feature_names)
    corr = df_feat.corr(numeric_only=True)

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(f"{exercise.title()} - Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{exercise}_feature_corr.png"), dpi=250)
    plt.close()


def plot_outcome_correlation(X_train, y_train, feature_names, exercise):
    if len(feature_names) == 0:
        return

    df_feat = pd.DataFrame(X_train, columns=feature_names)
    df_feat["Outcome"] = y_train

    corr_with_outcome = df_feat.corr(numeric_only=True)["Outcome"].drop("Outcome")
    corr_sorted = corr_with_outcome.reindex(
        corr_with_outcome.abs().sort_values(ascending=False).index
    )

    plt.figure(figsize=(16, 6))
    sns.barplot(x=corr_sorted.index, y=corr_sorted.values)
    plt.xticks(rotation=90)
    plt.title(f"{exercise.title()} - Feature Correlation with Good(1)/Bad(0)")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{exercise}_outcome_corr.png"), dpi=250)
    plt.close()

    corr_sorted.to_csv(os.path.join(OUTPUT_DIR, f"{exercise}_outcome_corr.csv"))


def plot_confusion(cm, exercise, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bad", "Good"],
                yticklabels=["Bad", "Good"])
    plt.title(f"{exercise.title()} - {model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{exercise}_{model_name}_confusion.png"), dpi=250)
    plt.close()


def plot_roc(y_true, y_prob, exercise, model_name):
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{exercise.title()} - {model_name} ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{exercise}_{model_name}_roc.png"), dpi=250)
    plt.close()

    return auc


# ==============================
# MODEL BUILDERS
# ==============================
def build_random_forest_search():
    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    return rf, param_grid


def build_mlp_search():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=1500, random_state=RANDOM_STATE))
    ])

    param_grid = {
        "mlp__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__learning_rate_init": [0.001, 0.01]
    }

    return pipeline, param_grid


# ==============================
# TRAIN / EVAL
# ==============================
def tune_and_train(model, param_grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True
    )

    search.fit(X_train, y_train)
    return search


def evaluate_model(best_model, X_test, y_test, exercise, model_name):
    y_pred = best_model.predict(X_test)

    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["bad", "good"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    plot_confusion(cm, exercise, model_name)
    plot_roc(y_test, y_prob, exercise, model_name)


    print(f"\n{model_name.upper()} Detailed Metrics:")

    print(f"Precision:   {precision:.3f}")

    print(f"Recall:      {recall:.3f}")

    print(f"F1 Score:    {f1:.3f}")

    print(f"Specificity: {specificity:.3f}")

    return {

        "accuracy": acc,

        "roc_auc": auc,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "specificity": specificity,

        "classification_report": report,

        "confusion_matrix": cm.tolist()
        }
 


# ==============================
# PER EXERCISE PROCESSING
# ==============================
def process_exercise(df, exercise):
    print(f"\n{'='*60}")
    print(f"EXERCISE: {exercise.upper()}")
    print(f"{'='*60}")

    df_ex = df[df["label"].str.contains(exercise, case=False, na=False)].copy()

    if df_ex.empty:
        print(f"No data found for {exercise}")
        return []

    df_ex["label"] = df_ex["label"].apply(lambda x: 1 if "good" in x.lower() else 0)

    feature_cols = get_available_sensor_columns(df_ex)
    if len(feature_cols) == 0:
        print(f"No usable sensor columns found for {exercise}")
        return []

    print("Using raw sensor columns:", feature_cols)

    X, y, feature_names = create_windows(df_ex, feature_cols)

    if len(X) == 0:
        print(f"No valid windows for {exercise}")
        return []

    print("Total windows:", len(X))
    print("Initial engineered feature dimension:", X.shape[1])
    print("Class counts:", pd.Series(y).value_counts().to_dict())

    X, feature_names, dropped_features = drop_engineered_features_by_completeness(
        X, feature_names, threshold=MISSING_THRESHOLD
    )

    print("Final engineered feature dimension:", X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train, X_test = fill_remaining_missing(X_train, X_test)

    plot_feature_correlation(X_train, feature_names, exercise)
    plot_outcome_correlation(X_train, y_train, feature_names, exercise)

    results = []

    print("\nTraining Random Forest...")
    rf_model, rf_grid = build_random_forest_search()
    rf_search = tune_and_train(rf_model, rf_grid, X_train, y_train)

    print("Random Forest best params:", rf_search.best_params_)
    print("Random Forest best CV ROC AUC:", round(rf_search.best_score_, 3))

    rf_eval = evaluate_model(rf_search.best_estimator_, X_test, y_test, exercise, "random_forest")
    print("Random Forest accuracy:", round(rf_eval["accuracy"], 3))
    print("Random Forest ROC AUC:", round(rf_eval["roc_auc"], 3))

    rf_model_path = os.path.join(OUTPUT_DIR, f"{exercise}_random_forest_model.pkl")
    joblib.dump(rf_search.best_estimator_, rf_model_path)

    results.append({
    "exercise": exercise,
    "model": "random_forest",
    "accuracy": rf_eval["accuracy"],
    "roc_auc": rf_eval["roc_auc"],
    "precision": rf_eval["precision"],
    "recall": rf_eval["recall"],
    "f1_score": rf_eval["f1_score"],
    "specificity": rf_eval["specificity"],
    "best_cv_roc_auc": rf_search.best_score_,
    "best_params": rf_search.best_params_,
    "model_path": rf_model_path,
    "dropped_engineered_features": dropped_features,
    })

    print("\nTraining Neural Network (MLP)...")
    mlp_model, mlp_grid = build_mlp_search()
    mlp_search = tune_and_train(mlp_model, mlp_grid, X_train, y_train)

    print("MLP best params:", mlp_search.best_params_)
    print("MLP best CV ROC AUC:", round(mlp_search.best_score_, 3))

    mlp_eval = evaluate_model(mlp_search.best_estimator_, X_test, y_test, exercise, "mlp")
    print("MLP accuracy:", round(mlp_eval["accuracy"], 3))
    print("MLP ROC AUC:", round(mlp_eval["roc_auc"], 3))

    mlp_model_path = os.path.join(OUTPUT_DIR, f"{exercise}_mlp_model.pkl")
    joblib.dump(mlp_search.best_estimator_, mlp_model_path)

    results.append({
    "exercise": exercise,
    "model": "mlp",
    "accuracy": mlp_eval["accuracy"],
    "roc_auc": mlp_eval["roc_auc"],
    "precision": mlp_eval["precision"],
    "recall": mlp_eval["recall"],
    "f1_score": mlp_eval["f1_score"],
    "specificity": mlp_eval["specificity"],
    "best_cv_roc_auc": mlp_search.best_score_,
    "best_params": mlp_search.best_params_,
    "model_path": mlp_model_path,
    "dropped_engineered_features": dropped_features})

    return results


# ==============================
# MAIN
# ==============================
def main():
    df = pd.read_csv(CSV_PATH)

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    check_required_columns(df)

    df, dropped_raw_cols = drop_raw_columns_by_completeness(df, threshold=MISSING_THRESHOLD)

    all_results = []

    for exercise in EXERCISES:
        results = process_exercise(df, exercise)
        all_results.extend(results)

    summary_df = pd.DataFrame(all_results)
    summary_csv = os.path.join(OUTPUT_DIR, "summary_results.csv")
    summary_json = os.path.join(OUTPUT_DIR, "summary_results.json")

    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nFinal Summary")
    print(summary_df[["exercise", "model", "accuracy", "roc_auc", "best_cv_roc_auc"]])

    best_models = (
        summary_df.sort_values(["exercise", "roc_auc"], ascending=[True, False])
        .groupby("exercise")
        .first()
        .reset_index()
    )

    print("\nBest model per exercise (by ROC AUC):")
    print(best_models[["exercise", "model", "accuracy", "roc_auc", "model_path"]])

    best_models.to_csv(os.path.join(OUTPUT_DIR, "best_models.csv"), index=False)


if __name__ == "__main__":
    main()