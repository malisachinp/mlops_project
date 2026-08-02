"""Train, tune, evaluate and save the best tourism purchase model."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("models")
TARGET = "ProdTaken"

def make_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

def main():
    train = pd.read_csv(ARTIFACT_DIR / "train.csv")
    test = pd.read_csv(ARTIFACT_DIR / "test.csv")
    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]

    pre = make_preprocessor(X_train)
    experiments = [
        ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced"),
         {"model__C": [0.1, 1.0, 10.0]}),
        ("RandomForest", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
         {"model__n_estimators": [150], "model__max_depth": [None, 12], "model__min_samples_split": [2, 5]}),
        ("GradientBoosting", GradientBoostingClassifier(random_state=42),
         {"model__n_estimators": [100], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]}),
        ("AdaBoost", AdaBoostClassifier(random_state=42),
         {"model__n_estimators": [100], "model__learning_rate": [0.5, 1.0]})
    ]

    results = []
    best = None
    for name, estimator, grid in experiments:
        pipe = Pipeline([("preprocessor", pre), ("model", estimator)])
        search = GridSearchCV(pipe, grid, cv=3, scoring="f1", n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        prob = search.predict_proba(X_test)[:, 1]
        row = {
            "model": name,
            "best_params": json.dumps(search.best_params_),
            "cv_f1": search.best_score_,
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, prob)
        }
        results.append(row)
        if best is None or row["test_f1"] > best[0]:
            best = (row["test_f1"], search.best_estimator_, row, pred, prob)

    result_df = pd.DataFrame(results).sort_values("test_f1", ascending=False)
    result_df.to_csv(ARTIFACT_DIR / "experiment_results.csv", index=False)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best[1], MODEL_DIR / "best_model.joblib")
    metrics = {k: v for k, v in best[2].items() if k != "best_params"}
    metrics["best_params"] = json.loads(best[2]["best_params"])
    (ARTIFACT_DIR / "best_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(result_df.to_string(index=False))
    print("\nBEST MODEL:", best[2]["model"])
    print(classification_report(y_test, best[3], digits=4))
    print("Saved models/best_model.joblib")

if __name__ == "__main__":
    main()
