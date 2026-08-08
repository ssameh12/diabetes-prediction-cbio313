"""Train, compare and tune the models, then write every artefact the report and README quote.

Outputs
-------
models/best_model.joblib   the tuned winning pipeline
results/metrics.json       every number quoted elsewhere in the repo
results/cv_comparison.csv  cross-validated comparison table
results/test_results.csv   held-out test scores
figures/*.png              EDA and evaluation figures

Run from anywhere:  python src/train.py
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import FEATURES, TARGET, ZERO_AS_MISSING, clean, load_raw, missing_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RANDOM_STATE = 42

for d in (FIGURES, RESULTS, MODELS):
    d.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def save(fig, name):
    path = FIGURES / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  figure ->", path.name)


def make_pipeline(model):
    """Imputer and scaler live inside the pipeline, so their parameters are fitted on
    training folds only. Fitting them on the full dataset would leak test information."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


MODEL_ZOO = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=11),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

PARAM_GRIDS = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10], "model__class_weight": [None, "balanced"]},
    "Random Forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 3, 5],
        "model__class_weight": [None, "balanced"],
    },
    "SVM (RBF)": {
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", 0.01, 0.1],
        "model__class_weight": [None, "balanced"],
    },
}


def eda(raw, cleaned):
    print("EDA")
    counts = raw[TARGET].value_counts().sort_index()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(["No diabetes (0)", "Diabetes (1)"], counts.values, color=["#4c72b0", "#c44e52"])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v + 6, f"{v}  ({v / len(raw):.1%})", ha="center")
    ax[0].set_ylabel("patients")
    ax[0].set_title("Class balance")
    ax[0].set_ylim(0, counts.max() * 1.18)

    rep = missing_report(raw)
    ax[1].barh(rep.index[::-1], rep["percent"][::-1], color="#dd8452")
    for i, v in enumerate(rep["percent"][::-1]):
        ax[1].text(v + 0.6, i, f"{v}%", va="center")
    ax[1].set_xlabel("% of rows recorded as 0")
    ax[1].set_title("Hidden missing values")
    ax[1].set_xlim(0, rep["percent"].max() * 1.25)
    fig.suptitle("Pima Indians Diabetes - dataset overview")
    save(fig, "01_overview.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for axis, col in zip(axes.ravel(), FEATURES):
        for label, colour in ((0, "#4c72b0"), (1, "#c44e52")):
            sns.kdeplot(
                cleaned.loc[cleaned[TARGET] == label, col].dropna(),
                ax=axis, fill=True, alpha=0.35, color=colour,
                label="diabetes" if label else "no diabetes",
            )
        axis.set_title(col)
        axis.set_xlabel("")
    axes.ravel()[0].legend()
    fig.suptitle("Feature distributions by outcome (missing values excluded)", y=1.02)
    fig.tight_layout()
    save(fig, "02_distributions.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cleaned[FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation matrix")
    save(fig, "03_correlation.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for axis, col in zip(axes.ravel(), FEATURES):
        sns.boxplot(data=cleaned, x=TARGET, y=col, ax=axis, hue=TARGET, legend=False,
                    palette=["#4c72b0", "#c44e52"])
        axis.set_title(col)
        axis.set_xlabel("")
    fig.suptitle("Outliers and spread by outcome", y=1.02)
    fig.tight_layout()
    save(fig, "04_boxplots.png")

    return {
        "n_samples": int(len(raw)),
        "n_features": len(FEATURES),
        "class_counts": {"no_diabetes": int(counts[0]), "diabetes": int(counts[1])},
        "positive_rate": round(float(counts[1] / len(raw)), 4),
        "missing": {k: {"zeros": int(v["zeros"]), "percent": float(v["percent"])}
                    for k, v in rep.to_dict("index").items()},
        "correlation_with_outcome": cleaned[FEATURES].corrwith(cleaned[TARGET]).round(3).to_dict(),
    }


def cross_validate_models(X_train, y_train):
    print("Cross-validation (stratified 5-fold, training set only)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, model in MODEL_ZOO.items():
        scores = cross_validate(make_pipeline(model), X_train, y_train, cv=cv, scoring=scoring)
        rows.append(
            {"model": name,
             **{m: round(float(scores[f"test_{m}"].mean()), 4) for m in scoring},
             "roc_auc_std": round(float(scores["test_roc_auc"].std()), 4)}
        )
        print(f"  {name:<22} ROC-AUC {rows[-1]['roc_auc']:.4f}  recall {rows[-1]['recall']:.4f}")
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    table.to_csv(RESULTS / "cv_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    melted = table.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
                        var_name="metric", value_name="score")
    sns.barplot(data=melted, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Cross-validated comparison of the five algorithms")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "05_model_comparison.png")
    return table


def evaluate_on_test(fitted, X_test, y_test):
    print("Held-out test evaluation")
    rows, roc_data = [], {}
    for name, pipe in fitted.items():
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
        })
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))
        print(f"  {name:<22} acc {rows[-1]['accuracy']:.4f}  recall {rows[-1]['recall']:.4f}")

    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    table.to_csv(RESULTS / "test_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, a) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "06_roc_curves.png")

    fig, axes = plt.subplots(1, len(fitted), figsize=(4 * len(fitted), 3.6))
    for axis, (name, pipe) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_test, pipe.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis,
                    xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
        axis.set_title(name, fontsize=10)
    fig.suptitle("Confusion matrices (test set)")
    fig.tight_layout()
    save(fig, "07_confusion_matrices.png")
    return table


def tune(X_train, y_train, X_test, y_test):
    """Grids are scored on ROC-AUC, not recall.

    Recall is what matters clinically, but optimising a grid directly on recall rewards
    models that simply predict "diabetic" more often, which collapses precision. ROC-AUC
    ranks models independently of the decision threshold; the threshold is then chosen
    separately in choose_threshold(), on training data only.
    """
    print("Hyperparameter tuning (GridSearchCV, scored on ROC-AUC)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuned, summary = {}, []
    for name, grid in PARAM_GRIDS.items():
        search = GridSearchCV(make_pipeline(MODEL_ZOO[name]), grid, cv=cv,
                              scoring="roc_auc", n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        pred = search.best_estimator_.predict(X_test)
        proba = search.best_estimator_.predict_proba(X_test)[:, 1]
        summary.append({
            "model": name,
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "cv_roc_auc": round(float(search.best_score_), 4),
            "test_accuracy": round(accuracy_score(y_test, pred), 4),
            "test_precision": round(precision_score(y_test, pred), 4),
            "test_recall": round(recall_score(y_test, pred), 4),
            "test_f1": round(f1_score(y_test, pred), 4),
            "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
        })
        print(f"  {name:<22} best {summary[-1]['best_params']}")
        print(f"  {'':<22} CV ROC-AUC {summary[-1]['cv_roc_auc']:.4f}  test ROC-AUC {summary[-1]['test_roc_auc']:.4f}")
    return tuned, summary


def choose_threshold(pipe, X_train, y_train, target_recall=0.80):
    """Pick the decision threshold on cross-validated TRAINING predictions.

    The default 0.5 cut-off optimises accuracy, which is the wrong target for screening.
    We take the highest threshold that still reaches `target_recall` out-of-fold, so the
    test set plays no part in choosing it.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    grid = np.linspace(0.05, 0.95, 181)
    feasible = [(t, recall_score(y_train, (oof >= t).astype(int))) for t in grid]
    ok = [t for t, r in feasible if r >= target_recall]
    threshold = float(max(ok)) if ok else 0.5
    print(f"  out-of-fold threshold for recall >= {target_recall:.0%}: {threshold:.3f}")
    return threshold, oof


def transform_only(pipe, X):
    """Apply the pipeline's impute+scale steps, keeping column names for the plots."""
    return pd.DataFrame(
        pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(X)),
        columns=FEATURES,
    )


def explain(best_pipe, X_train, X_test, y_test, best_name):
    """Model-agnostic importance plus SHAP. Never allowed to break the run."""
    print("Explainability")
    out = {}
    model = best_pipe.named_steps["model"]
    train_t = transform_only(best_pipe, X_train)
    test_t = transform_only(best_pipe, X_test)

    # Permutation importance works for every estimator, so figure 08 always exists.
    perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=20,
                                  random_state=RANDOM_STATE, scoring="roc_auc")
    imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    out["permutation_importance"] = imp.round(4).to_dict()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color="#4c72b0")
    ax.set_xlabel("drop in ROC-AUC when the feature is shuffled")
    ax.set_title(f"Permutation importance - {best_name}")
    save(fig, "08_feature_importance.png")
    print("  permutation top three:", list(imp.head(3).index))

    try:
        import shap
    except ImportError:
        print("  shap not installed - skipping SHAP plots")
        return out

    try:
        if hasattr(model, "feature_importances_"):
            values = shap.TreeExplainer(model).shap_values(test_t)
            if isinstance(values, list):
                values = values[1]
            elif values.ndim == 3:
                values = values[:, :, 1]
            frame = test_t
        elif hasattr(model, "coef_"):
            values = shap.LinearExplainer(model, train_t).shap_values(test_t)
            frame = test_t
        else:
            # KernelExplainer samples internally; seed it so the run is reproducible.
            np.random.seed(RANDOM_STATE)
            background = shap.kmeans(train_t, 25)
            explainer = shap.KernelExplainer(lambda d: model.predict_proba(d)[:, 1], background)
            frame = test_t.iloc[:100]
            values = explainer.shap_values(frame, nsamples=100, silent=True)

        shap.summary_plot(values, frame, show=False, plot_size=(9, 5))
        save(plt.gcf(), "09_shap_summary.png")

        shap.summary_plot(values, frame, plot_type="bar", show=False, plot_size=(9, 4.5))
        save(plt.gcf(), "10_shap_importance.png")

        mean_abs = pd.Series(np.abs(values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
        out["shap_mean_abs"] = mean_abs.round(4).to_dict()
        print("  SHAP top three:", list(mean_abs.head(3).index))
    except Exception as exc:  # noqa: BLE001 - explainability must never break the run
        print("  SHAP failed:", exc)
    return out


def main():
    raw = load_raw()
    cleaned = clean(raw)
    print(f"Loaded {len(raw)} rows, {len(FEATURES)} features\n")

    eda_stats = eda(raw, cleaned)

    X, y = cleaned[FEATURES], cleaned[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"\nTrain {X_train.shape[0]} rows / test {X_test.shape[0]} rows (stratified)\n")

    cv_table = cross_validate_models(X_train, y_train)

    fitted = {name: make_pipeline(model).fit(X_train, y_train) for name, model in MODEL_ZOO.items()}
    test_table = evaluate_on_test(fitted, X_test, y_test)

    tuned, tuning_summary = tune(X_train, y_train, X_test, y_test)

    # Selected on cross-validated ROC-AUC from the TRAINING set only. Picking the winner by
    # test score would leak the test set into model selection and inflate the reported result.
    best_row = max(tuning_summary, key=lambda r: r["cv_roc_auc"])
    best_name = best_row["model"]
    best_pipe = tuned[best_name]
    print(f"\nSelected model: {best_name} (CV ROC-AUC {best_row['cv_roc_auc']:.4f})")

    print("Threshold selection")
    threshold, _ = choose_threshold(best_pipe, X_train, y_train, target_recall=0.80)
    proba_test = best_pipe.predict_proba(X_test)[:, 1]
    pred_default = (proba_test >= 0.5).astype(int)
    pred_tuned_t = (proba_test >= threshold).astype(int)

    def scores(pred):
        return {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }

    at_default, at_threshold = scores(pred_default), scores(pred_tuned_t)
    print(f"  threshold 0.50 -> recall {at_default['recall']:.4f}  precision {at_default['precision']:.4f}")
    print(f"  threshold {threshold:.2f} -> recall {at_threshold['recall']:.4f}  precision {at_threshold['precision']:.4f}")

    baseline_pred = fitted[best_name].predict(X_test)
    explain_stats = explain(best_pipe, X_train, X_test, y_test, best_name)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, pipe in (("before tuning", fitted[best_name]), ("after tuning", best_pipe)):
        proba = pipe.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec, prec, label=f"{label} (area = {auc(rec, prec):.3f})")
    ax.scatter([at_default["recall"]], [at_default["precision"]], s=70, marker="o",
               color="k", zorder=5, label="threshold 0.50")
    ax.scatter([at_threshold["recall"]], [at_threshold["precision"]], s=90, marker="*",
               color="#c44e52", zorder=5, label=f"threshold {threshold:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall trade-off - {best_name}")
    ax.legend()
    save(fig, "11_precision_recall.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for axis, (title, res) in zip(axes, ((f"Threshold 0.50", at_default),
                                         (f"Threshold {threshold:.2f}", at_threshold))):
        sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    ax=axis, xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
        axis.set_title(f"{title}\nrecall {res['recall']:.2f} / precision {res['precision']:.2f}",
                       fontsize=10)
    fig.suptitle(f"Effect of the decision threshold - {best_name}")
    fig.tight_layout()
    save(fig, "12_threshold_effect.png")

    joblib.dump({"pipeline": best_pipe, "threshold": threshold, "features": FEATURES,
                 "model_name": best_name},
                MODELS / "best_model.joblib")
    print("  model  -> models/best_model.joblib")

    metrics = {
        "random_state": RANDOM_STATE,
        "split": {"train": int(len(X_train)), "test": int(len(X_test)), "test_size": 0.2,
                  "stratified": True},
        "dataset": eda_stats,
        "cross_validation": cv_table.to_dict("records"),
        "test_results": test_table.to_dict("records"),
        "tuning": tuning_summary,
        "best_model": {
            "name": best_name,
            "params": best_row["best_params"],
            "cv_roc_auc": best_row["cv_roc_auc"],
            "selected_on": "cross-validated ROC-AUC, training set only",
            "test_roc_auc": best_row["test_roc_auc"],
            "threshold": round(threshold, 3),
            "at_default_threshold": at_default,
            "at_tuned_threshold": at_threshold,
            "classification_report": classification_report(
                y_test, pred_tuned_t,
                target_names=["No diabetes", "Diabetes"], output_dict=True, zero_division=0),
            "baseline_before_tuning": {
                "accuracy": round(accuracy_score(y_test, baseline_pred), 4),
                "recall": round(recall_score(y_test, baseline_pred), 4),
                "f1": round(f1_score(y_test, baseline_pred), 4),
            },
        },
        "explainability": explain_stats,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("  results -> results/metrics.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
