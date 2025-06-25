#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Data Science Pipeline for WDBC with Hyperparameter Tuning,
Feature Importance Analysis, Robustness Testing, and Additional Models
Author: Damian Baniel
Date: 2025-06-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


# --- Data Loading & Preprocessing ---

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["ID number"])
    return df


def remove_high_corr(df: pd.DataFrame, thresh: float = 0.85) -> pd.DataFrame:
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > thresh)]
    return df.drop(columns=to_drop)


def remove_outliers(df: pd.DataFrame, features: list, factor: float = 1.5) -> pd.DataFrame:
    df_clean = df.copy()
    for col in features:
        q1, q3 = df_clean[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - factor * iqr, q3 + factor * iqr
        df_clean = df_clean[(df_clean[col] >= low) & (df_clean[col] <= high)]
    return df_clean


# --- Exploration & Composite Biomarkers (PCA) ---

def plot_histograms(df: pd.DataFrame, title: str):
    df.hist(bins=30, figsize=(20, 15))
    plt.suptitle(title)
    plt.show()


def compute_pca(df: pd.DataFrame, n_components: int = 3):
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(df)
    print("PCA Explained Variance Ratios:", pca.explained_variance_ratio_)
    for i, var in enumerate(pca.explained_variance_ratio_, start=1):
        print(f" Component {i}: {var:.2%}")
    return comps, pca


# --- Hyperparameter Tuning ---

def tune_models(X_train, y_train):
    param_grids = {
        'LogisticRegression': {'clf__C': [0.01, 0.1, 1, 10]},
        'KNeighbors': {'clf__n_neighbors': [3, 5, 7, 9]},
        'RandomForest': {'clf__n_estimators': [50, 100], 'clf__max_depth': [None, 5, 10]},
        'GradientBoosting': {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.01, 0.1]},
        'SVC': {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf']}
    }

    pipelines = {
        'LogisticRegression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(solver='liblinear'))]),
        'KNeighbors': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),
        'GaussianNB': Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())]),
        'RandomForest': Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state=42))]),
        'GradientBoosting': Pipeline([('scaler', StandardScaler()), ('clf', GradientBoostingClassifier(random_state=42))]),
        'SVC': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))])
    }
    best_models = {}
    for name, pipe in pipelines.items():
        if name in param_grids:
            gs = GridSearchCV(pipe, param_grids[name], cv=5, scoring='roc_auc')
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            print(f"Best params for {name}: {gs.best_params_}")
        else:
            pipe.fit(X_train, y_train)
            best = pipe
        best_models[name] = best
    return best_models


# --- Feature Importance ---

def display_feature_importance(model, feature_names):
    if hasattr(model.named_steps['clf'], 'feature_importances_'):
        imp = model.named_steps['clf'].feature_importances_
    elif hasattr(model.named_steps['clf'], 'coef_'):
        imp = np.abs(model.named_steps['clf'].coef_).ravel()
    else:
        return
    idx = np.argsort(imp)[::-1]
    print("Feature importance for", type(model.named_steps['clf']).__name__)
    for i in idx[:10]:
        print(f" {feature_names[i]}: {imp[i]:.4f}")


# --- Robustness Testing ---

def test_noise_robustness(models, X_test, y_test, noise_levels=[0.1, 0.2]):
    results = {}
    for nl in noise_levels:
        X_noisy = X_test + np.random.normal(scale=nl * X_test.std(), size=X_test.shape)
        scores = {name: accuracy_score(y_test, m.predict(X_noisy))
                  for name, m in models.items()}
        results[nl] = scores
        print(f"Noise level {nl}:", scores)
    return results


def test_outlier_robustness(models, X_test, y_test, outlier_fraction=0.05):
    X_out = X_test.copy()
    n_out = int(outlier_fraction * X_out.size)
    idx = np.unravel_index(
        np.random.choice(X_out.size, n_out, replace=False), X_out.shape)
    X_out.values[idx] = X_out.values[idx] * 10  # exaggerate
    scores = {name: accuracy_score(y_test, m.predict(X_out))
              for name, m in models.items()}
    print(f"Outlier robustness ({outlier_fraction*100}%):", scores)
    return scores


# --- Main Flow ---

def main():
    df = load_data("wdbc.csv")
    df['Diagnosis'] = LabelEncoder().fit_transform(df['Diagnosis'])

    # EDA
    plot_histograms(df.drop(columns=['Diagnosis']), "Original Feature Distributions")

    # Feature selection & cleaning
    df = remove_high_corr(df.drop(columns=['Diagnosis']), thresh=0.85)
    df['Diagnosis'] = LabelEncoder().fit_transform(
        pd.read_csv("wdbc.csv")["Diagnosis"].str.strip())
    df = remove_outliers(df, df.columns.difference(['Diagnosis']).tolist())

    # Composite biomarker via PCA
    comps, pca = compute_pca(df.drop(columns=['Diagnosis']), n_components=3)

    # Prepare data
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Hyperparameter tuning and training
    best_models = tune_models(X_train, y_train)

    # Evaluate
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
        display_feature_importance(model, X.columns.tolist())

    # Confusion matrices & ROC
    for name, model in best_models.items():
        cm = confusion_matrix(y_test, model.predict(X_test))
        disp = ConfusionMatrixDisplay(cm, display_labels=['B', 'M'])
        disp.plot()
        plt.title(name)
        plt.show()
    plt.figure(figsize=(8, 6))
    for name, model in best_models.items():
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title("ROC Curves")
    plt.show()

    # Robustness tests
    test_noise_robustness(best_models, X_test, y_test)
    test_outlier_robustness(best_models, X_test, y_test)

if __name__ == "__main__":
    main()
