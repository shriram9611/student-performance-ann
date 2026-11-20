"""Train the baseline + Keras ANN, evaluate, save model and reports.

Run:  python train.py
Outputs:
  models/student_performance.keras   trained ANN
  models/scaler.joblib               fitted StandardScaler
  models/metadata.json               features, classes, metrics
  reports/*.png                      plots
  reports/evaluation_report.md       full evaluation report
"""
import json
import os
import random

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from tensorflow.keras import layers

from common import (CLASSES, FEATURES, META_PATH, MODEL_DIR, MODEL_PATH, REPORT_DIR,
                    SCALER_PATH, SEED, TARGET, data_summary, evaluate, load_data,
                    prepare_data)

# Reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


def save_confusion(cm, title, path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_ann(n_features: int) -> keras.Model:
    """Input -> Dense(32, ReLU) -> Dropout -> Dense(16, ReLU) -> Dense(3, Softmax)."""
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(len(CLASSES), activation="softmax"),
    ], name="student_performance_ann")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",   # integer labels 0/1/2
        metrics=["accuracy"],
    )
    return model


def main():
    # ---- Step 1: data understanding ------------------------------------
    df = load_data()
    summary = data_summary(df)
    print("Data summary:", json.dumps(summary, indent=2))

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET, order=CLASSES)
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 7))
    sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature correlation")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # ---- Step 2: preprocessing -----------------------------------------
    d = prepare_data(df)
    print({k: v.shape for k, v in d.items() if k != "scaler"})

    # ---- Step 3: baseline (Logistic Regression) ------------------------
    baseline = LogisticRegression(max_iter=1000, random_state=SEED)
    baseline.fit(d["X_train"], d["y_train"])
    base_res = evaluate(d["y_test"], baseline.predict(d["X_test"]))
    print("Baseline accuracy:", round(base_res["accuracy"], 4))

    # ---- Steps 4-5: ANN design + training ------------------------------
    model = build_ann(len(FEATURES))
    model.summary()
    history = model.fit(
        d["X_train"], d["y_train"],
        validation_data=(d["X_val"], d["y_val"]),
        epochs=150, batch_size=64, verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
        ],
    )

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="validation")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[1].plot(history.history["accuracy"], label="train")
    ax[1].plot(history.history["val_accuracy"], label="validation")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "training_curves.png", dpi=150)
    plt.close()

    # ---- Step 7: evaluation on unseen test data ------------------------
    probs = model.predict(d["X_test"], verbose=0)
    ann_res = evaluate(d["y_test"], probs.argmax(axis=1))
    print("ANN accuracy:", round(ann_res["accuracy"], 4))
    save_confusion(np.array(ann_res["confusion_matrix"]), "ANN - confusion matrix",
                   REPORT_DIR / "confusion_matrix_ann.png")
    save_confusion(np.array(base_res["confusion_matrix"]), "Logistic Regression - confusion matrix",
                   REPORT_DIR / "confusion_matrix_baseline.png")

    # ---- Step 8: save model --------------------------------------------
    model.save(MODEL_PATH)
    joblib.dump(d["scaler"], SCALER_PATH)
    META_PATH.write_text(json.dumps({
        "features": FEATURES, "classes": CLASSES,
        "epochs_trained": len(history.history["loss"]),
        "ann": ann_res, "baseline": base_res,
    }, indent=2))

    write_report(summary, base_res, ann_res, len(history.history["loss"]), model)
    print(f"Saved model to {MODEL_PATH}")


def write_report(summary, base, ann, epochs, model):
    lines = []
    add = lines.append
    add("# Evaluation Report - Student Performance ANN\n")
    add(f"Dataset: {summary['rows']} students, {len(FEATURES)} input features, "
        f"target `{TARGET}` with classes {', '.join(CLASSES)}.  ")
    add(f"Missing values: {summary['missing_values']}, duplicate rows: {summary['duplicate_rows']}.  ")
    add(f"Class distribution: {summary['class_distribution']}.\n")
    add("Split: 70% train / 15% validation / 15% test (stratified, seed 42). "
        "StandardScaler fitted on the training split only.\n")
    add("## Model comparison (test set)\n")
    add("| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |")
    add("|---|---|---|---|---|")
    for name, r in (("Logistic Regression (baseline)", base), ("ANN (Keras)", ann)):
        add(f"| {name} | {r['accuracy']:.4f} | {r['precision_macro']:.4f} | "
            f"{r['recall_macro']:.4f} | {r['f1_macro']:.4f} |")
    add("\n## ANN per-class results\n")
    add("| Class | Precision | Recall | F1 | Support |")
    add("|---|---|---|---|---|")
    for c, v in ann["per_class"].items():
        add(f"| {c} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")
    add("\n## Confusion matrix (ANN, rows = actual, columns = predicted)\n")
    add("| | " + " | ".join(CLASSES) + " |")
    add("|---|" + "---|" * len(CLASSES))
    for c, row in zip(CLASSES, ann["confusion_matrix"]):
        add(f"| {c} | " + " | ".join(map(str, row)) + " |")
    add("\n![ANN confusion matrix](confusion_matrix_ann.png)\n")
    add("![Training curves](training_curves.png)\n")
    add("## Architecture\n")
    add("Input(%d) -> Dense(32, ReLU) -> Dropout(0.2) -> Dense(16, ReLU) -> Dense(3, Softmax). "
        "Total parameters: %d. Adam (lr 1e-3), sparse categorical cross-entropy, batch size 64, "
        "early stopping on validation loss (%d epochs run).\n" % (len(FEATURES), model.count_params(), epochs))
    add("## Observations\n")
    diff = ann["accuracy"] - base["accuracy"]
    add(f"- The ANN differs from the Logistic Regression baseline by {diff:+.4f} accuracy. "
        "A small difference suggests the classes are mostly separable by roughly linear relationships "
        "in these features, so extra network capacity adds little.")
    worst = min(ann["per_class"].items(), key=lambda kv: kv[1]["f1"])[0]
    add(f"- The weakest class by F1 is **{worst}**; check the confusion matrix to see which neighbouring "
        "class it is confused with (Medium is usually confused with Low/High because it lies between them).")
    add("- Study hours per week is the feature whose class means differ most (see correlation heatmap and notebook).")
    add("- Train and validation curves staying close indicates the dropout and early stopping controlled overfitting.")
    (REPORT_DIR / "evaluation_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
