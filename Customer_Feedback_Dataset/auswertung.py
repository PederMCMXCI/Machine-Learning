# fine_tune_distilbert_simple.py
# DistilBERT-Feintuning für binäre Sentiments (Positive/Negative)
# - Lädt dein ;-getrenntes CSV
# - Nutzt Cleaned_Text + Sentiment
# - Train/Val-Split (stratifiziert)
# - Metriken für TRAIN & VALIDATION
# - Speichert Modell + Tokenizer + Validation-Predictions

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)

# =========================
# Konfiguration
# =========================
CSV_PATH    = r"D:\Machine_Learning\Customer_Feedback_Dataset\cleaned_tokenized_sentiment_analysis_2.csv"
TEXT_COL    = "Cleaned_Text"
LABEL_COL   = "Sentiment"
# Guter Startpunkt (bereits sentiment-vorgetrimmt). Du kannst auch "distilbert-base-uncased" setzen.
MODEL_NAME  = "distilbert-base-uncased-finetuned-sst-2-english"

OUTPUT_DIR  = "./outputs_distilbert"
EPOCHS      = 8
BATCH_SIZE  = 8
LR          = 2e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH  = 128
SEED        = 42
PIN_MEMORY  = False  # auf CPU Warnung vermeiden

LABELS      = ["Negative", "Positive"]
LABEL2ID    = {"Negative": 0, "Positive": 1}
ID2LABEL    = {0: "Negative", 1: "Positive"}

# =========================
# Datenvorbereitung
# =========================
def load_and_prepare(csv_path, text_col, label_col):
    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip()
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Spalten nicht gefunden. Verfügbar: {list(df.columns)}")
    df[text_col]  = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip().str.capitalize()
    df = df[df[label_col].isin(LABELS)].copy()
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)  # harte Duplikate raus
    df["label_id"] = df[label_col].map(LABEL2ID)
    return df

class TextDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def print_block_metrics(y_true, y_pred, title):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n===== {title} =====")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
    return acc, report, cm

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    # Daten laden
    df = load_and_prepare(CSV_PATH, TEXT_COL, LABEL_COL)

    # Train/Val Split
    tr, va = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label_id"]
    )
    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)

    # Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds  = TextDataset(tr[TEXT_COL].tolist(), tr["label_id"].tolist(), tokenizer, MAX_LENGTH)
    val_ds    = TextDataset(va[TEXT_COL].tolist(), va["label_id"].tolist(), tokenizer, MAX_LENGTH)

    # Modell
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    # TrainingArguments – bewusst ohne evaluation/save-strategy, damit es auch mit älteren Versionen läuft
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=10,
        report_to=["none"],
        seed=SEED,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        dataloader_pin_memory=PIN_MEMORY,
        load_best_model_at_end=False,  # kompatibel halten
        save_strategy="no",            # kein Autospeichern zwischen Epochen
        # KEIN evaluation_strategy => wir evaluieren manuell danach
    )

    # Trainer (ohne EarlyStopping, um Kompatibilität zu wahren)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,   # nur für predict() später
        # processing_class=tokenizer  # optional: kann Deprecation-Warnung verhindern
    )

    # Training
    trainer.train()

    # === Manuelle Evaluation: TRAIN ===
    train_out = trainer.predict(train_ds)
    train_preds = np.argmax(train_out.predictions, axis=1)
    print_block_metrics(tr["label_id"].to_numpy(), train_preds, "Train Set")

    # === Manuelle Evaluation: VALIDATION ===
    val_out  = trainer.predict(val_ds)
    val_preds = np.argmax(val_out.predictions, axis=1)
    acc, val_report, val_cm = print_block_metrics(va["label_id"].to_numpy(), val_preds, "Validation Set")

    # Predictions-CSV (Validation)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = os.path.join(OUTPUT_DIR, f"validation_predictions_{ts}.csv")
    pd.DataFrame({
        "Text": va[TEXT_COL].tolist(),
        "True_Label": va[LABEL_COL].tolist(),
        "Pred_Label": [ID2LABEL[i] for i in val_preds]
    }).to_csv(pred_path, index=False)
    print(f"\n✔ Validation-Predictions gespeichert: {os.path.abspath(pred_path)}")

    # Modell & Tokenizer speichern
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✔ Modell & Tokenizer gespeichert in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
