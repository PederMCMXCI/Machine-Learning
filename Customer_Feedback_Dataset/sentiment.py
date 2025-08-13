# preprocess_with_spacy_fixed_paths.py
# Erzeugt CSV im Format:
# Text; Sentiment; Source; Date/Time; User ID; Location; Confidence Score; Cleaned_Text

import os
import re
import pandas as pd
import regex as reg
from unidecode import unidecode
import spacy

# ===== Feste Pfade =====
INPUT_PATH  = r"D:\Machine_Learning\Customer_Feedback_Dataset\cleaned_sentiment_analysis.csv"
OUTPUT_PATH = r"D:\Machine_Learning\Customer_Feedback_Dataset\cleaned_tokenized_sentiment_analysis_2.csv"

# ===== spaCy laden (vortrainiertes Modell) =====
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

# ===== Grobe Vorreinigung (vor dem NLP-Lauf) =====
_url_pat   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_email_pat = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_user_pat  = re.compile(r"@[A-Za-z0-9_]+")
_hash_pat  = re.compile(r"#\w+")
_html_pat  = re.compile(r"<[^>]+>")
_emoji_pat = reg.compile(r"\p{Extended_Pictographic}", reg.UNICODE)

# Negationen, die wir NICHT entfernen wollen
NEGATIONS = {"not", "no", "never", "nor", "n't", "dont", "don't", "cant", "can't", "cannot", "wont", "won't"}

def preclean(text: str) -> str:
    if pd.isna(text):
        return ""
    x = str(text)
    x = unidecode(x)
    x = x.replace("\n", " ").replace("\r", " ").lower()
    x = _html_pat.sub(" ", x)
    x = _url_pat.sub(" ", x)
    x = _email_pat.sub(" ", x)
    x = _user_pat.sub(" ", x)
    x = _hash_pat.sub(" ", x)
    x = _emoji_pat.sub(" ", x)
    return x

def spacy_clean(texts):
    """
    Cleaned_Text:
    - Tokenisierung + Lemmatisierung (spaCy)
    - Stopwörter raus (token.is_stop), NEGATIONS bleiben drin
    - Satzzeichen/Zahlen/URLs/Mails/Emojis/Leerraum werden entfernt
    - nur einfache alphanumerische Tokens (inkl. 'n't)
    """
    cleaned = []
    for doc in nlp.pipe(texts, batch_size=256):
        toks = []
        for t in doc:
            if t.is_space or t.is_punct or t.like_num:
                continue
            if t.like_url or t.like_email:
                continue
            lemma = t.lemma_.lower().strip()
            if not lemma:
                continue
            # Stopwort entfernen, außer es ist eine Negation
            if t.is_stop and lemma not in NEGATIONS:
                continue
            # nur a–z / 0–9 / ' erlauben
            if not re.match(r"^[a-z0-9']+$", lemma):
                continue
            toks.append(lemma)
        cleaned.append(" ".join(toks))
    return cleaned

def main():
    print(f"📥 Lese: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    # Zielspalten wie im Screenshot (fehlende werden leer angelegt)
    required = ["Text","Sentiment","Source","Date/Time","User ID","Location","Confidence Score"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    # Vorreinigung + spaCy-Cleaning
    raw_texts = df["Text"].astype(str).map(preclean)
    df["Cleaned_Text"] = spacy_clean(raw_texts.tolist())

    # Sentiment vereinheitlichen (nur Positive/Negative durchlassen)
    df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.capitalize()
    df = df[df["Sentiment"].isin(["Positive","Negative"])].copy()

    # Spalten exakt sortieren
    df_out = df[["Text","Sentiment","Source","Date/Time","User ID","Location","Confidence Score","Cleaned_Text"]]

    # Speichern (Semikolon)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, sep=";", index=False, encoding="utf-8")

    print(f"✅ Fertig! Gespeichert: {OUTPUT_PATH}")
    print(f"Zeilen: {len(df_out)} | Positive: {(df_out['Sentiment']=='Positive').sum()} | Negative: {(df_out['Sentiment']=='Negative').sum()}")

if __name__ == "__main__":
    main()
