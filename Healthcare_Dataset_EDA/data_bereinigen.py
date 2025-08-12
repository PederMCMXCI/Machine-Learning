#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Healthcare Cleanup
- Trim/Normalize Strings (Gender, Blood Type, Hospital, Doctor, etc.)
- Numerik: Age, Billing Amount, Room Number
- Dates: Date of Admission / Discharge Date
- Duplikate entfernen, Ausreißer (Billing) winsorisieren
- Konsolen-Report + optionales Speichern einer bereinigten CSV
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ====== Pfade / Optionen ======
INPATH  = Path(r"D:\Machine_Learning\Healthcare Dataset - EDA\healthcare_dataset.csv")
OUTPATH = INPATH.with_name("healthcare_dataset.cleaned.csv")
SAVE_CLEANED = True           # auf False, wenn du nicht speichern willst
AGE_MIN, AGE_MAX = 0, 120
BILL_WINSOR_Q = 0.01          # 1% Winsorizing (untere/obere Ausreißer kappen); auf None setzen zum Abschalten
DROP_NEGATIVE_BILL = True     # negative/0-Beträge entfernen
COERCE_DISCHARGE_BEFORE_ADMISSION_TO_NA = True  # falsche Reihenfolge -> Discharge auf NaT

# ====== Hilfen ======
def col(df, name): return name in df.columns

def normalize_strings(df, cols, title_case=False):
    for c in cols:
        if not col(df, c): continue
        s = df[c].astype(str).str.strip()
        s = s.str.replace(r"\s+", " ", regex=True)
        df[c] = s.str.title() if title_case else s
    return df

def clean_gender(df):
    if not col(df, "Gender"): return df
    m = {
        "m":"Male","male":"Male","f":"Female","female":"Female",
        "w":"Female","weiblich":"Female","männlich":"Male",
        "other":"Other","divers":"Other","non-binary":"Other","nb":"Other"
    }
    s = df["Gender"].astype(str).str.strip().str.lower()
    df["Gender"] = s.map(m).fillna(df["Gender"]).astype(str).str.strip().str.title()
    return df

def clean_blood_type(df):
    if not col(df, "Blood Type"): return df
    s = df["Blood Type"].astype(str).str.strip().str.replace(" ", "", regex=False).str.upper()
    valid = {"A+","A-","B+","B-","AB+","AB-","O+","O-"}
    s = s.where(s.isin(valid), pd.NA)
    df["Blood Type"] = s
    return df

def clean_age(df):
    if not col(df, "Age"): return df
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    before = len(df)
    df = df[df["Age"].between(AGE_MIN, AGE_MAX, inclusive="both")]
    print(f"Age: entfernt {before - len(df)} Zeilen außerhalb [{AGE_MIN},{AGE_MAX}] oder NaN.")
    df["Age"] = df["Age"].round().astype("Int64")
    return df

def clean_room_number(df):
    if not col(df, "Room Number"): return df
    df["Room Number"] = pd.to_numeric(df["Room Number"], errors="coerce").round().astype("Int64")
    return df

def clean_billing(df):
    if not col(df, "Billing Amount"): return df
    df["Billing Amount"] = pd.to_numeric(df["Billing Amount"], errors="coerce")
    if DROP_NEGATIVE_BILL:
        before = len(df)
        df = df[df["Billing Amount"] > 0]
        print(f"Billing: entfernt {before - len(df)} Zeilen mit Betrag ≤ 0 oder NaN.")
    if BILL_WINSOR_Q is not None and 0 < BILL_WINSOR_Q < 0.5:
        lo = df["Billing Amount"].quantile(BILL_WINSOR_Q)
        hi = df["Billing Amount"].quantile(1 - BILL_WINSOR_Q)
        df["Billing Amount"] = df["Billing Amount"].clip(lower=lo, upper=hi)
        print(f"Billing: winsorisiert auf [{lo:,.2f}, {hi:,.2f}] (q={BILL_WINSOR_Q:.2%}).")
    return df

def clean_dates(df):
    # robuste Datumsparse
    for c in ["Date of Admission", "Discharge Date"]:
        if not col(df, c): continue
        df[c] = pd.to_datetime(df[c], errors="coerce")
        print(f"{c}: gültige Datumswerte: {df[c].notna().sum()} / {len(df)}")

    # Admission muss existieren; Discharge darf fehlen
    if col(df, "Date of Admission"):
        before = len(df)
        df = df.dropna(subset=["Date of Admission"])
        print(f"Dates: entfernt {before - len(df)} Zeilen ohne Date of Admission.")

    # Unplausible Reihenfolge: Discharge vor Admission -> Discharge auf NaT (oder tauschen, wenn du willst)
    if col(df, "Date of Admission") and col(df, "Discharge Date") and COERCE_DISCHARGE_BEFORE_ADMISSION_TO_NA:
        mask = df["Discharge Date"].notna() & (df["Discharge Date"] < df["Date of Admission"])
        n_bad = int(mask.sum())
        if n_bad:
            df.loc[mask, "Discharge Date"] = pd.NaT
            print(f"Dates: {n_bad} Discharge<Admission → Discharge Date auf NaT gesetzt.")
    return df

def drop_dupes(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplikate: entfernt {before - len(df)} exakte Duplikate.")
    return df

# ====== Pipeline ======
def clean_healthcare_df(df):
    df = df.copy()

    # 1) Strings trimmen/normalisieren
    str_cols = df.select_dtypes(include="object").columns.tolist()
    df = normalize_strings(df, str_cols, title_case=False)
    df = normalize_strings(df, ["Hospital","Doctor"], title_case=True)

    # 2) Domänenspezifisch
    df = clean_gender(df)
    df = clean_blood_type(df)
    df = clean_age(df)
    df = clean_room_number(df)
    df = clean_billing(df)
    df = clean_dates(df)

    # 3) Duplikate
    df = drop_dupes(df)

    # 4) Kurzer Bericht
    print("\n=== Kurzbericht nach Bereinigung ===")
    for c in ["Age","Gender","Blood Type","Medical Condition","Hospital","Doctor",
              "Insurance Provider","Admission Type","Room Number","Billing Amount",
              "Date of Admission","Discharge Date","Medication","Test Results"]:
        if col(df, c):
            na = int(df[c].isna().sum())
            print(f"{c}: {len(df)-na} vorhanden | {na} NaN")

    return df

def main():
    if not INPATH.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {INPATH}")
    raw = pd.read_csv(INPATH)
    print(f"Eingelesen: {len(raw)} Zeilen, {len(raw.columns)} Spalten.")
    cleaned = clean_healthcare_df(raw)
    print(f"\nFertig: {len(cleaned)} Zeilen nach Bereinigung.")

    if SAVE_CLEANED:
        cleaned.to_csv(OUTPATH, index=False)
        print(f"✔ Gespeichert: {OUTPATH}")

if __name__ == "__main__":
    main()
