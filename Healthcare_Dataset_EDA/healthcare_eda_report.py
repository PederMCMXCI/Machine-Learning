#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

# ====== Pfad zur CSV anpassen ======
CSV_PATH = Path(r"D:\Machine_Learning\Healthcare Dataset - EDA\healthcare_dataset.cleaned.csv")

# ====== Laden ======
df = pd.read_csv(CSV_PATH)

# ====== Helfer ======
def exists(col):
    return col in df.columns

def counts_with_percentage(series, label, top_n=5, show_all=False):
    """
    Gibt Häufigkeiten + Prozent aus.
    - show_all=True -> alle Kategorien (kein top_n-Limit)
    - show_all=False -> Top-N (Standard: 5)
    """
    s = series.astype(str).str.strip()
    counts_full = s.value_counts()
    counts = counts_full if show_all else counts_full.head(top_n)
    total = int(s.shape[0])
    perc = counts / total * 100

    header = f"=== Häufigkeiten {label} (alle Kategorien) ===" if show_all \
             else f"=== Top {top_n} häufigste {label} ==="
    print(f"\n{header}")
    for val, cnt in counts.items():
        print(f"{val}: {cnt} ({perc[val]:.2f}%)")

def top5_counts_numeric(series, label):
    s = pd.to_numeric(series, errors="coerce").dropna()
    counts = s.value_counts().head(5)
    total = int(s.shape[0])
    perc = counts / total * 100
    print(f"\n=== Top 5 häufigste {label} ===")
    for val, cnt in counts.items():
        vv = int(val) if float(val).is_integer() else float(val)
        print(f"{vv}: {cnt} ({perc[val]:.2f}%)")

def most_common_month_per_year(date_series, label):
    s = pd.to_datetime(date_series, errors="coerce").dropna()
    if s.empty:
        print(f"\n[Übersprungen] {label}: keine gültigen Datumswerte.")
        return
    tmp = pd.DataFrame({label: s})
    tmp["Year"] = tmp[label].dt.year
    tmp["Month"] = tmp[label].dt.month_name()
    print(f"\n=== Häufigster Monat pro Jahr ({label}) ===")
    for year, grp in tmp.groupby("Year"):
        mc = grp["Month"].value_counts()
        top_m = mc.idxmax()
        top_c = int(mc.max())
        year_total = int(len(grp))
        pct = top_c / year_total * 100
        print(f"{year}: {top_m} – {top_c} Fälle ({pct:.2f}% von {year_total})")

def top5_costs_by_category(cat_col, bill_col="Billing Amount"):
    if not (exists(cat_col) and exists(bill_col)):
        print(f"\n[Übersprungen] '{cat_col}' oder '{bill_col}' fehlt.")
        return
    tmp = df[[cat_col, bill_col]].copy()
    tmp[bill_col] = pd.to_numeric(tmp[bill_col], errors="coerce")
    tmp = tmp.dropna(subset=[cat_col, bill_col])
    if tmp.empty:
        print(f"\n[Übersprungen] '{cat_col}': keine validen Werte.")
        return
    agg = tmp.groupby(cat_col)[bill_col].sum().sort_values(ascending=False)
    grand_total = agg.sum()
    print(f"\n=== Top 5 teuerste '{cat_col}' nach Gesamtkosten ===")
    for cat, total in agg.head(5).items():
        pct = total / grand_total * 100
        print(f"{cat}: {total:,.2f} ({pct:.2f}%)")

# ====== A) Häufigkeiten ======
if exists("Age"): top5_counts_numeric(df["Age"], "Age")
if exists("Gender"): counts_with_percentage(df["Gender"], "Gender", top_n=5, show_all=False)

# Diese vier: ALLE Kategorien zeigen (kein Top-Limit)
if exists("Blood Type"): counts_with_percentage(df["Blood Type"], "Blood Types", show_all=True)
if exists("Medical Condition"): counts_with_percentage(df["Medical Condition"], "Medical Conditions", show_all=True)
if exists("Insurance Provider"): counts_with_percentage(df["Insurance Provider"], "Insurance Providers", show_all=True)
if exists("Medication"): counts_with_percentage(df["Medication"], "Medications", show_all=True)

# Rest weiterhin Top-5
if exists("Hospital"): counts_with_percentage(df["Hospital"], "Hospitals", top_n=5, show_all=False)
if exists("Doctor"): counts_with_percentage(df["Doctor"], "Doctors", top_n=5, show_all=False)
if exists("Test Results"): counts_with_percentage(df["Test Results"], "Test Results", top_n=5, show_all=False)
if exists("Admission Type"): counts_with_percentage(df["Admission Type"], "Admission Types", top_n=5, show_all=False)
if exists("Room Number"): top5_counts_numeric(df["Room Number"], "Room Numbers")

# ====== B) Datum: häufigster Monat pro Jahr ======
if exists("Date of Admission"): most_common_month_per_year(df["Date of Admission"], "Date of Admission")
if exists("Discharge Date"): most_common_month_per_year(df["Discharge Date"], "Discharge Date")

# ====== C) Kostenanalysen (Billing Amount) – Top-5 nach Gesamtsumme ======
if exists("Billing Amount"):
    top5_costs_by_category("Hospital")
    top5_costs_by_category("Blood Type")
    top5_costs_by_category("Medical Condition")
    top5_costs_by_category("Test Results")
    top5_costs_by_category("Medication")
else:
    print("\n[Hinweis] 'Billing Amount' nicht vorhanden – Kostenanalysen übersprungen.")

# ====== D) Test-Results Kombi-Analysen ======
if exists("Test Results"):
    print("\n=== Test Results – Kombi-Analysen ===")
    # 1) Kosten pro Test Result
    if exists("Billing Amount"):
        print("\n-- Kosten pro Test Result (Gesamt / Ø / Median) --")
        costs = df.copy()
        costs["Billing Amount"] = pd.to_numeric(costs.get("Billing Amount"), errors="coerce")
        stats = costs.groupby("Test Results")["Billing Amount"].agg(
            total="sum", mean="mean", median="median", count="count"
        ).sort_values("total", ascending=False)
        for res, row in stats.iterrows():
            print(f"{res}: Gesamt {row['total']:,.2f} | Ø {row['mean']:,.2f} | Median {row['median']:,.2f} | Fälle {row['count']}")
    # 2) Altersgruppen
    if exists("Age"):
        print("\n-- Häufigste Altersgruppen je Test Result --")
        bins = [0,18,40,60,80,120]; labels = ["0-18","19-40","41-60","61-80","81+"]
        tmp = df.copy()
        tmp["Age Group"] = pd.cut(pd.to_numeric(tmp["Age"], errors="coerce"), bins=bins, labels=labels, right=False)
        for res in tmp["Test Results"].astype(str).unique():
            sub = tmp[tmp["Test Results"]==res]
            vc = sub["Age Group"].value_counts().head(5)
            total = int(sub.shape[0]) if sub is not None else 0
            print(f"\n{res}:")
            for grp, cnt in vc.items():
                pct = cnt / total * 100 if total else 0
                print(f"  {grp}: {cnt} ({pct:.2f}%)")
    # 3) Geschlecht
    if exists("Gender"):
        print("\n-- Geschlecht je Test Result --")
        tmp = df.groupby(["Test Results","Gender"]).size().reset_index(name="count")
        for res in df["Test Results"].astype(str).unique():
            sub = tmp[tmp["Test Results"]==res].sort_values("count", ascending=False)
            total = int(df[df["Test Results"]==res].shape[0])
            print(f"\n{res}:")
            for _, row in sub.iterrows():
                pct = row["count"] / total * 100 if total else 0
                print(f"  {row['Gender']}: {row['count']} ({pct:.2f}%)")
    # 4) Krankheiten
    if exists("Medical Condition"):
        print("\n-- Häufigste Krankheiten je Test Result --")
        for res in df["Test Results"].astype(str).unique():
            sub = df[df["Test Results"]==res]
            vc = sub["Medical Condition"].astype(str).value_counts().head(5)
            total = int(sub.shape[0])
            print(f"\n{res}:")
            for cond, cnt in vc.items():
                print(f"  {cond}: {cnt} ({cnt/total*100:.2f}%)")
    # 5) Krankenhäuser
    if exists("Hospital"):
        print("\n-- Top Krankenhäuser je Test Result --")
        for res in df["Test Results"].astype(str).unique():
            sub = df[df["Test Results"]==res]
            vc = sub["Hospital"].astype(str).value_counts().head(5)
            total = int(sub.shape[0])
            print(f"\n{res}:")
            for hosp, cnt in vc.items():
                print(f"  {hosp}: {cnt} ({cnt/total*100:.2f}%)")
    # 6) Medikamente
    if exists("Medication"):
        print("\n-- Häufigste Medikamente je Test Result --")
        for res in df["Test Results"].astype(str).unique():
            sub = df[df["Test Results"]==res]
            vc = sub["Medication"].astype(str).value_counts().head(5)
            total = int(sub.shape[0])
            print(f"\n{res}:")
            for med, cnt in vc.items():
                print(f"  {med}: {cnt} ({cnt/total*100:.2f}%)")
    # 7) Zeit (Monat)
    if exists("Date of Admission"):
        print("\n-- Häufigste Monate je Test Result (Aufnahmen) --")
        tmp = df.copy()
        tmp["Date of Admission"] = pd.to_datetime(tmp["Date of Admission"], errors="coerce")
        tmp["Month-Year"] = tmp["Date of Admission"].dt.to_period("M")
        for res in tmp["Test Results"].astype(str).unique():
            sub = tmp[tmp["Test Results"]==res]
            vc = sub["Month-Year"].value_counts().head(5)
            total = int(sub.shape[0])
            print(f"\n{res}:")
            for my, cnt in vc.items():
                print(f"  {my}: {cnt} ({cnt/total*100:.2f}%)")
else:
    print("\n[Hinweis] 'Test Results' nicht vorhanden – Kombi-Analysen übersprungen.")
