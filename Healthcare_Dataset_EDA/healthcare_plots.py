#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from pathlib import Path

# ==================== Konfiguration ====================
CSV_PATH = Path(r"D:\Machine_Learning\Healthcare Dataset - EDA\healthcare_dataset.cleaned.csv")
OUTDIR = Path("figures"); OUTDIR.mkdir(parents=True, exist_ok=True)
SHOW = True  # True = anzeigen + speichern; False = nur speichern

# ==================== Hilfsfunktionen ====================
def save_show(fig, name):
    p = OUTDIR / name
    fig.savefig(p, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)

def get_color_list(n):
    cmap = plt.get_cmap("tab20")
    return [cmap(i) for i in np.linspace(0, 1, max(n, 2))][:n]

def wrap_labels(labels, width=16):
    return ["\n".join(textwrap.wrap(str(l), width=width)) for l in labels]

def human_int(x):
    x = float(x)
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.1f}k"
    return f"{int(round(x))}"

def add_bar_labels(ax, rects, fmt="int", horizontal=False, pct_den=None):
    for i, r in enumerate(rects):
        val = r.get_width() if horizontal else r.get_height()
        if callable(fmt):
            txt = fmt(val)
        elif fmt == "human":
            txt = human_int(val)
        else:
            txt = f"{int(round(val))}"
        if pct_den is not None:
            den = pct_den[i] if isinstance(pct_den, (list, tuple, np.ndarray)) else pct_den
            if den and den > 0:
                pct = val / den * 100
                txt = f"{txt} ({pct:.1f}%)"
        if horizontal:
            ax.text(r.get_x() + val + (0.01 * max(1, ax.get_xlim()[1])),
                    r.get_y() + r.get_height()/2, txt, va="center", ha="left", fontsize=9)
        else:
            ax.text(r.get_x() + r.get_width()/2,
                    r.get_y() + val + (0.01 * max(1, ax.get_ylim()[1])),
                    txt, va="bottom", ha="center", fontsize=9)

def bar_plot(labels, values, title, xlabel, ylabel, fname,
             rotation=0, horizontal=False, label_fmt="int",
             pct_den=None):
    colors = get_color_list(len(labels))
    lab_wrapped = wrap_labels(labels, width=18)

    # dynamische Höhe bei vielen Kategorien (horizontale Balken)
    if horizontal:
        fig_h = max(4.5, min(0.35 * len(labels) + 1.5, 18))
    else:
        fig_h = 5.2

    fig, ax = plt.subplots(figsize=(10, fig_h))
    if horizontal:
        rects = ax.barh(lab_wrapped, values, color=colors, zorder=2)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
        xmax = max(values) if values else 1
        ax.set_xlim(0, xmax * 1.18)
    else:
        rects = ax.bar(lab_wrapped, values, color=colors, zorder=2)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if rotation:
            plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ymax = max(values) if values else 1
        ax.set_ylim(0, ymax * 1.18)

    ax.set_title(title)
    add_bar_labels(ax, rects, fmt=label_fmt, horizontal=horizontal, pct_den=pct_den)
    plt.tight_layout()
    save_show(fig, fname)

def top5_counts(series):
    s = series.astype(str).str.strip()
    counts = s.value_counts().head(5)
    labels = counts.index.tolist()
    values = counts.values.tolist()
    total_all = int(s.count())
    return labels, values, total_all

def all_counts(series):
    s = series.astype(str).str.strip()
    counts = s.value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    total_all = int(s.count())
    return labels, values, total_all

def top5_counts_numeric(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    counts = s.value_counts().head(5)
    labels = [int(x) if float(x).is_integer() else float(x) for x in counts.index]
    values = counts.values.tolist()
    total_all = int(s.count())
    return labels, values, total_all

# ==================== Daten laden ====================
df = pd.read_csv(CSV_PATH)

# ==================== 1) Häufigkeiten ====================
# Age (Top-5)
if "Age" in df.columns:
    labels, vals, total = top5_counts_numeric(df["Age"])
    bar_plot([str(x) for x in labels], vals,
             "Top 5 Age", "Age", "Anzahl",
             "top5_age.png", rotation=0, horizontal=False,
             label_fmt="int", pct_den=total)

# Spalten, die ALLE Kategorien zeigen sollen
SHOW_ALL = {"Blood Type", "Medical Condition", "Insurance Provider", "Medication"}

for col, title in [
    ("Gender","Gender"),
    ("Blood Type","Blood Types"),
    ("Medical Condition","Medical Conditions"),
    ("Hospital","Hospitals"),
    ("Doctor","Doctors"),
    ("Test Results","Test Results (Counts)"),
    ("Medication","Medications"),
    ("Admission Type","Admission Types"),
    ("Insurance Provider","Insurance Providers"),
]:
    if col in df.columns:
        if col in SHOW_ALL:
            labels, vals, total = all_counts(df[col])
            # bei vielen/langen Labels -> horizontal und dynamische Höhe
            horizontal = True if (len(labels) > 12 or any(len(str(x)) > 14 for x in labels)) else False
            rot = 0 if horizontal else (45 if any(len(str(x)) > 12 for x in labels) else 0)
            bar_plot(labels, vals,
                     f"{title} (alle Kategorien)", col, "Anzahl",
                     f"{title.lower().replace(' ','_')}_all.png",
                     rotation=rot, horizontal=horizontal,
                     label_fmt="int", pct_den=total)
        else:
            labels, vals, total = top5_counts(df[col])
            horizontal = any(len(str(x)) > 14 for x in labels)
            rot = 0 if horizontal else (45 if any(len(str(x)) > 12 for x in labels) else 0)
            bar_plot(labels, vals,
                     f"Top 5 {title}", col, "Anzahl",
                     f"top5_{title.lower().replace(' ','_')}.png",
                     rotation=rot, horizontal=horizontal,
                     label_fmt="int", pct_den=total)

# Room Number (Top-5)
if "Room Number" in df.columns:
    labels, vals, total = top5_counts_numeric(df["Room Number"])
    bar_plot([str(x) for x in labels], vals,
             "Top 5 Room Numbers", "Room Number", "Anzahl",
             "top5_room_number.png", rotation=0, horizontal=False,
             label_fmt="int", pct_den=total)

# ==================== 2) Datum: häufigster Monat je Jahr ====================
def most_common_month_per_year_plot(date_col, label, fname):
    if date_col not in df.columns:
        return
    s = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if s.empty: return
    tmp = pd.DataFrame({label: s})
    tmp["Year"] = tmp[label].dt.year
    tmp["Month"] = tmp[label].dt.month_name()

    years, top_months, top_counts, denoms = [], [], [], []
    for year, grp in tmp.groupby("Year"):
        counts = grp["Month"].value_counts()
        top_m = counts.idxmax()
        top_c = int(counts.max())
        years.append(str(int(year)))
        top_months.append(top_m)
        top_counts.append(top_c)
        denoms.append(int(len(grp)))

    bar_plot(years, top_counts,
             f"Häufigster Monat pro Jahr – {label}",
             "Jahr", "Anzahl",
             fname,
             rotation=0, horizontal=False,
             label_fmt="int", pct_den=denoms)

    # Zusatzplot mit Monatsnamen im Label
    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = get_color_list(len(years))
    rects = ax.bar(years, top_counts, color=colors, zorder=2)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_title(f"Häufigster Monat pro Jahr – {label} (mit Monatsnamen)")
    ax.set_xlabel("Jahr"); ax.set_ylabel("Anzahl")
    ymax = max(top_counts) if top_counts else 1
    ax.set_ylim(0, ymax * 1.22)
    for i, r in enumerate(rects):
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + (0.01*ymax),
                f"{top_months[i]}\n{top_counts[i]} ({top_counts[i]/denoms[i]*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_show(fig, fname.replace(".png", "_monthlabels.png"))

most_common_month_per_year_plot("Date of Admission", "Date of Admission",
                                "top_month_per_year_admission.png")
most_common_month_per_year_plot("Discharge Date", "Discharge Date",
                                "top_month_per_year_discharge.png")

# ==================== 3) Kosten-Top-5 (Summen) je Kategorie ====================
def top5_costs_plot(cat_col, title_stub, fname):
    bill = "Billing Amount"
    if cat_col not in df.columns or bill not in df.columns:
        return
    tmp = df[[cat_col, bill]].copy()
    tmp[bill] = pd.to_numeric(tmp[bill], errors="coerce")
    tmp = tmp.dropna(subset=[cat_col, bill])
    if tmp.empty: return
    agg = tmp.groupby(cat_col)[bill].sum().sort_values(ascending=False).head(5)
    labels = agg.index.tolist()
    vals = agg.values.tolist()
    total_cost_all = float(tmp[bill].sum())
    horizontal = any(len(str(x)) > 14 for x in labels)
    bar_plot(labels, vals,
             f"Top 5 {title_stub} (Gesamtkosten)",
             cat_col, "Summe Billing Amount",
             fname,
             rotation=45 if (not horizontal and any(len(str(x))>12 for x in labels)) else 0,
             horizontal=horizontal,
             label_fmt="human",
             pct_den=total_cost_all)

for col, stub, fn in [
    ("Hospital","Hospitals","top5_costs_hospital.png"),
    ("Blood Type","Blood Types","top5_costs_blood_type.png"),
    ("Medical Condition","Medical Conditions","top5_costs_med_condition.png"),
    ("Test Results","Test Results","top5_costs_test_results.png"),
    ("Medication","Medications","top5_costs_medication.png"),
]:
    top5_costs_plot(col, stub, fn)

# ==================== 4) Test Results – Breakdown (Age Groups & Gender) ====================
if "Test Results" in df.columns:
    # Age Groups
    if "Age" in df.columns:
        t = df.copy()
        t["Age"] = pd.to_numeric(t["Age"], errors="coerce")
        bins = [0,18,40,60,80,120]
        labels_ag = ["0-18","19-40","41-60","61-80","81+"]
        t["Age Group"] = pd.cut(t["Age"], bins=bins, labels=labels_ag, right=False)
        for res in t["Test Results"].astype(str).unique():
            sub = t[t["Test Results"] == res]
            vc = sub["Age Group"].value_counts().reindex(labels_ag, fill_value=0)
            total_res = int(sub.shape[0])
            bar_plot(vc.index.tolist(), vc.values.tolist(),
                     f"Age Groups – {res}", "Age Group", "Anzahl",
                     f"age_groups_{res.lower().replace(' ','_')}.png",
                     rotation=0, horizontal=False, label_fmt="int",
                     pct_den=total_res)
    # Gender
    if "Gender" in df.columns:
        t = df.copy()
        for res in t["Test Results"].astype(str).unique():
            sub = t[t["Test Results"] == res]
            vc = sub["Gender"].astype(str).str.strip().value_counts()
            total_res = int(sub.shape[0])
            bar_plot(vc.index.tolist(), vc.values.tolist(),
                     f"Gender – {res}", "Gender", "Anzahl",
                     f"gender_{res.lower().replace(' ','_')}.png",
                     rotation=0, horizontal=False, label_fmt="int",
                     pct_den=total_res)
