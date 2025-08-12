Healthcare EDA Toolkit — Zusammenfassung
Dieses Repository enthält ein schlankes Toolkit für Exploratory Data Analysis (EDA) im Gesundheitskontext – mit Datenbereinigung, Konsolenreports und Visualisierungen als eigenständige Python-Skripte (ohne Notebook).

Inhalt
healthcare_clean.py – Bereinigt die Roh-CSV (String-Normalisierung, Datentypen, Datumsparsing, Ausreißer/Winsorizing, Duplikate) und erzeugt healthcare_dataset.cleaned.csv.

healthcare_full_console_report.py – Konsolenbasierte Auswertung:

Häufigkeiten (Top-5 oder alle, je nach Spalte)

Häufigster Monat pro Jahr für Date of Admission & Discharge Date

Kosten-Top-5 (nach Billing Amount) für Hospital, Blood Type, Medical Condition, Test Results, Medication

Kombi-Analysen zu Test Results (Altersgruppen, Gender, Krankheiten, Krankenhäuser, Medikamente, Monate)

healthcare_plots_pretty.py – Erzeugt farbige Balkendiagramme mit Grid, Wertlabels & Prozent; speichert PNGs nach ./figures/.

Voraussetzungen
Python 3.9+

Pakete: pandas, matplotlib, numpy

Kurz-Nutzung
Bereinigen: python healthcare_clean.py

Report: python healthcare_full_console_report.py

Plots: python healthcare_plots_pretty.py

Pfad zur CSV jeweils über CSV_PATH anpassen (für Reports/Plots ideal die bereinigte Datei *.cleaned.csv verwenden).

Ausgaben
Bereinigte Datei: healthcare_dataset.cleaned.csv

Konsolenberichte (Textausgabe)

Grafiken (PNG) im Ordner figures/