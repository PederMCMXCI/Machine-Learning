# Customer Feedback Sentiment Analysis

## 📌 Projektbeschreibung
Dieses Projekt führt eine **binäre Sentiment-Analyse** (Positive / Negative) auf Kundenbewertungen durch.  
Die Originaldaten werden bereinigt, tokenisiert und anschließend mit vortrainierten Modellen (z. B. DistilBERT)  
oder klassischen Methoden (SVM + TF-IDF) klassifiziert.

## 📂 Dateien
- **raw_reviews.csv** → Rohdaten mit Spalten wie `Text` und `Sentiment`
- **cleaned_tokenized_sentiment_analysis.csv** → Vorverarbeitete Datei mit zusätzlicher Spalte `Cleaned_Text`
- **train_model.py** → Skript zum Trainieren und Evaluieren des Modells
- **preprocess_sentiment_dataset.py** → Skript zur Datenbereinigung und Tokenisierung

## ⚙️ Funktionen
1. **Datenbereinigung**
   - Entfernt Satzzeichen, Sonderzeichen, Zahlen und Stopwörter
   - Wandelt Wörter in Lemma-Form um
   - Bewahrt Negationen
2. **Modelltraining**
   - Fine-Tuning eines vortrainierten DistilBERT-Modells
   - Alternativ: Klassisches SVM-Modell mit TF-IDF-Vektorisierung
3. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix mit Heatmap
   - Ausgabe falsch klassifizierter Beispiele

## 🚀 Nutzung
```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. Preprocessing durchführen
python preprocess_sentiment_dataset.py

# 3. Modell trainieren
python train_model.py


Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1-Score: 1.00
