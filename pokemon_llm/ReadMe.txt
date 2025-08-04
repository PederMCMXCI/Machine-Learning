# 🧠 Pokémon LLM – Intelligente Pokédex-Fragen mit RAG & LLM

Dieses Projekt kombiniert **strukturierte Datenanalyse** mit **Retrieval-Augmented Generation (RAG)**, um komplexe Fragen zu Pokémon zu beantworten. Nutzer können entweder nach konkreten Werten filtern (z. B. „Zeige mir alle Pokémon mit mehr als 100 Angriff“) oder frei formulierte Fragen stellen (z. B. „Was ist das stärkste Wasser-Pokémon?“).

---

## 📦 Was das Projekt macht

✅ Unterstützt zwei Arten von Fragen:
1. **Strukturierte Fragen** → Zahlen, Vergleiche, Sortierungen (z. B. „Speed between 50 and 100“)
2. **Unstrukturierte Fragen (natürliche Sprache)** → über ein lokal laufendes Sprachmodell (LLM)

✅ Verwendet:
- 🧠 LlamaCpp + GGUF-Modell (z. B. Llama 2 7B)
- 📚 FAISS-Vektorindex basierend auf Pokémon-Beschreibungen
- 🌐 DeepL-ähnliche automatische Übersetzung mit `deep-translator`
- 🔍 Kontextuelle Anzeige passender Pokémon-Daten direkt aus der CSV

---

## 📁 Projektstruktur

```text
pokemon-llm/
├── main.py                  # Hauptskript mit Benutzerinteraktion (CLI)
├── pokemon_llm.py           # RAG-Logik mit FAISS + LLM + Antwortgenerierung
├── structured_query.py      # Verarbeitung strukturierter Fragen
├── requirements.txt         # Python-Abhängigkeiten
├── README.md                # Dieses Dokument
├── .gitignore               # Ausgeschlossene Dateien wie Modell & CSV
