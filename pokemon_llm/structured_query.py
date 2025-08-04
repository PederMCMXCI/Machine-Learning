# === FILE: test_structur.py ===
import pandas as pd
import re

csv_path = r"D:\Machine_Learning\pokemon_llm\pokedex_with_text.csv"

def handle_structured_question(question: str):
    df = pd.read_csv(csv_path, encoding="ISO-8859-1", sep=";")
    df.columns = df.columns.str.strip().str.lower()
    df['name'] = df['name'].str.lower()

    q = question.lower()

    # === Superlativ-Erkennung (z. B. highest attack, tallest, lowest hp) ===
    superlative_keywords = ["highest", "most", "max", "größte", "stärkste", "tallest"]
    lowest_keywords = ["lowest", "least", "min", "kleinste", "schwächste"]

    # dynamisch passende Spalte für "tallest" etc. finden
    keyword_map = {
        "tallest": "height",
        "heaviest": "weight",
    }

    for col in df.columns:
        if col in q:
            if any(k in q for k in superlative_keywords):
                result = df.sort_values(by=col, ascending=False).head(1)
                return result
            if any(k in q for k in lowest_keywords):
                result = df.sort_values(by=col, ascending=True).head(1)
                return result

    for keyword, column in keyword_map.items():
        if keyword in q and column in df.columns:
            result = df.sort_values(by=column, ascending=False).head(1)
            return result

    # === Dynamische Bereichsfilterung für numerische Spalten ===
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for col in numeric_cols:
        if col in q:
            match = re.search(rf"{col}.*?(?:between|zwischen)?\s*(\d+)\s*(?:-|\u2013|to|bis|und)\s*(\d+)", q)
            if match:
                min_v, max_v = int(match.group(1)), int(match.group(2))
                result = df[(df[col] >= min_v) & (df[col] <= max_v)]
                return result

            match = re.search(rf"{col}.*?(?:over|greater than|mehr als|above)\s*(\d+)", q)
            if match:
                min_v = int(match.group(1))
                result = df[df[col] > min_v]
                return result

            match = re.search(rf"{col}.*?(?:under|less than|weniger als|below)\s*(\d+)", q)
            if match:
                max_v = int(match.group(1))
                result = df[df[col] < max_v]
                return result

            match = re.search(rf"{col}\s*(=|is|ist)?\s*(\d+)", q)
            if match:
                value = int(match.group(2))
                result = df[df[col] == value]
                return result

    # === Dynamische Kategorie-Erkennung für Textspalten ===
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in text_cols:
        unique_vals = df[col].dropna().astype(str).str.lower().unique()
        for val in unique_vals:
            if val in q:
                result = df[df[col].astype(str).str.lower().str.contains(val)]
                return result

    return None

def test_loop():
    while True:
        question = input("\n❓ Testfrage für Strukturierte Abfrage (oder 'exit'): ")
        if question.lower() in {"exit", "quit"}:
            break

        result = handle_structured_question(question)
        if result is not None and not result.empty:
            print("\n📊 Ergebnisse:")
            print(result.to_string(index=False))
        else:
            print("❌ Keine passenden Daten gefunden.")

if __name__ == "__main__":
    test_loop()
