# === FILE: main.py ===
from test_structur import handle_structured_question
from rag_module import ask_rag, last_answer
import pandas as pd
import re
from deep_translator import GoogleTranslator

csv_path = r"D:\Machine_Learning\pokemon_llm\pokedex_with_text.csv"
df = pd.read_csv(csv_path, encoding="ISO-8859-1", sep=";")
df.columns = df.columns.str.strip().str.lower()
df['name'] = df['name'].str.lower()


def is_structured(question: str) -> bool:
    structured_keywords = [
        "between", "over", "under", "greater", "less", "min", "max",
        "highest", "lowest", "most", "least", "id", "hp", "attack", "defense",
        "s_attack", "s_defense", "speed", "weight", "height"
    ]
    return any(word in question.lower() for word in structured_keywords)


def display_pokemon_info(mentioned: set):
    print("\n📘 Matching Pokémon Info:")
    for name in mentioned:
        match = df[df['name'] == name]
        if not match.empty:
            row = match.iloc[0]
            print(f"\n➡️ {row['name'].title()} (ID: {row['id']})")
            print(f"Info: {row['info']}")
            print(f"Height: {row['height']} dm, Weight: {row['weight']} hg")
            print(f"HP: {row['hp']}, Attack: {row['attack']}, Defense: {row['defense']}")
            print(f"Sp. Attack: {row['s_attack']}, Sp. Defense: {row['s_defense']}, Speed: {row['speed']}")


def show_additional_info(answer: str):
    mentioned = set()
    answer_lc = f" {answer.lower()} "

    for name in df['name']:
        if f" {name} " in answer_lc:
            mentioned.add(name)

    if not mentioned:
        print("\n📘 Keine zusätzlichen Pokémon-Daten gefunden.")
        return

    display_pokemon_info(mentioned)


def main():
    while True:
        user_input = input("\n❓ Deine Frage (oder 'exit'): ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        translated = GoogleTranslator(source='auto', target='en').translate(user_input)
        print(f"🌐 Übersetzt: {translated}")

        if is_structured(translated):
            result = handle_structured_question(translated)
            if result is not None and not result.empty:
                print("\n📊 Strukturierte Antwort:")
                print(result.to_string(index=False))
            else:
                print("❌ Keine passenden Daten gefunden.")
        else:
            print("\n🤖 RAG-Antwort:")
            ask_rag(translated)


if __name__ == "__main__":
    main()