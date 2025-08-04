from pydantic import BaseModel
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
import pandas as pd
import os
import shutil
import requests
from tqdm import tqdm

# === EINSTELLUNGEN ===
REBUILD_INDEX = False  # Nur auf True setzen, wenn sich CSV ändert
csv_path = r"D:\Machine_Learning\pokemon_llm\pokedex_with_text.csv"
index_path = r"D:\Machine_Learning\pokemon_llm\faiss_index"
model_dir = r"D:\Machine_Learning\models"
model_filename = "llama-2-7b-chat.Q4_K_M.gguf"  # Kleinere Modellvariante für 16 GB RAM
model_path = os.path.join(model_dir, model_filename)

# === MODELL-DOWNLOAD ===
def download_model(url, filepath):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024
    with open(filepath, "wb") as file, tqdm(
        desc=f"📥 Lade Modell herunter...",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))
    print(f"\n✅ Modell gespeichert unter: {filepath}")

# Modell ggf. herunterladen
model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"
os.makedirs(model_dir, exist_ok=True)
if not os.path.exists(model_path):
    download_model(model_url, model_path)
else:
    print(f"✔️ Modell existiert bereits: {model_path}")

# === EMBEDDING MODELL ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

# === INDEX ERSTELLEN / LADEN ===
if REBUILD_INDEX:
    print("🔄 Erstelle FAISS-Vektorindex neu...")
    df = pd.read_csv(csv_path, encoding="ISO-8859-1", sep=";")
    df.columns = df.columns.str.strip().str.lower()

    df['full_text'] = df.apply(lambda r: f"{r['name'].lower()}: {r['info']} "
                                         f"{r['name']} is {r['height']} decimeters tall. "
                                         f"{r['name']} weighs {r['weight']} hectograms. "
                                         f"{r['name']} has {r['hp']} HP. "
                                         f"{r['name']} has {r['attack']} attack points. "
                                         f"{r['name']} has {r['defense']} defense points. "
                                         f"{r['name']} has {r['s_attack']} special attack. "
                                         f"{r['name']} has {r['s_defense']} special defense. "
                                         f"{r['name']} has {r['speed']} speed.", axis=1)

    df.to_csv(csv_path, index=False, sep=";", encoding="ISO-8859-1")

    documents = [Document(page_content=row['full_text']) for _, row in df.iterrows()]

    if os.path.exists(index_path):
        shutil.rmtree(index_path)

    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(index_path)
    print("✅ Index neu erstellt.")
else:
    print("📂 Lade bestehenden Vektorindex...")
    vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

# === LLM LAUNCH ===
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.2,
    max_tokens=512,
    n_ctx=4096,
    n_threads=8,         # passe ggf. an deine CPU an
    f16_kv=True,
    use_mlock=True,
    verbose=True
)

# === STATE-KLASSE ===
class State(BaseModel):
    question: str
    context: Optional[List[Document]]
    answer: str

# === RETRIEVE ===
def retrieve(db, state: State, k: int = 6):
    retrieved_docs = db.similarity_search(state.question, k=k)
    return {"context": retrieved_docs}

# === GENERATE ===
def generate(state: State):
    context_text = "\n\n".join(doc.page_content for doc in state.context)
    prompt_text = f"""You are a helpful assistant. Use the following Pokémon information to answer the user's question.

Context:
{context_text}

Question: {state.question}
Answer:"""

    response = llm.invoke(prompt_text)
    return {"answer": response.strip()}

# === INTERAKTIVE FUNKTION ===
def ask(question: str, k: int = 6):
    state = State(question=question, context=None, answer="")
    context = retrieve(vector_store, state, k=k)
    state.context = context["context"]
    generated = generate(state)
    state.answer = generated["answer"]

    # Kontextuelle Pokémon extrahieren
    found_names = []
    for doc in state.context:
        name = doc.page_content.split(":")[0].strip().lower()
        if name not in found_names:
            found_names.append(name)

    # CSV laden und Spalten normalisieren
    df = pd.read_csv(csv_path, encoding="ISO-8859-1", sep=";")
    df.columns = df.columns.str.strip().str.lower()
    df['name'] = df['name'].str.lower()

    # Nur Pokémon anzeigen, die in der Antwort erwähnt werden
    mentioned = [name for name in found_names if name in state.answer.lower()]

    # Ausgabe
    print(f"\n❓ Question:\t{question}")
    print(f"💬 Answer:\t{state.answer}")

    if not mentioned:
        print("\n📘 Keine relevanten Pokémon im Antworttext erkannt.")
        return state

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

    return state

# === INTERAKTIVES CLI ===
if __name__ == "__main__":
    while True:
        user_input = input("\n❓ Deine Frage (oder 'exit'): ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        ask(user_input)
