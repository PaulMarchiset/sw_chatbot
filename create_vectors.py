import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DOSSIER_CORPUS = "starships_starwars" 

print("🚀 ÉTAPE 1 : Chargement des documents...")
documents = []
# On parcourt tous les fichiers .txt du dossier
for filepath in Path(DOSSIER_CORPUS).glob("*.txt"):
    # On lit le texte (avec utf-8 pour éviter les bugs d'accents)
    texte = filepath.read_text(encoding="utf-8", errors="ignore")
    # On crée un "Document" LangChain en gardant le nom du fichier en métadonnée
    doc = Document(page_content=texte, metadata={"source": filepath.name})
    documents.append(doc)
    
print(f"✅ {len(documents)} vaisseaux chargés !")

print("\n✂️ ÉTAPE 2 : Découpage en chunks...")
# On configure le découpeur (~500 caractères, avec un léger chevauchement)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50, # Le chevauchement évite de couper une phrase en plein milieu
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Corpus découpé en {len(chunks)} morceaux (chunks).")

print("\n🧠 ÉTAPE 3 & 4 : Vectorisation et création de la base FAISS...")
# On télécharge et prépare le modèle d'embedding (ça peut prendre 1 ou 2 min la première fois)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# On transforme nos chunks en vecteurs et on les stocke dans FAISS
vectorstore = FAISS.from_documents(chunks, embedding_model)

# IMPORTANT : On sauvegarde la base de données sur le disque !
# Comme ça, pas besoin de tout recalculer la prochaine fois.
dossier_faiss = "faiss_starwars_index"
vectorstore.save_local(dossier_faiss)
print(f"✅ Base de données sauvegardée dans le dossier '{dossier_faiss}'.")

# --- TEST BONUS ---
print("\n🔍 Test de la base de données :")
question = "What is the fastest starship in the Star Wars universe?"
print(f"Question : {question}")

# On demande à FAISS de trouver les 2 morceaux de texte les plus pertinents
docs_trouves = vectorstore.similarity_search(question, k=2)

for i, doc in enumerate(docs_trouves):
    print(f"\n--- Résultat {i+1} (Source: {doc.metadata['source']}) ---")
    print(doc.page_content)