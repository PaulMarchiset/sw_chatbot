from dotenv import load_dotenv

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION DE L'IA ---
# Remplace par ta vraie clé OpenRouter
load_dotenv() # Charge les variables d'environnement depuis le fichier .env
CLEF_OPENROUTER = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    api_key=CLEF_OPENROUTER,
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/hunter-alpha",
    temperature=0.3
)

# --- 2. CHARGEMENT DE TA MÉMOIRE (FAISS) ---
print("🧠 Réveil du droïde (Architecture LCEL Nouvelle Génération)...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_starwars_index", embedding_model, allow_dangerous_deserialization=True)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20}
)

# --- 3. LE NOUVEAU PROMPT ---
template = """Tu es un expert absolu de l'univers Star Wars. 
Utilise UNIQUEMENT le contexte fourni ci-dessous pour répondre à la question. 
Si la réponse n'est pas dans le contexte, dis simplement que tes archives sont incomplètes.
Réponds toujours en français.

Contexte extrait des archives :
{context}

Question : {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 4. ASSEMBLAGE LCEL (LE FUTUR 🚀) ---
# Fini les vieilles "chains". On utilise les "pipes" (|) pour faire passer la donnée !

# Petite fonction pour mettre en forme les textes trouvés
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# La chaîne qui génère la réponse
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser() # Ça extrait juste le texte de la réponse de l'IA
)

# La chaîne finale qui garde aussi les sources en mémoire
rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


# --- 5. L'INTERFACE DE DISCUSSION ---
print("\n✅ Système RAG opérationnel ! (Tape 'quit' ou 'exit' pour quitter)")
print("-" * 50)

while True:
    question = input("\n🧑‍🚀 Pose ta question sur les vaisseaux : ")
    
    if question.lower() in ['quit', 'exit', 'quitter']:
        print("Que la Force soit avec toi. Au revoir !")
        break
        
    print("🤖 Recherche hyperespace en cours...")
    
    # On invoque notre nouvelle chaîne LCEL
    resultat = rag_chain_with_source.invoke(question)
    
    print("\n" + "="*50)
    print(resultat["answer"])
    print("="*50)
    
    # On récupère les sources proprement
    print("\n(Sources consultées : ", end="")
    sources = set([doc.metadata.get('source', 'Inconnue') for doc in resultat["context"]])
    print(", ".join(sources) + ")")