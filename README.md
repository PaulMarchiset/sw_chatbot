# SW Chatbot - RAG Star Wars

Application RAG (Retrieval-Augmented Generation) basee sur un corpus Star Wars local.
Le projet utilise FAISS + embeddings HuggingFace pour la recherche de contexte, puis un LLM via OpenRouter pour generer la reponse en francais.

## Fonctionnalites

- Chat web Flask avec interface "Holonet".
- Endpoint API POST /api/chat.
- Recherche semantique sur index FAISS local.
- Reponses contraintes au contexte du corpus.
- Affichage optionnel d'un modele 3D (GLB) si la source correspond a un vaisseau mappe.
- Script CLI pour discuter dans le terminal.
- Scripts pour telecharger/nettoyer un corpus MediaWiki.

## Structure du projet

- web_app.py: application Flask principale (UI + API).
- api/index.py: point d'entree pour deployment serverless (Vercel).
- chatbot.py: version CLI du chatbot RAG.
- create_vectors.py: creation de l'index FAISS a partir du corpus.
- wiki_downloader.py: telechargement de pages depuis un wiki MediaWiki/Fandom.
- cleaner.py: nettoyage des fichiers texte du corpus.
- corpus_starwars/: corpus texte local.
- faiss_starwars_index/: index vectoriel genere.
- static/models/: modeles 3D GLB optionnels.

## Prerequis

- Python 3.10+
- Une cle API OpenRouter
- Acces internet (necessaire pour:
  - installation des dependances,
  - telechargement du modele d'embeddings HuggingFace,
  - appel LLM OpenRouter)

## Installation

1. Cloner le repo puis se placer a la racine.
2. Creer un environnement virtuel.
3. Installer les dependances.

Exemple Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Creer un fichier .env a la racine:

```env
OPENROUTER_API_KEY=ta_cle_openrouter
```

## Preparation des donnees (optionnel mais recommande)

Si tu veux reconstruire ton corpus/index depuis zero:

1. Telecharger des pages wiki (exemples):

```powershell
py .\wiki_downloader.py search "Millennium Falcon" -n 5
py .\wiki_downloader.py page "Millennium Falcon" --format text -o .\corpus_starwars\vehicules\Millennium_Falcon.txt
```

2. Nettoyer les fichiers .txt du corpus:

```powershell
py .\cleaner.py
```

3. Regenerer l'index FAISS:

```powershell
py .\create_vectors.py
```

## Lancer l'application web (local)

```powershell
py .\web_app.py
```

Puis ouvrir:

- http://127.0.0.1:5000

## Utiliser la version CLI

```powershell
py .\chatbot.py
```

Commandes de sortie: quit, exit, quitter

## API

Endpoint local:

- POST /api/chat
- Content-Type: application/json

Exemple de payload:

```json
{
  "question": "Qui pilote le Faucon Millenium ?"
}
```

Exemple de reponse:

```json
{
  "answer": "...",
  "sources": ["vehicules/Millennium_Falcon.txt"],
  "model_url": "/static/models/falcon.glb"
}
```

## Deployment Vercel

Le fichier vercel.json redirige toutes les routes vers api/index.py.
Ce module importe app depuis web_app.py.

Points importants:

- L'index FAISS doit etre present et accessible en production.
- OPENROUTER_API_KEY doit etre configuree dans les variables d'environnement Vercel.
- Le cold start peut etre plus long (chargement embeddings/index).

## Depannage rapide

Si py .\web_app.py renvoie une erreur au demarrage:

- Verifier que .env existe et contient OPENROUTER_API_KEY.
- Verifier que le dossier faiss_starwars_index existe.
- Si l'index est absent/corrompu: relancer py .\create_vectors.py.
- Verifier les dependances: pip install -r requirements.txt.

Erreurs frequentes:

- OPENROUTER_API_KEY est introuvable dans .env
  - Cause: variable absente.
  - Fix: ajouter la variable dans .env.

- Index FAISS introuvable. Lance d'abord create_vectors.py
  - Cause: index non genere.
  - Fix: lancer create_vectors.py.

## Notes

- Le projet est configure pour repondre en francais.
- Le mapping des modeles 3D est dans web_app.py (SHIP_MODELS_MAP).
- Pour enrichir les sources renvoyees, ajoute/ameliores les fichiers dans corpus_starwars puis regenere FAISS.
