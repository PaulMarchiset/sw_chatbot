import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

app = Flask(__name__)

# --- MAPPING 3D ---
# Lie le nom de tes fichiers textes sources au chemin de tes modèles 3D dans le dossier 'static'
SHIP_MODELS_MAP = {
    "A-wing_starfighter.txt": "/static/models/a-wing.glb",
    "ARC-170_starfighter.txt": "/static/models/arc-170_fighter.glb",
    "Death_Star.txt": "/static/models/death-star.glb",
    "Imperial_shuttle.txt": "/static/models/imperial_lambda-class_shuttle.glb",
    "Star_Destroyer.txt": "/static/models/imperial_ii_star_destroyer.glb",
    "Millennium_Falcon.txt": "/static/models/millennium_falcon.glb",
    "Mon_Calamari_Star_Cruiser.txt": "/static/models/mon_calamari.glb",
    "Naboo_N-1_starfighter.txt": "/static/models/naboo_starfighter.glb",
    "Slave_I.txt": "/static/models/slave1.glb",
    "Razor_Crest.txt": "/static/models/razor_crest.glb",
    "TIE_D_Defender.txt": "/static/models/tied_defender.glb",
    "TIE_interceptor.txt": "/static/models/interceptor_tie.glb",
    "TIE_bomber.txt": "/static/models/tie_bomber.glb",
    "TIE_LN_starfighter.txt": "/static/models/tie.glb",
    "X-wing_starfighter.txt": "/static/models/xwing.glb",
}

TEMPLATE = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Holonet RAG - Star Wars</title>
  <link rel="icon" href="/favicon.ico" type="image/x-icon" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,700&display=swap" rel="stylesheet">
  <style>
    /* ... [GARDE TOUT TON CSS EXISTANT ICI] ... */
    :root {
      --bg-1: #081120;
      --bg-2: #0d1a2f;
      --card: rgba(10, 16, 32, 0.75);
      --stroke: rgba(120, 180, 255, 0.24);
      --text: #f6f8ff;
      --muted: #a9b4d0;
      --accent: #6bdcff;
      --accent-2: #ffd77a;
      --error: #ff7e86;
      --ok: #89ffbe;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh; font-family: "Space Grotesk", sans-serif;
      color: var(--text); overflow-x: hidden;
      background: radial-gradient(1200px 600px at 85% -10%, rgba(107, 220, 255, 0.22), transparent 65%),
                  radial-gradient(1000px 500px at -20% 100%, rgba(255, 215, 122, 0.2), transparent 70%),
                  linear-gradient(160deg, var(--bg-1), var(--bg-2));
    }
    .stars, .stars:before, .stars:after {
      position: fixed; inset: 0; content: ""; pointer-events: none; opacity: 0.6;
      background-image: radial-gradient(2px 2px at 20% 30%, rgba(255,255,255,0.85), transparent),
                        radial-gradient(1px 1px at 60% 80%, rgba(255,255,255,0.6), transparent),
                        radial-gradient(1px 1px at 80% 10%, rgba(255,255,255,0.5), transparent),
                        radial-gradient(2px 2px at 35% 60%, rgba(255,255,255,0.7), transparent),
                        radial-gradient(1px 1px at 70% 35%, rgba(255,255,255,0.65), transparent);
      animation: drift 18s linear infinite;
    }
    .stars:before { animation-duration: 26s; opacity: 0.35; }
    .stars:after { animation-duration: 34s; opacity: 0.2; }
    @keyframes drift { from { transform: translateY(0); } to { transform: translateY(-25px); } }
    .wrap { max-width: 980px; margin: 0 auto; padding: 28px 16px 24px; position: relative; z-index: 1; }
    .hero { margin-bottom: 18px; padding: 24px; border: 1px solid var(--stroke); border-radius: 22px; background: var(--card); backdrop-filter: blur(12px); box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35); animation: reveal 450ms ease-out; }
    .tag { display: inline-flex; padding: 6px 10px; border-radius: 999px; border: 1px solid rgba(255, 215, 122, 0.45); color: var(--accent-2); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; }
    h1 { font-family: "Fraunces", serif; font-size: clamp(1.6rem, 2.8vw, 2.5rem); margin: 10px 0 8px; line-height: 1.2; }
    .subtitle { margin: 0; color: var(--muted); max-width: 70ch; line-height: 1.45; }
    .chat-card { border: 1px solid var(--stroke); border-radius: 22px; background: var(--card); backdrop-filter: blur(12px); box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35); overflow: hidden; animation: reveal 700ms ease-out; }
    @keyframes reveal { from { transform: translateY(8px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    .chat-log { padding: 18px; display: flex; flex-direction: column; gap: 14px; }
    .bubble { max-width: 88%; border-radius: 16px; padding: 12px 14px; line-height: 1.45; white-space: pre-wrap; border: 1px solid transparent; animation: reveal 260ms ease-out; }
    .user { align-self: flex-end; background: linear-gradient(145deg, rgba(107, 220, 255, 0.25), rgba(107, 220, 255, 0.1)); border-color: rgba(107, 220, 255, 0.35); }
    .assistant { align-self: flex-start; background: rgba(255, 255, 255, 0.06); border-color: rgba(255, 255, 255, 0.18); width: 100%;}
    
    /* CSS POUR LE CANVAS 3D */
    .model-viewer {
      width: 100%;
      height: 300px;
      margin-top: 15px;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid rgba(107, 220, 255, 0.2);
      background: radial-gradient(circle at center, #112240 0%, #0a1120 100%);
      position: relative;
    }
    .model-loading {
        position: absolute;
        top: 50%; left: 50%; transform: translate(-50%, -50%);
        color: var(--accent); font-size: 0.9em; font-weight: bold;
        pointer-events: none;
    }

    /* Suite de ton CSS... */
    .assistant h1, .assistant h2, .assistant h3, .assistant h4 { margin: 0.2em 0 0.45em; line-height: 1.25; font-family: "Fraunces", serif; }
    .assistant p { margin: 0.35em 0; }
    .sources { margin-top: 8px; font-size: 0.85rem; color: var(--muted); border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;}
    .composer { border-top: 1px solid var(--stroke); padding: 12px; display: grid; grid-template-columns: 1fr auto; gap: 10px; background: rgba(5, 11, 22, 0.65); }
    textarea { width: 100%; min-height: 56px; max-height: 160px; resize: vertical; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.18); background: rgba(255, 255, 255, 0.04); color: var(--text); font-family: inherit; padding: 12px; font-size: 0.96rem; outline: none; }
    textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(107, 220, 255, 0.25); }
    button { border: 0; border-radius: 12px; padding: 0 18px; background: linear-gradient(145deg, var(--accent), #74a9ff); color: #02131d; font-weight: 700; cursor: pointer; transition: transform 130ms ease, filter 130ms ease; }
    button:hover { filter: brightness(1.06); }
    button:active { transform: translateY(1px); }
    button:disabled { filter: grayscale(0.35); cursor: not-allowed; opacity: 0.7; }
    .status { margin-top: 8px; font-size: 0.9rem; color: var(--muted); min-height: 1.3em; }
    .status.error { color: var(--error); }
    .status.ok { color: var(--ok); }
  </style>

  <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
      }
    }
  </script>
</head>
<body>
  <div class="stars"></div>
  <main class="wrap">
    <section class="hero">
      <span class="tag">Holonet Console</span>
      <h1>Interroge tes archives Star Wars</h1>
      <p class="subtitle">Pose une question sur ton corpus local. Le bot répond en se basant uniquement sur l'index FAISS, puis affiche le modèle 3D du vaisseau s'il existe.</p>
    </section>

    <section class="chat-card">
      <div id="chatLog" class="chat-log"></div>
      <form id="chatForm" class="composer">
        <textarea id="question" placeholder="Ex: Qui pilote le Faucon Millenium ?" required></textarea>
        <button id="sendBtn" type="submit">Envoyer</button>
      </form>
    </section>

    <div id="status" class="status"></div>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.min.js"></script>
  
  <script type="module">
    import * as THREE from 'three';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    marked.setOptions({ breaks: true, gfm: true });

    const chatLog = document.getElementById("chatLog");
    const chatForm = document.getElementById("chatForm");
    const questionInput = document.getElementById("question");
    const sendBtn = document.getElementById("sendBtn");
    const statusEl = document.getElementById("status");

    function init3DViewer(container, modelUrl) {
        // 1. Initialisation de la scène
        const scene = new THREE.Scene();
        
        // 2. Caméra
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(0, 2, 5);

        // 3. Rendu
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // 4. Lumières (Ambiance + Directionnelle spatiale)
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const dirLight = new THREE.DirectionalLight(0x6bdcff, 2);
        dirLight.position.set(5, 5, 5);
        scene.add(dirLight);

        const dirLight2 = new THREE.DirectionalLight(0xffd77a, 1);
        dirLight2.position.set(-5, -5, -5);
        scene.add(dirLight2);

        // 5. Contrôles (Permet de tourner autour avec la souris)
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true; // Fait tourner le vaisseau lentement
        controls.autoRotateSpeed = 2.0;

        // Message de chargement
        const loadingDiv = document.createElement('div');
        loadingDiv.className = "model-loading";
        loadingDiv.textContent = "Téléchargement du plan holocron...";
        container.appendChild(loadingDiv);

        // 6. Chargement du modèle GLTF/GLB
        const loader = new GLTFLoader();
        loader.load(
            modelUrl,
            function (gltf) {
                loadingDiv.remove(); // Enlève le message
                
                // Centrage automatique (les modèles du net ont souvent des tailles aléatoires)
                const box = new THREE.Box3().setFromObject(gltf.scene);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 3 / maxDim; // Force le vaisseau à rentrer dans une boite de taille 3
                
                gltf.scene.scale.set(scale, scale, scale);
                gltf.scene.position.sub(center.multiplyScalar(scale)); 
                
                scene.add(gltf.scene);
            },
            undefined, // Callback de progression (optionnel)
            function (error) {
                console.error("Erreur de chargement 3D :", error);
                loadingDiv.textContent = "Archive 3D corrompue.";
            }
        );

        // 7. Boucle d'animation
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // Gestion du redimensionnement de la fenêtre
        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
    }

    function addMessage(role, text, sources = [], modelUrl = null) {
      const bubble = document.createElement("div");
      bubble.className = `bubble ${role}`;

      if (role === "assistant") {
        const rawHtml = marked.parse(text || "");
        bubble.innerHTML = DOMPurify.sanitize(rawHtml);
      } else {
        bubble.textContent = text;
      }

      // Si un modèle 3D a été renvoyé par le backend, on l'injecte !
      if (modelUrl) {
          const viewerDiv = document.createElement("div");
          viewerDiv.className = "model-viewer";
          bubble.appendChild(viewerDiv);
          // Initialise la 3D dans ce nouveau conteneur
          setTimeout(() => init3DViewer(viewerDiv, modelUrl), 100); 
      }

      if (role === "assistant" && sources.length > 0) {
        const src = document.createElement("div");
        src.className = "sources";
        src.textContent = "Sources: " + sources.join(", ");
        bubble.appendChild(src);
      }

      chatLog.appendChild(bubble);
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    }

    function setStatus(message, kind = "") {
      statusEl.className = "status" + (kind ? ` ${kind}` : "");
      statusEl.textContent = message;
    }

    addMessage("assistant", "Bienvenue dans Holonet. Pose une question sur les vaisseaux ou les planètes de tes archives.");

    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = questionInput.value.trim();
      if (!question) return;

      addMessage("user", question);
      questionInput.value = "";
      sendBtn.disabled = true;
      setStatus("Recherche en hyperspace...", "");

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question })
        });

        const payload = await response.json();

        if (!response.ok) {
          throw new Error(payload.error || "Erreur inconnue");
        }

        // On passe maintenant payload.model_url à la fonction
        addMessage("assistant", payload.answer, payload.sources || [], payload.model_url);
        setStatus("Réponse prête.", "ok");
      } catch (error) {
        addMessage("assistant", "Impossible de générer une réponse pour le moment.");
        setStatus(error.message, "error");
      } finally {
        sendBtn.disabled = false;
        questionInput.focus();
      }
    });
  </script>
</body>
</html>
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY est introuvable dans .env")

    index_path = Path(__file__).resolve().parent / "faiss_starwars_index"
    if not index_path.exists():
        raise RuntimeError("Index FAISS introuvable. Lance d'abord create_vectors.py")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/hunter-alpha",
        temperature=0.3,
    )

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        str(index_path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """Tu es un expert absolu de l'univers Star Wars.
Utilise UNIQUEMENT le contexte fourni ci-dessous pour répondre à la question.
Si la réponse n'est pas dans le contexte, dis simplement que tes archives sont incomplètes.
Réponds toujours en français.

Contexte extrait des archives :
{context}

Question : {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source

try:
    rag_chain = build_rag_chain()
    startup_error = None
except Exception as exc:
    rag_chain = None
    startup_error = str(exc)

@app.route("/", methods=["GET"])
def index():
    return render_template_string(TEMPLATE)


@app.route("/favicon.ico", methods=["GET"])
def favicon():
  assets_dir = Path(__file__).resolve().parent / "assets"
  return send_from_directory(assets_dir, "favicon.ico", mimetype="image/vnd.microsoft.icon")

@app.route("/api/chat", methods=["POST"])
def chat():
    if startup_error:
        return jsonify({"error": startup_error}), 500

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({"error": "La question est vide."}), 400

    try:
        result = rag_chain.invoke(question)
        sources = sorted({doc.metadata.get("source", "Inconnue") for doc in result["context"]})
        
        # --- NOUVEAUTÉ : Détection du modèle 3D ---
        # On vérifie si un des fichiers sources utilisés pour la réponse correspond
        # à un vaisseau dont on possède le modèle 3D.
        model_url = None
        for source in sources:
          source_basename = Path(source).name
          if source in SHIP_MODELS_MAP:
            model_url = SHIP_MODELS_MAP[source]
            break 
          if source_basename in SHIP_MODELS_MAP:
            model_url = SHIP_MODELS_MAP[source_basename]
            break

        return jsonify({
            "answer": result["answer"], 
            "sources": sources,
            "model_url": model_url  # On renvoie l'URL au Frontend !
        })
        
    except Exception as exc:
        return jsonify({"error": f"Erreur pendant la génération: {exc}"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)