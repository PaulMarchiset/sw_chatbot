import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Holonet RAG - Star Wars</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,700&display=swap" rel="stylesheet">
  <style>
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
      margin: 0;
      min-height: 100vh;
      font-family: "Space Grotesk", sans-serif;
      color: var(--text);
      background:
        radial-gradient(1200px 600px at 85% -10%, rgba(107, 220, 255, 0.22), transparent 65%),
        radial-gradient(1000px 500px at -20% 100%, rgba(255, 215, 122, 0.2), transparent 70%),
        linear-gradient(160deg, var(--bg-1), var(--bg-2));
      overflow-x: hidden;
    }

    .stars,
    .stars:before,
    .stars:after {
      position: fixed;
      inset: 0;
      content: "";
      pointer-events: none;
      background-image:
        radial-gradient(2px 2px at 20% 30%, rgba(255,255,255,0.85), transparent),
        radial-gradient(1px 1px at 60% 80%, rgba(255,255,255,0.6), transparent),
        radial-gradient(1px 1px at 80% 10%, rgba(255,255,255,0.5), transparent),
        radial-gradient(2px 2px at 35% 60%, rgba(255,255,255,0.7), transparent),
        radial-gradient(1px 1px at 70% 35%, rgba(255,255,255,0.65), transparent);
      animation: drift 18s linear infinite;
      opacity: 0.6;
    }

    .stars:before { animation-duration: 26s; opacity: 0.35; }
    .stars:after { animation-duration: 34s; opacity: 0.2; }

    @keyframes drift {
      from { transform: translateY(0); }
      to { transform: translateY(-25px); }
    }

    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 28px 16px 24px;
      position: relative;
      z-index: 1;
    }

    .hero {
      margin-bottom: 18px;
      padding: 24px;
      border: 1px solid var(--stroke);
      border-radius: 22px;
      background: var(--card);
      backdrop-filter: blur(12px);
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
      animation: reveal 450ms ease-out;
    }

    .tag {
      display: inline-flex;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255, 215, 122, 0.45);
      color: var(--accent-2);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    h1 {
      font-family: "Fraunces", serif;
      font-size: clamp(1.6rem, 2.8vw, 2.5rem);
      margin: 10px 0 8px;
      line-height: 1.2;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      max-width: 70ch;
      line-height: 1.45;
    }

    .chat-card {
      border: 1px solid var(--stroke);
      border-radius: 22px;
      background: var(--card);
      backdrop-filter: blur(12px);
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
      overflow: hidden;
      animation: reveal 700ms ease-out;
    }

    @keyframes reveal {
      from {
        transform: translateY(8px);
        opacity: 0;
      }
      to {
        transform: translateY(0);
        opacity: 1;
      }
    }

    .chat-log {
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .bubble {
      max-width: 88%;
      border-radius: 16px;
      padding: 12px 14px;
      line-height: 1.45;
      white-space: pre-wrap;
      border: 1px solid transparent;
      animation: reveal 260ms ease-out;
    }

    .user {
      align-self: flex-end;
      background: linear-gradient(145deg, rgba(107, 220, 255, 0.25), rgba(107, 220, 255, 0.1));
      border-color: rgba(107, 220, 255, 0.35);
    }

    .assistant {
      align-self: flex-start;
      background: rgba(255, 255, 255, 0.06);
      border-color: rgba(255, 255, 255, 0.18);
    }

    .assistant h1,
    .assistant h2,
    .assistant h3,
    .assistant h4 {
      margin: 0.2em 0 0.45em;
      line-height: 1.25;
      font-family: "Fraunces", serif;
    }

    .assistant p {
      margin: 0.35em 0;
    }

    .assistant ul,
    .assistant ol {
      margin: 0.35em 0;
      padding-left: 1.2em;
    }

    .assistant code {
      font-family: Consolas, "Courier New", monospace;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 6px;
      padding: 0.05em 0.35em;
      font-size: 0.93em;
    }

    .assistant pre {
      overflow-x: auto;
      margin: 0.5em 0;
      padding: 0.75em;
      border-radius: 10px;
      background: rgba(6, 9, 14, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.15);
    }

    .assistant pre code {
      background: transparent;
      border: 0;
      padding: 0;
    }

    .assistant blockquote {
      margin: 0.5em 0;
      padding: 0.3em 0.8em;
      border-left: 3px solid rgba(107, 220, 255, 0.75);
      color: var(--muted);
      background: rgba(107, 220, 255, 0.08);
      border-radius: 0 8px 8px 0;
    }

    .assistant a {
      color: var(--accent);
      text-decoration: underline;
      text-underline-offset: 2px;
    }

    .sources {
      margin-top: 8px;
      font-size: 0.85rem;
      color: var(--muted);
    }

    .composer {
      border-top: 1px solid var(--stroke);
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      background: rgba(5, 11, 22, 0.65);
    }

    textarea {
      width: 100%;
      min-height: 56px;
      max-height: 160px;
      resize: vertical;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      font-family: inherit;
      padding: 12px;
      font-size: 0.96rem;
      outline: none;
    }

    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(107, 220, 255, 0.25);
    }

    button {
      border: 0;
      border-radius: 12px;
      padding: 0 18px;
      background: linear-gradient(145deg, var(--accent), #74a9ff);
      color: #02131d;
      font-weight: 700;
      cursor: pointer;
      transition: transform 130ms ease, filter 130ms ease;
    }

    button:hover { filter: brightness(1.06); }
    button:active { transform: translateY(1px); }
    button:disabled {
      filter: grayscale(0.35);
      cursor: not-allowed;
      opacity: 0.7;
    }

    .status {
      margin-top: 8px;
      font-size: 0.9rem;
      color: var(--muted);
      min-height: 1.3em;
    }

    .status.error { color: var(--error); }
    .status.ok { color: var(--ok); }

    @media (max-width: 680px) {
      .wrap { padding: 16px 10px 14px; }
      .hero { padding: 16px; }
      .chat-log { padding: 12px; }
      .composer {
        grid-template-columns: 1fr;
      }
      button {
        height: 44px;
      }
      .bubble { max-width: 100%; }
    }
  </style>
</head>
<body>
  <div class="stars"></div>
  <main class="wrap">
    <section class="hero">
      <span class="tag">Holonet Console</span>
      <h1>Interroge tes archives Star Wars</h1>
      <p class="subtitle">
        Pose une question sur ton corpus local. Le bot répond en se basant uniquement sur l'index FAISS, puis affiche les sources consultées.
      </p>
    </section>

    <section class="chat-card">
      <div id="chatLog" class="chat-log"></div>
      <form id="chatForm" class="composer">
        <textarea id="question" placeholder="Ex: Quel est le rôle d'un Acclamator-class assault ship ?" required></textarea>
        <button id="sendBtn" type="submit">Envoyer</button>
      </form>
    </section>

    <div id="status" class="status"></div>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.min.js"></script>
  <script>
    marked.setOptions({
      breaks: true,
      gfm: true
    });

    const chatLog = document.getElementById("chatLog");
    const chatForm = document.getElementById("chatForm");
    const questionInput = document.getElementById("question");
    const sendBtn = document.getElementById("sendBtn");
    const statusEl = document.getElementById("status");

    function addMessage(role, text, sources = []) {
      const bubble = document.createElement("div");
      bubble.className = `bubble ${role}`;

      if (role === "assistant") {
        const rawHtml = marked.parse(text || "");
        bubble.innerHTML = DOMPurify.sanitize(rawHtml);
      } else {
        bubble.textContent = text;
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

    addMessage(
      "assistant",
      "Bienvenue dans Holonet. Pose une question sur les vaisseaux ou les planètes de tes archives."
    );

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

        addMessage("assistant", payload.answer, payload.sources || []);
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

    index_path = Path("faiss_starwars_index")
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
        return jsonify({"answer": result["answer"], "sources": sources})
    except Exception as exc:
        return jsonify({"error": f"Erreur pendant la génération: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
