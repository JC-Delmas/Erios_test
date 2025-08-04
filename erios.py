#!/usr/bin/env python3
# app.py – Q/R PAC (CPU)

## Code source exécutable depuis Google Colab ##

import os, sys, types, time, re, textwrap, docx, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Installation des dépendances (CPU)
pip install -qU \
  llama-cpp-python \
  sentence-transformers \
  faiss-cpu \
  python-docx \
  tqdm

# Téléchargement du modèle & import du .docx (augmentation du prompt par RAG)
# Modèle quantifié (3,5 Go)
wget -q https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_S.gguf -O /content/mistral.gguf

from google.colab import files
import pathlib

# Import du fichier qui augmente le prompt (RAG)
uploaded = files.upload()
DOC_PATH = pathlib.Path(next(iter(uploaded)))   # on garde le chemin
print("Document chargé :", DOC_PATH)

MODEL  = "/content/mistral.gguf" # on choisi arbitrairement un modèle llm open source français
DOC    = str(DOC_PATH)      #  document renseigné dans la cellule 2
TOP_K  = 4                  # 4 passages pour raccourcir le prompt pour optimiser la vitesse de réponse du llm
N_CTX  = 2048
THREAD = os.cpu_count()

# --- correctif fileno() uniquement pour Colab
for s,fd in ((sys.stdout,1),(sys.stderr,2)):
    s.fileno = types.MethodType(lambda self,fd=fd: fd, s)

# 1) Extraction du texte
paras=[p.text.strip() for p in docx.Document(DOC).paragraphs if p.text.strip()]
chunks,buff=[], ""
for p in paras:
    buff += " " + p
    if len(buff.split()) > 40:
        chunks.append(buff.strip()); buff=""
if buff: chunks.append(buff.strip())
print(f"✅ {len(chunks)} passages extraits.") # on quantifie les passages

# 2) Embeddings + index (pour fonctionnement CPU)
enc = SentenceTransformer("all-MiniLM-L6-v2")
emb = enc.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb.astype("float32"))
print("✅ Index vectoriel prêt.")

# 3) Chargement du modèle CPU
print("Chargement du LLM, cela peut prendre quelques instants ...")
llm = Llama(model_path=MODEL, n_ctx=N_CTX, n_threads=THREAD, n_batch=256, verbose=False)
print(f"✅ Modèle chargé ({THREAD} threads, contexte {N_CTX})")

SYSTEM = ("Tu es un assistant médical. Réponds en français de façon concise, "
          "sans jamais poser de nouvelle question. "
          "Cite tes sources sous la forme [source N] et termine ta réponse "
          "par le symbole <FIN>.")

def repondre(question: str):
    global enc, index, chunks
    # recherche vectorielle
    q_vec = enc.encode([question], normalize_embeddings=True)
    _, idxs = index.search(q_vec.astype("float32"), TOP_K)
    idxs = idxs[0].tolist()

    # filtrage simple par mots-clés extraits de la question
    mots = [w.lower().strip(".,;:!?") for w in question.split() if len(w) > 4] # on garde les mots de plus de quatre lettres, utile pour le nom des médicaments
    nombres = re.findall(r"\d+", question)     # récupère tous les nombres de la question
    idxs = [i for i in idxs
            if any(m in chunks[i].lower() for m in mots) or      # contient un mot clef
               any(n in chunks[i] for n in nombres)]              # ou contient un nombre

    # construction du contexte
    contexte = "\n\n".join(f"(source {i}) {chunks[i]}" for i in idxs)
    prompt = (f"{SYSTEM}\n\nContexte :\n{contexte}\n\n"
              f"Q : {question}\nA : ")

    print("\n▶️ génération :", end=" ", flush=True); t0 = time.time()

    # streaming token-par-token
    out = []
    for chunk in llm(prompt,
                     stream=True,
                     max_tokens=256,
                     temperature=0.1,
                     stop=["<FIN>"]):
        tok = chunk["choices"][0]["text"]
        if tok:
            print(tok, end="", flush=True)
            out.append(tok)     # mémorise le texte

    print(f"\n⏱ {time.time()-t0:.1f} s")

    # on récupère les numéros réellement cités
    answer_text = "".join(out)
    cited = {int(n) for n in re.findall(r"\[source (\d+)", answer_text)} #regex qui permet de récupérer plusieurs sources ex. [source 2, 9]

    print("\n---\nSources :")
    for i in sorted(cited):                          # on n’affiche QUE ceux cités
        if i < len(chunks):                          # sécurité indice
            extrait = textwrap.shorten(chunks[i], width=180, placeholder="…")
            print(f"[source {i}] « {extrait} »")
    print()

# Boucle interactive
while True:
    try:
        q = input("\n❓ Posez votre question médicale (Entrée pour quitter) : ").strip()
        if not q: 
          print("Au revoir.")
          break
        repondre(q)
    except KeyboardInterrupt:
        print("Interruption volontaire du programme.")
        break
