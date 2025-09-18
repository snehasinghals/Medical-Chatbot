# create_memory_with_llm.py
import os
import traceback
import requests
from dotenv import load_dotenv, find_dotenv

# LangChain / community imports
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    # if import fails, we'll still try to proceed (embeddings required for FAISS loader)
    HuggingFaceEmbeddings = None

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    raise RuntimeError("langchain_community.vectorstores.FAISS import failed. Install langchain-community.")

# load .env if present
load_dotenv(find_dotenv())

# ---------------- CONFIG ----------------
# Token environment names this script checks (use one)
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")
if not HF_TOKEN:
    print("Warning: No Hugging Face token found in environment (HUGGINGFACEHUB_API_TOKEN or HF_TOKEN).")
    print("If you want model-generated answers you should set it; the script will still return retrieved passages.\n")

# FAISS DB path and embedding model (must match what you used to build the DB)
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Which HF model to call for generation (recommended: text-generation capable model)
HF_GEN_MODEL = "tiiuae/falcon-7b-instruct"  # recommended free option
# ----------------------------------------

# ---------- Helpers ----------

def load_embedding_model():
    if HuggingFaceEmbeddings is None:
        raise RuntimeError("HuggingFaceEmbeddings not available. Install langchain-huggingface or ensure imports are correct.")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def robust_load_faiss(path: str, embedding_model):
    """
    Try several signatures for FAISS.load_local and ensure db.embedding_function is callable.
    """
    db = None
    load_errors = []
    try:
        # common signature
        db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e1:
        load_errors.append(("positional", e1))
        try:
            db = FAISS.load_local(path, embeddings=embedding_model, allow_dangerous_deserialization=True)
        except Exception as e2:
            load_errors.append(("embeddings=", e2))
            try:
                db = FAISS.load_local(path, embedding_function=embedding_model, allow_dangerous_deserialization=True)
            except Exception as e3:
                load_errors.append(("embedding_function=", e3))
                try:
                    db = FAISS.load_local(path, allow_dangerous_deserialization=True)
                except Exception as e4:
                    load_errors.append(("no-arg", e4))
                    # Raise combined error with traces
                    msg = "Failed to load FAISS DB. Attempts:\n"
                    for tag, ex in load_errors:
                        msg += f" - {tag}: {repr(ex)}\n"
                    raise RuntimeError(msg)

    # Ensure there's a callable embedding function attached to db
    if getattr(db, "embedding_function", None) is None:
        # prefer embed_query
        if hasattr(embedding_model, "embed_query"):
            db.embedding_function = embedding_model.embed_query
        elif hasattr(embedding_model, "embed_documents"):
            db.embedding_function = lambda text: embedding_model.embed_documents([text])[0]
        elif callable(embedding_model):
            db.embedding_function = embedding_model
        else:
            raise RuntimeError("Could not attach an embedding function to FAISS DB -- embedding model lacks expected methods.")

    # Some versions expect db.embeddings attribute
    try:
        db.embeddings = embedding_model
    except Exception:
        pass

    return db

def hf_generate_via_requests(model: str, api_token: str, prompt: str,
                             max_new_tokens: int = 256, temperature: float = 0.2, timeout: int = 60):
    """
    Simple wrapper to call the Hugging Face Inference REST API directly using requests.
    Returns generated text string or raises a RuntimeError.
    """
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature},
        # "options": {"use_cache": False}  # optional
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Hugging Face API returned {resp.status_code}: {resp.text}")
    try:
        parsed = resp.json()
    except Exception:
        raise RuntimeError(f"Unable to parse Hugging Face response: {resp.text}")

    # Typical output: list of dict(s) with 'generated_text'
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "generated_text" in parsed[0]:
        return parsed[0]["generated_text"]
    # Some endpoints return dict with 'generated_text'
    if isinstance(parsed, dict) and "generated_text" in parsed:
        return parsed["generated_text"]
    # fallback: return stringified response
    return str(parsed)

def is_medical_query(q: str) -> bool:
    """Simple keyword check for medical queries to avoid producing treatment instructions."""
    ql = q.lower()
    keywords = ["cancer", "treatment", "cure", "diagnosis", "chemotherapy", "radiation", "surgery", "tumor"]
    return any(k in ql for k in keywords)

# ---------- Main ----------

if __name__ == "__main__":
    try:
        embedding_model = load_embedding_model()
    except Exception as e:
        print("Error loading embedding model:", repr(e))
        raise

    try:
        db = robust_load_faiss(DB_FAISS_PATH, embedding_model)
    except Exception as e:
        print("Error loading FAISS DB:", repr(e))
        raise

    print("Ready. Type a question and press Enter. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            q = input("\nWrite Query Here: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        try:
            # Get top-k documents (k=3)
            try:
                docs = db.similarity_search(q, k=3)
            except TypeError:
                # some versions use similarity_search_with_score
                docs = [d for d, _score in db.similarity_search_with_score(q, k=3)]

            if not docs:
                print("No relevant documents found in FAISS DB.")
                continue

            # Build a context string from top docs (trim each doc to avoid huge prompts)
            snippets = []
            for d in docs:
                text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
                text = text.strip()
                # take first 1000 chars of each doc for context
                snippets.append(text[:1000])

            context_text = "\n\n---\n\n".join(snippets)

            # Safety: if it's a medical query, do NOT produce medical treatment instructions.
            if is_medical_query(q):
                print("\n⚠️ This appears to be a medical question. I will NOT provide medical treatment advice.")
                print("I will show the top retrieved passages from your documents instead; consult a medical professional for care.\n")
                for i, s in enumerate(snippets, 1):
                    print(f"Source {i} (snippet):\n{s}\n{'-'*40}")
                continue

            # If we have an HF token, try to generate an answer via HF Inference API (requests)
            answer = None
            if HF_TOKEN:
                prompt = (
                    "Use only the information provided in the context to answer the question.\n"
                    "If the answer isn't present in the context, say 'I don't know'. Do not hallucinate.\n\n"
                    f"Context:\n{context_text}\n\nQuestion: {q}\n\nAnswer:"
                )
                try:
                    answer = hf_generate_via_requests(HF_GEN_MODEL, HF_TOKEN, prompt,
                                                     max_new_tokens=256, temperature=0.2)
                except Exception as hf_err:
                    # If HF API call fails, show traceback and fall back to displaying sources
                    print("Hugging Face model call failed. Falling back to returning retrieved passages.")
                    print("HF call error:", repr(hf_err))
                    # print detailed trace for debugging
                    traceback.print_exc()
                    answer = None

            # If no answer generated, return the retrieved passages
            if not answer:
                print("\nNo generated answer (HF model not used or failed). Showing retrieved passages instead:\n")
                for i, s in enumerate(snippets, 1):
                    print(f"Source {i} (snippet):\n{s}\n{'-'*40}")
            else:
                print("\nGenerated Answer:\n")
                print(answer)
                print("\n\nTop source snippets used:")
                for i, s in enumerate(snippets, 1):
                    print(f"{i}. {s[:400]}{'...' if len(s)>400 else ''}")
        except Exception as err_outer:
            print("Error:", repr(err_outer))
            traceback.print_exc()
            # Do not crash — continue the loop
            continue
