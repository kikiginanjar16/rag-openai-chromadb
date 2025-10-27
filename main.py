import os
from typing import List, Tuple
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

import pdfplumber

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import requests
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def read_pdfs(pdf_dir: str) -> List[Tuple[str, str]]:
    docs = []
    for root, _, files in os.walk(pdf_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                try:
                    text = extract_text_pdf(path)
                    if text.strip():
                        docs.append((path, text))
                except Exception as e:
                    print(f"[warn] Failed to read {path}: {e}")
    return docs


def extract_text_pdf(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                parts.append(txt)
    return "\n\n".join(parts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    text = text.replace("\r", "\n")
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
        i += max(1, chunk_size - overlap)
    return chunks


def get_client(persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.Client(Settings(persist_directory=persist_dir, is_persistent=True))
    return client


def get_collection(client, name: str = "rag"):
    try:
        col = client.get_collection(name)
    except Exception:
        col = client.create_collection(name)
    return col


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


def ingest(pdf_dir: str, db_dir: str, model_name: str):
    client = get_client(db_dir)
    col = get_collection(client)
    embedder = get_embedder(model_name)

    docs = read_pdfs(pdf_dir)
    if not docs:
        print("No PDFs found or extracted text is empty.")
        return

    ids = []
    texts = []
    metadatas = []

    uid = 0
    for path, text in docs:
        chunks = chunk_text(text)
        for c in chunks:
            ids.append(f"doc-{uid}")
            texts.append(c)
            metadatas.append({"source": path})
            uid += 1

    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    # Upsert in manageable batches
    batch = 512
    for i in range(0, len(texts), batch):
        col.upsert(
            ids=ids[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=metadatas[i : i + batch],
            embeddings=embeddings[i : i + batch],
        )
    print(f"Ingested {len(texts)} chunks into Chroma at {db_dir}.")


def retrieve(db_dir: str, query: str, top_k: int, model_name: str):
    client = get_client(db_dir)
    col = get_collection(client)
    embedder = get_embedder(model_name)
    q_emb = embedder.encode([query]).tolist()[0]
    res = col.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    docs = []
    for i in range(len(res.get("ids", [[]])[0])):
        docs.append({
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i].get("source", "unknown"),
            "distance": res["distances"][0][i],
        })
    return docs


def chatgpt_generate(prompt: str, model: str = "gpt-4o-mini") -> str:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY or None)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[Generation error] {e}"


def build_prompt(question: str, contexts: List[dict]) -> str:
    ctx = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in contexts])
    instructions = (
        "You are a helpful assistant. Answer the user's question using the provided context. "
        "Cite sources as (source path) when relevant. If the answer is not contained in the context, say you don't know."
    )
    return f"{instructions}\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer:"


def cmd_ingest(args):
    ingest(args.pdf_dir, args.db, args.embed_model)


def cmd_query(args):
    docs = retrieve(args.db, args.q, args.top_k, args.embed_model)
    if not docs:
        print("No results found. Did you ingest first?")
        return
    prompt = build_prompt(args.q, docs)
    answer = chatgpt_generate(prompt, model=args.openai_model)
    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== Top Contexts ===")
    for i, d in enumerate(docs, 1):
        print(f"\n[{i}] {d['source']} (distance: {d['distance']:.4f})\n{d['text'][:600]}...")


app = FastAPI(title="RAG API (Chroma + pdfplumber + ChatGPT)")


@app.post("/ingest")
async def ingest_endpoint(
    db: str = Form("./chroma_db"),
    embed_model: str = Form("sentence-transformers/all-MiniLM-L6-v2"),
    files: List[UploadFile] = File(description="Upload one or more PDF files"),
):
    temp_dir = "./_tmp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    pdf_dir = os.path.join(temp_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    saved = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, f.filename)
        with open(path, "wb") as out:
            out.write(await f.read())
        saved.append(path)

    if not saved:
        return JSONResponse({"status": "no_pdfs"}, status_code=400)

    ingest(pdf_dir, db, embed_model)
    return {"status": "ok", "pdfs": saved, "db": db}


@app.post("/query")
async def query_endpoint(
    q: str = Form(...),
    db: str = Form("./chroma_db"),
    top_k: int = Form(5),
    embed_model: str = Form("sentence-transformers/all-MiniLM-L6-v2"),
    openai_model: str = Form("gpt-4o-mini"),
):
    docs = retrieve(db, q, top_k, embed_model)
    if not docs:
        return JSONResponse({"status": "empty", "message": "No results. Did you ingest?"}, status_code=404)
    prompt = build_prompt(q, docs)
    answer = chatgpt_generate(prompt, model=openai_model)
    return {"answer": answer, "contexts": docs}


def start():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start()
