# RAG API with ChromaDB + SentenceTransformers + pdfplumber

A lightweight Retrieval-Augmented Generation (RAG) service:
- Ingest PDFs, chunk + embed with `sentence-transformers`
- Store embeddings locally in ChromaDB
- Query similar chunks and optionally generate an answer via OpenAI
- Exposes FastAPI endpoints for `/ingest` and `/query`

Works locally or in Docker. CPU-only by default; CUDA users can customize build.

## Requirements
- Python 3.10+
- Or Docker 24+
- OpenAI API key (optional, for generation): set `OPENAI_API_KEY`

Key files:
- `main.py`: FastAPI app and CLI entry
- `requirements.txt`: dependencies
- `Dockerfile`: production-ready image

## Quick Start (Local)

Install deps and run API:

```
python -m venv .venv
. .venv/bin/activate  # on Windows: .\\.venv\\Scripts\\activate
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Now open Swagger UI at: http://localhost:8000/docs

## Docker Usage

Build image:

```
docker build -t rag-chroma:latest .
```

Run container (CPU):

```
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/chroma_db:/app/chroma_db \
  rag-chroma:latest
```

- Persists Chroma index to `./chroma_db` on host.
- Swagger: http://localhost:8000/docs

### Notes for CUDA Users
This project installs `torch` as listed in `requirements.txt`. If you want a CUDA build, pass the official index URL at build time (adjust for desired CUDA version):

```
docker build --build-arg PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 -t rag-chroma:cu121 .
```

And update the Dockerfile to respect `PIP_EXTRA_INDEX_URL` by exporting it before `pip install` if needed.

## API Endpoints

- POST `/ingest`
  - Form fields:
    - `db` (str, default `./chroma_db`): Chroma persist dir
    - `embed_model` (str, default `sentence-transformers/all-MiniLM-L6-v2`)
    - `files` (file[]): one or more PDF uploads
  - Effect: Extracts text, chunks, embeds, and upserts into Chroma.

- POST `/query`
  - Form fields:
    - `q` (str): query text
    - `db` (str, default `./chroma_db`)
    - `top_k` (int, default 5)
    - `embed_model` (str, default `sentence-transformers/all-MiniLM-L6-v2`)
    - `openai_model` (str, default `gpt-4o-mini`)
  - Returns: `{ answer, contexts[] }` where `answer` uses OpenAI if `OPENAI_API_KEY` is set, otherwise an error message string is returned.

## Example: Ingest via cURL

```
curl -X POST "http://localhost:8000/ingest" \
  -F "db=./chroma_db" \
  -F "embed_model=sentence-transformers/all-MiniLM-L6-v2" \
  -F "files=@/path/to/doc1.pdf" \
  -F "files=@/path/to/doc2.pdf"
```

## Example: Query via cURL

```
curl -X POST "http://localhost:8000/query" \
  -F "q=What does the policy say about refunds?" \
  -F "db=./chroma_db" \
  -F "top_k=5"
```

## Data & Volumes
- Chroma persistence: set `db` to a bind-mounted directory (default `./chroma_db`).
- Temporary uploads go to `./_tmp_uploads` inside the container.

## Troubleshooting
- Empty results: ensure you ingested first and PDFs contain extractable text.
- Torch install size: consider CPU-only `torch==2.x+cpu` if you don’t need CUDA.
- pdfplumber extraction: depends on PDF structure; some files may yield sparse text.
- OpenAI generation issues: verify `OPENAI_API_KEY` and model availability.

## License
This project uses dependencies under their respective licenses. No license is specified for this repo.

