# SimplyRAG

A small Retrieval-Augmented Generation (RAG) model focused on indexing local files and running a simple chat interface over the indexed content.

## Overview

This repository provides minimal scripts to ingest local files, build a search index, and run a simple chat interface that uses the index to retrieve context for responses.

Key scripts:

- `ingest.py` - prepares and converts source files into documents suitable for indexing.
- `createIndex.py` - builds/persists the searchable index from ingested documents.
- `chat.py` - lightweight chat interface that queries the index and composes responses.

The `data/` directory in this repo contains source files used for ingestion and retrieval.

## Requirements

- Python 3.10+ (3.11 recommended)
- A virtual environment (recommended)

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

If you use an external LLM service, add its client library (for example, `openai`) and configure credentials in an environment file as described below.

## Environment

Copy the example environment file and set any provider keys/configs:

```bash
cp .env-example .env
# edit .env to add API keys and other settings
```
Vars required in the env file:
- PINECONE_API_KEY=ADD_YOUR_PINECONE_API_KEY_HERE
- OPENROUTER_API_KEY=ADD_YOUR_OPENROUTER_API_KEY_HERE

## Usage
1. Prepare environment and dependencies (see Requirements).
2. Build or update the search index:

```bash
python createIndex.py
```

3. Ingest source files into document form:

```bash
python ingest.py
```

4. Run the chat interface:

```bash
python chat.py
```

The scripts are intentionally small and opinionated. If you want to change where files are read from or where the index is persisted, edit the top-level constants in the scripts or provide an `INDEX_DIR` environment variable.

## Project layout

- `chat.py` - chat entrypoint that queries the index and composes responses
- `ingest.py` - converts files from `data/` into indexable documents
- `createIndex.py` - builds and saves the vector/text index
- `data/` - Source Files for Ingestion
- `.env-example` - example environment variables

## Extending / Customizing

- Add or change the embedding model used in `createIndex.py` and `chat.py` to improve semantic search quality.
- Replace the ANN backend (e.g., FAISS) with another library if desired.
- Hook in an LLM provider in `chat.py` to generate responses using retrieved context.

## Troubleshooting

- If ingestion fails, confirm `data/` files are readable and in expected formats.
- If embeddings fail to create, ensure the embedding model and its dependencies are installed and configured.
- For API-key related errors, verify the values in `.env` and restart the environment.

## Contributing

Contributions are welcome. Open an issue to discuss changes, then send a pull request with tests or an explanation of the change.

## License

This repository does not include a license file. Add a license if you plan to publish or share the project.
