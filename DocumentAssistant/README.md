Smart Document Assistant - Minimal Prototype

This is a small prototype that ingests documents, indexes them with FAISS, and allows question-answering using Google Gemini via the LangChain `langchain-google-genai` integration.

Prerequisites

- Python 3.10+ installed (recommended)
- A Google Cloud API key with access to the Generative AI models (set as environment variable `GOOGLE_API_KEY` or use Application Default Credentials)

Quick local setup (Windows PowerShell)

1. Create a virtual environment (use `venv` directory):

```powershell
py -3 -m venv venv
```

2. Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation due to execution policy, allow in the current session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Create a `.env` file (copy from `.env.example` if present) and set your keys:

```
GOOGLE_API_KEY=your-google-api-key
GEMINI_MODEL=gemini-2.0-flash
```

Make sure `.env` is not committed to source control.

Run the app (Streamlit)

```powershell
streamlit run app.py
```

Notes & troubleshooting

- This project uses `sentence-transformers` for embeddings and `faiss-cpu` for a local vector store by default.
- The app uses the `langchain-google-genai` package to talk to Google Gemini. Make sure the package is installed in your active venv.
- If you see import errors for `dotenv`, install `python-dotenv` in your venv: `pip install python-dotenv`.
- If you see errors about model names, set `GEMINI_MODEL` in your `.env` to a supported model (for example `gemini-2.0-flash` or `gemini-pro`).

Extending

- Swap FAISS for Pinecone/Weaviate by replacing `src/vectorstore.py` and adjusting the ingestion pipeline.
- Replace the embedding model in `src/embeddings.py` with an OpenAI or Instructor model if desired.

