from typing import List
from src.embeddings import Embedder
from src.vectorstore import FaissStore
from langchain_google_genai.llms import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

class QASession:
    def __init__(self, store: FaissStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self.llm =  ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    def answer(self, question: str) -> str:
        q_vec = self.embedder.embed_query(question)
        docs = self.store.similarity_search(q_vec, k=4)
        context = "\n\n---\n\n".join(docs)

        prompt = f"Use the following context to answer the question. If not available, say you don't know. Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        if self.llm is None:
            raise RuntimeError('langchain_google_genai ChatGoogleGenerativeAI is not available (check installation and imports).')

        # Prefer predict/callable, then generate
        # 1) try predict
        try:
            if hasattr(self.llm, 'predict'):
                out = self.llm.predict(prompt)
                if out:
                    return str(out)
        except Exception:
            pass

        # 2) try calling the model
        try:
            if callable(self.llm):
                out = self.llm(prompt)
                if out:
                    return str(out)
        except Exception:
            pass

        # 3) try generate (LangChain LLMResult)
        try:
            if hasattr(self.llm, 'generate'):
                gen = self.llm.generate([prompt])
                if hasattr(gen, 'generations'):
                    gens = gen.generations
                    if gens and gens[0] and hasattr(gens[0][0], 'text'):
                        return gens[0][0].text
                return str(gen)
        except Exception:
            pass

        raise RuntimeError('Failed to get an answer from ChatGoogleGenerativeAI. Check API key, model name, and package compatibility.')
