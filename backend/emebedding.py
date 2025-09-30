import os
import numpy as np
from pypdf import PdfReader
from utils import get_api_key
from vectorStore import VectorStore
from text_utils import smart_chunk_text # <-- 1. Importar a função de chunking
from openai import OpenAI

# --- Configurações (Consistentes com rag.py) ---
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIM = 1536
INDEX_PATH = "embeddings/index.faiss"

def load_and_process_pdfs(client: OpenAI, pdf_folder: str = "Docs") -> tuple[list[list[float]], list[str]]:
    """
    Carrega PDFs, extrai texto, divide em chunks e cria embeddings.
    """
    all_chunks = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            print(f"Processando arquivo: {file}...")
            reader = PdfReader(os.path.join(pdf_folder, file))
            full_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            
            # 2. Aplicar o chunking inteligente no texto completo do arquivo
            chunks = smart_chunk_text(full_text, chunk_size=1024, chunk_overlap=128)
            all_chunks.extend(chunks)
            print(f"-> Extraídos {len(chunks)} chunks.")

    if not all_chunks:
        print("Nenhum chunk de texto foi gerado. Verifique a pasta 'Docs'.")
        return [], []

    # 3. Criar embeddings para todos os chunks de uma vez
    print(f"\nGerando embeddings para um total de {len(all_chunks)} chunks...")
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=all_chunks
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings, all_chunks

if __name__ == "__main__":
    # Garante que o diretório de embeddings exista
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    api_client = get_api_key()
    embeddings, chunks = load_and_process_pdfs(api_client)

    if embeddings and chunks:
        # 4. Adicionar os embeddings e os chunks ao VectorStore
        vs = VectorStore(dim=VECTOR_DIM, path=INDEX_PATH)
        vs.add_vectors(embeddings, chunks)
        vs.save()
        print(f"\nSucesso! {len(embeddings)} embeddings foram criados e salvos em '{INDEX_PATH}'.")