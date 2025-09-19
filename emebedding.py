import os
import numpy as np
from pypdf import PdfReader
from utils import get_api_key
from vectorStore import VectorStore

#carrega o conteudo dos pdfs (apenas texto)
def load_pdf(pdf_folder = "Docs"):
    texts = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(pdf_folder, file))
            for page in reader.pages:
                texts.append(page.extract_text())
    return texts

def create_embeddings(texts):
    client = get_api_key()
    response = client.embeddings.create(
        model="",
        input=texts
    )
    return [item.embedding for item in response.data]

if __name__ == "__main__":
    texts = load_pdf()
    embeddings = create_embeddings(texts)

    vs = VectorStore(dim=len(embeddings[0]))
    vs.add_vectors(embeddings, texts)
    vs.save()
    print("embedding criados e salvos")