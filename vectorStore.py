import faiss
import numpy as np
import os

#modelo
class VectorStore:
    #construtor da classe
    def __init__(self, dim, path="embeddings/index.faiss"):
        #dim = dimensao
        #path = caminho do arquivo
        self.dim = dim
        self.path = path
        self.index = faiss.IndexFlatL2(dim)

    def add_vectors(self, vectors, metadata):
        #convertidos para um ponto flutuante de 32bits
        self.index.add(np.array(vectors).astype('float32'))

        #salvar os dados
        np.save(self.path.replace(".faiss", "_meta.npy"), metadata)