from utils import get_api_key
from vectorStore import VectorStore
import numpy as np

def gerar_pergunta(pergunta):
    client = get_api_key()

    #carrega o indice
    vs = VectorStore()
    index = vs.load()

    #embedding da pergunta
    q_embedding = client.embeddings.create(
        model="",
        input=pergunta
    ).data[0].embedding

    #busca nos veotres mais proximos
    I, D = vs.search(q_embedding, k=3)

    #carrega os textos normais
    texts = np.load("vector_store_texts.npy", allow_pickle=True)

    #gera o contexto da conversa
    context = "\n\n---\n\n".join([texts[i] for i in I[0]])

    #cria a resposta com parametros
    completion = client.chat.completions.create(
        model="",
        #mais fiel ao contexto nao alucina 
        temperature=0.2,
        #consideras as possiveis opcoes
        top_p=1,
        #nao penaliza repeticoes
        frequency_penalty=0,
        #nao inventa
        presence_penalty=0,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": f"Pergunta: {pergunta}\n\nContexto:\n{context}"}
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    pergunta = input("O que deseja ?")
    resposta = gerar_pergunta(pergunta)
    print("\n Resposta:", resposta)
