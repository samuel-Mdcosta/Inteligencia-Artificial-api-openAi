import json
import os 
import random
import numpy as np
from utils import get_api_key
from vectorStore import VectorStore

HISTORICO_PATH = "historico.json"

def contexto():
    #faz as perguntas de acordo com o embedding
    client = get_api_key()
    vs = VectorStore(dim=1536, path="embeddings/index.faiss")
    index = vs.load()

    quiz_embedding = client.embeddings.create(
        model = "text-embedding-3-small",
        input = "Gerar Quiz"
    ).data[0].embedding

    I, D = vs.search(quiz_embedding, k = 1)
    texts = np.load("embeddings/index_meta.npy", allow_pickle=True)

    resultado_contexto = "\n".join([texts[i] for i in I[0]])
    return resultado_contexto

def gerar_pergunta():
    client = get_api_key()

    usar_contexto = random.random() > 0.05
    resultado_contexto = contexto() if usar_contexto else " Tema geral da neurociencia "

    persona = f"""
    Gere uma pergunta de múltipla escolha COM BASE no contexto abaixo.
    O contexto deve ser usado obrigatoriamente, a não ser que seja a parte livre (5%), onde ainda assim deve estar relacionado ao tema do contexto.

    Contexto:
    {resultado_contexto}

    Pergunta:

    Crie 4 alternativas (A, B, C, D).
    Indique claramente qual é a alternativa correta e explique por que ela está certa.

    Retorne SOMENTE em JSON no formato:
    {{
        "pergunta": "texto da pergunta",
        "alternativas": {{
            "A": "texto",
            "B": "texto",
            "C": "texto",
            "D": "texto"
        }},
        "correta": "A",
        "explicacao": "texto explicando a resposta correta"
    }}
    """
    resposta = client.chat.completions.creat(
        model = "gpt-3.5-turbo",
        temperature = 0.2,
        messages=[{"role": "user", "content": persona}]
    )
    
    conteudo = resposta.choices[0].message.content.strip()
    try:
        pergunta_json = json.loads(conteudo)
    except:
        raise ValueError("o modelo parou\n" + conteudo)
    return pergunta_json

def executar():
    pergunta = gerar_pergunta()

    print("?", pergunta["pergunta"])
    for alt, texto in pergunta["alternativas"].items():
        print(f"{alt}: {texto}")

    resposta_usuario = input("Digite sua Resposta:").strip().upper()

    correto = (resposta_usuario == pergunta["correta"])

    if correto:
        print("Parabens, resposta correta")
    else:
        print("Resposta errada. Explicação:")
        print("A resposta certa era: {pergunta['correta']}){pergunta['alternativas'][pergunta['correta']]}")
        print(pergunta["explicacao"])

if __name__ == "__main__":
    executar()