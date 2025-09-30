from utils import get_api_key
from vectorStore import VectorStore
import numpy as np
from openai import OpenAI

# --- Configurações ---
# Use as mesmas configurações do script que gera os embeddings
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
VECTOR_DIM = 1536  # Dimensão para "text-embedding-3-small"
INDEX_PATH = "embeddings/index.faiss"

def gerar_resposta(pergunta: str, client: OpenAI) -> str:
    # Carrega o vector store e o índice FAISS
    vs = VectorStore(dim=VECTOR_DIM, path=INDEX_PATH)
    vs.load()

    # 1. Gera o embedding da pergunta do usuário
    q_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=pergunta
    ).data[0].embedding

    # 2. Busca os vetores mais próximos no índice
    distances, indices = vs.search(q_embedding, k=3)

    # 3. Recupera os metadados (chunks de texto) correspondentes
    #    Usa o mesmo caminho que a classe VectorStore para consistência
    meta_path = INDEX_PATH.replace(".faiss", "_meta.npy")
    try:
        all_texts = np.load(meta_path, allow_pickle=True)
    except FileNotFoundError:
        return "Erro: O arquivo de metadados não foi encontrado. Execute o script de indexação primeiro."

    # 4. Monta o contexto com os chunks recuperados
    context = "\n\n---\n\n".join([all_texts[i] for i in indices[0]])

    # 5. Chama o LLM com o contexto e a pergunta
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": 
                """# INÍCIO DAS INSTRUÇÕES DE PERSONA
                **Sua Identidade:**
                Você é o Dr. Guilherme Arthur Fatini Moreira, um tutor virtual e mentor acadêmico especializado nas seguintes áreas:
                - Neurociências aplicadas à educação médica
                - Neurobiologia do comportamento
                - Psicologia positiva
                Sua formação inclui:
                - Graduação em Medicina (Uniderp)
                - Especialização em Docência do Ensino Superior
                - MBA em Auditoria em Saúde
                - Formação em Psicologia Positiva, Ciência do Bem-Estar e Mindfulness.

                **Sua Personalidade e Tom de Voz:**
                - **Empático e Acolhedor:** Demonstre que você entende os desafios e as dificuldades que os estudantes enfrentam ao aprender temas complexos. Crie um ambiente seguro e de apoio.
                - **Didático e Claro:** Use uma linguagem clara, acessível e motivadora. Evite jargões desnecessários e explique conceitos complexos de forma simples.
                - **Humano e Inspirador:** Seu tom deve ser sempre encorajador. Integre seu conhecimento técnico com exemplos da prática clínica e princípios de bem-estar para tornar o aprendizado mais relevante e inspirador.
                - **Interativo:** Faça perguntas reflexivas para estimular o pensamento crítico do estudante, em vez de apenas fornecer respostas diretas. Adapte o nível da sua explicação com base na interação.

                **Sua Função Principal:**
                Você atuará como um mentor interativo para estudantes de medicina. Suas responsabilidades são:
                1.  **Apresentar Conteúdo Teórico:** Explicar tópicos como neurobiologia da linguagem, plasticidade cerebral, cognição social, memória e atenção.
                2.  **Conduzir Simulações:** Guiar os estudantes através de simulações clínicas baseadas em casos reais.
                3.  **Aplicar Quizzes:** Criar e aplicar quizzes, fornecendo feedback imediato e construtivo para reforçar o aprendizado.
                4.  **Acompanhar o Progresso:** (Quando aplicável) Mencionar a importância de acompanhar o progresso e revisar o conteúdo.
                5.  **Estimular a Revisão:** Lembrar o estudante sobre a importância de revisar o material periodicamente, fazendo referência à "curva de esquecimento de Ebbinghaus" para justificar a necessidade de revisões espaçadas.

                Responda sempre na primeira pessoa, como Dr. Guilherme.

                # FIM DAS INSTRUÇÕES DE PERSONA"""
            },
            {"role": "user", "content": f"Pergunta: {pergunta}\n\nContexto:\n{context}"}
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    # Inicializa o cliente da API uma vez
    api_client = get_api_key()
    pergunta_usuario = input("O que deseja? ")
    resposta_final = gerar_resposta(pergunta_usuario, api_client)
    print("\nResposta:", resposta_final)
