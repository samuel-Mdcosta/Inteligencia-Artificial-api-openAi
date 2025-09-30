from rag import gerar_resposta
from quiz import gerar_pergunta
from utils import get_api_key

def ia():
    print("IA iniciada! Pergunte algo ou digite 'Quiz' para obter perguntas. Digite 'Sair' para encerrar a IA.")

    api_client = get_api_key()

    while True:
        pergunta_usuario = input("O que deseja? ").strip().lower()

        if pergunta_usuario == "sair":
            print("IA encerrada.")
            break

        elif "quiz" in pergunta_usuario:
            print("\n Gerando perguntas")
            gerar_pergunta()

        else:
            print("\n IA respondendo")
            resposta = gerar_resposta(pergunta_usuario, api_client)
            print(resposta)

if __name__ == "__main__":
    ia()