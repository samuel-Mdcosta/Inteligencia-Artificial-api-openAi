from rag import gerar_resposta
from quiz import gerar_pergunta

def ia():
    print("IA iniciada! Pergunte algo ou digite 'Quiz' para obter perguntas. Digite 'Sair' para encerrar a IA.")

    while True:
        pergunta_usuario = input("O que deseja? ").strip().lower()

        