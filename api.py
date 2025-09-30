from fastapi import FastAPI
from pydantic import BaseModel
from rag import gerar_resposta
from quiz import gerar_pergunta

app = FastAPI()

class PerguntaRequest(BaseModel):
    pergunta: str

class RespostaQuizRequest(BaseModel):
    pergunta: str
    gerar_resposta: str
    correta: str

#rotas
@app.post("/chat")
def chat(pergunta: PerguntaRequest):
    resposta = gerar_resposta(pergunta.pergunta)
    return {"resposta": resposta}

@app.get("/quiz")
def quiz():
    pergunta = gerar_pergunta()
    return pergunta

@app.post("/resposta_quiz")
def resposta_quiz(resposta: RespostaQuizRequest):
    correta = (resposta.gerar_resposta.upper() == resposta.correta.upper())
    return {"correta": correta}