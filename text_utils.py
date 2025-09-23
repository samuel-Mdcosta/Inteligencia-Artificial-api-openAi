import re
from typing import List

def smart_chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[str]:

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap deve ser menor que chunk_size.")

    if not text or not text.strip():
        return []

    # Divide em frases mantendo pontuação
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Se a frase sozinha já for maior que chunk_size, quebra por caracteres
        if len(sentence) > chunk_size:
            # Fecha o chunk atual antes de quebrar a frase grande
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Quebra a frase longa em pedaços fixos
            start = 0
            while start < len(sentence):
                end = start + chunk_size
                chunks.append(sentence[start:end].strip())
                start += chunk_size - chunk_overlap
        else:
            # Se ainda cabe dentro do limite, adiciona a frase no chunk atual
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += sentence + " "
            else:
                # Fecha o chunk atual
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Cria novo chunk com sobreposição
                overlap_text = current_chunk[-chunk_overlap:]
                current_chunk = overlap_text.strip() + " " + sentence + " "

    # Adiciona o último chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
