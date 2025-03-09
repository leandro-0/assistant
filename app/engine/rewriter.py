import os
from huggingface_hub import InferenceClient

client = InferenceClient(provider="sambanova", api_key=os.getenv("HF_TOKEN"))


def rewrite(q: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Eres un experto en historia latinoamericana. Tu tarea es tomar consultas relacionadas con la historia de América Latina y reformularlas para hacerlas más precisas, claras y detalladas. Deberías asegurarte de que la nueva consulta sea más específica, completa y comprensible. Responde solo con la consulta reformulada, sin agregar respuestas completas ni contenido adicional. No uses listas, títulos ni formato Markdown.",
        },
        {"role": "user", "content": q},
    ]

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=messages,
        max_tokens=256,
    )

    return completion.choices[0].message.content
