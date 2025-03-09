import torch
from transformers import pipeline


model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def get_messages(q):
    return [
        {
            "role": "system",
            "content": "Eres un experto en historia latinoamericana. Tu tarea es tomar consultas relacionadas con la historia de América Latina y reformularlas para hacerlas más precisas, claras y detalladas. Deberías asegurarte de que la nueva consulta sea más específica, completa y comprensible. Responde solo con la consulta reformulada, sin agregar respuestas completas ni contenido adicional. No uses listas, títulos ni formato Markdown.",
        },
        {"role": "user", "content": q},
    ]


def rewrite(q: str) -> str:
    outputs = pipe(
        get_messages(q),
        max_new_tokens=256,
    )

    return outputs[0]["generated_text"][-1]["content"]
