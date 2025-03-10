import os
import torch
from transformers import pipeline
from huggingface_hub import InferenceClient

local_pipe = None
client_inference = None


def __init_inference_client():
    global client_inference
    if client_inference is None:
        client_inference = InferenceClient(
            provider="sambanova", api_key=os.getenv("HF_TOKEN")
        )


def __init_pipe():
    global local_pipe
    if local_pipe is None:
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        local_pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )


def __rewrite_local(q: str, messages: str) -> str:
    __init_pipe()
    global local_pipe

    outputs = local_pipe(
        messages,
        max_new_tokens=256,
    )

    return outputs[0]["generated_text"][-1]["content"]


def __rewrite_inference(q: str, messages: str) -> str:
    __init_inference_client()
    global client_inference

    completion = client_inference.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=messages,
        max_tokens=256,
    )

    return completion.choices[0].message.content


def rewrite(q: str, use_local_model: bool = False) -> str:
    messages = [
        {
            "role": "system",
            "content": "Eres un experto en historia latinoamericana. Tu tarea es tomar consultas relacionadas con la historia de América Latina y reformularlas para hacerlas más precisas, claras y detalladas. Deberías asegurarte de que la nueva consulta sea más específica, completa y comprensible. Responde solo con la consulta reformulada, sin agregar respuestas completas ni contenido adicional. No uses listas, títulos ni formato Markdown.",
        },
        {"role": "user", "content": q},
    ]

    if use_local_model:
        return __rewrite_local(q, messages)

    return __rewrite_inference(q, messages)
