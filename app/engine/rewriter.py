import os
import torch
import boto3
import json
import logging
from transformers import pipeline
from botocore.exceptions import ClientError

logger = logging.getLogger("uvicorn.error")
local_pipe = None
client = None


def __init_inference_client():
    global client
    if client is None:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")


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


def __rewrite_inference(q: str) -> str:
    __init_inference_client()
    global client

    formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Eres un experto en historia latinoamericana. Tu tarea es tomar consultas relacionadas con la historia de América Latina y reformularlas para hacerlas más precisas, claras y detalladas. Deberías asegurarte de que la nueva consulta sea más específica, completa y comprensible. Responde solo con la consulta reformulada, sin agregar respuestas completas ni contenido adicional. No uses listas, títulos ni formato Markdown.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{q}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
    """
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 256,
    }
    request = json.dumps(native_request)

    try:
        response = client.invoke_model(
            modelId="us.meta.llama3-2-1b-instruct-v1:0", body=request
        )
        model_response = json.loads(response["body"].read())
        return model_response["generation"]
    except (ClientError, Exception) as e:
        logger.error(
            f"ERROR: Could not invoke 'us.meta.llama3-2-1b-instruct-v1:0'. Reason: {e}"
        )
        raise e


def rewrite(q: str, use_local_model: bool = False) -> str:
    if use_local_model:
        messages = [
            {
                "role": "system",
                "content": "Eres un experto en historia latinoamericana. Tu tarea es tomar consultas relacionadas con la historia de América Latina y reformularlas para hacerlas más precisas, claras y detalladas. Deberías asegurarte de que la nueva consulta sea más específica, completa y comprensible. Responde solo con la consulta reformulada, sin agregar respuestas completas ni contenido adicional. No uses listas, títulos ni formato Markdown.",
            },
            {"role": "user", "content": q},
        ]
        return __rewrite_local(q, messages)

    return __rewrite_inference(q)
