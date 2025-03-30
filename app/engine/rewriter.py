import json
import logging
from botocore.exceptions import ClientError
from app.core.lifespan import get_bedrock_client

logger = logging.getLogger("uvicorn.error")


def rewrite(q: str) -> str:
    bedrock = get_bedrock_client()
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
        response = bedrock.invoke_model(
            modelId="us.meta.llama3-2-1b-instruct-v1:0", body=request
        )
        model_response = json.loads(response["body"].read())
        return model_response["generation"]
    except (ClientError, Exception) as e:
        logger.error(
            f"ERROR: Could not invoke 'us.meta.llama3-2-1b-instruct-v1:0'. Reason: {e}"
        )
        raise e
