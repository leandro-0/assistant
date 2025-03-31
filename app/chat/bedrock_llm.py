import logging
import json
from botocore.exceptions import ClientError
from langchain.llms.base import LLM
from app.core.lifespan import get_bedrock_client

logger = logging.getLogger("uvicorn.error")


class BedrockLLM(LLM):
    model_id: str = "us.meta.llama3-2-3b-instruct-v1:0"
    max_gen_len: int = 512

    @property
    def _llm_type(self) -> str:
        return "amazon_bedrock"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        bedrock = get_bedrock_client()

        formatted_prompt = f"""
{prompt}
        """

        request_payload = {
            "prompt": formatted_prompt,
            "max_gen_len": self.max_gen_len,
            "temperature": 0.1,
        }

        request = json.dumps(request_payload)

        try:
            response = bedrock.invoke_model(modelId=self.model_id, body=request)
            model_response = json.loads(response["body"].read())
            return model_response["generation"]
        except (ClientError, Exception) as e:
            logger.error(f"ERROR: Could not invoke '{self.model_id}'. Reason: {e}")
            raise e
