from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
from fastapi import FastAPI
from boto3 import client
import logging
from flashrank import Ranker

logger = logging.getLogger("uvicorn.error")
__bedrock_client = None
__reranker = None
__models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global __models, __bedrock_client, __reranker
    __models["nomic"] = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True
    )
    __bedrock_client = client("bedrock-runtime", region_name="us-east-1")
    __reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/opt")
    yield
    __models.clear()
    __bedrock_client = None
    __reranker = None


def get_nomic_model() -> SentenceTransformer:
    global __models
    return __models["nomic"]


def get_bedrock_client():
    global __bedrock_client
    return __bedrock_client


def get_reranker() -> Ranker:
    global __reranker
    return __reranker
