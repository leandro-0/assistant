import bisect
import logging
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from app.chat.bedrock_llm import BedrockLLM
from app.core.lifespan import get_chat_embeddings_model
from app.state.documents_store_state import get_documents_store_state

logger = logging.getLogger("uvicorn.error")


def find_index_range(store_state, target: int) -> tuple[int, int] | None:
    start_index = bisect.bisect_left(store_state.store.chunks_ids, target)
    end_index = bisect.bisect_right(store_state.store.chunks_ids, target) - 1

    if (
        start_index == len(store_state.store.chunks_ids)
        or store_state.store.chunks_ids[start_index] != target
    ):
        return None
    return (start_index, end_index)


def get_chunks(target: int) -> list[str]:
    store_state = get_documents_store_state()
    indices = find_index_range(store_state, target)
    return store_state.store.chunks[indices[0] : indices[1]]


def get_db(testing_chunks, embeddings):
    return FAISS.from_texts(texts=testing_chunks, embedding=embeddings)


def get_qa_chain(llm, db):
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 3})
    )


def build_qa_chain_for_target(
    target,
    bedrock_model_id="us.meta.llama3-2-3b-instruct-v1:0",
):
    testing_chunks = get_chunks(target)
    embeddings_model = get_chat_embeddings_model()
    db = get_db(testing_chunks, embeddings_model)
    llm = BedrockLLM(model_id=bedrock_model_id)
    return get_qa_chain(llm, db)


def answer_question(
    question: str,
    target_df_index: int,
) -> str:
    try:
        chain = build_qa_chain_for_target(target_df_index)
        raw_answer = chain.invoke(question)
        return raw_answer["result"].strip()
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return "Hubo un error al intentar responder la pregunta, intenta más tarde."
