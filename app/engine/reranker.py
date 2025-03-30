from flashrank import Ranker, RerankRequest
from app.core.lifespan import get_reranker


def rerank(query: str, passages: list[dict[str, str]]) -> list[int]:
    ranker: Ranker = get_reranker()
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    return [r["id"] for r in results]
