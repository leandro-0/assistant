from spacy import Language
import re
from sqlalchemy.orm import joinedload
import logging
from app.core.lifespan import get_nomic_model
from app.core.utils import get_filename
from app.database.models import Article
from app.database.connection import Session
from app.state.documents_store_state import get_documents_store_state

logger = logging.getLogger("uvicorn.error")
try:
    db = Session()
    articles: list[Article] = list(
        db.query(Article)
        .options(joinedload(Article.issue))
        .filter(Article.download_url.isnot(None), Article.download_url != "")
        .all()
    )
    logger.info("Articles loaded")
except Exception as e:
    logger.error(f"Error loading articles: {e}")
finally:
    db.close()


def clean_query(nlp: Language, query: str) -> str:
    query = query.replace('"', "").replace("'", "")
    query = query.replace("\n", " ").replace("\r", " ")
    query = query.replace("*", "").replace("-", "").replace("~", "")
    query = query.replace("  ", " ").strip()

    doc = nlp(query)
    sentences = [sent.text.strip() for sent in doc.sents]
    return " ".join(
        [sent for sent in sentences if len(sent) > 0 and sent[-1] == "."]
    ).lower()


def search(
    query: str,
    top_k: int = 10,
) -> list[Article]:
    store_state = get_documents_store_state()
    model = get_nomic_model()
    query_embedding = (
        model.encode(
            [clean_query(store_state.nlp, query)],
            convert_to_tensor=True,
            prompt_name="query",
        )
        .cpu()
        .numpy()
    )
    _, indices = store_state.index.search(query_embedding, k=top_k * 4)

    already_in = set()
    results = []
    for i in indices[0]:
        id = store_state.store.chunks_ids[i]
        if id in already_in:
            continue

        already_in.add(id)
        results.append(id)
        if len(results) == top_k:
            break

    titles = [
        re.sub(r"pack\\.*?\\", "", p)
        for p in store_state.store.df.iloc[results]["Path"]
    ]
    arts = list(filter(lambda art: get_filename(art) in titles, articles))
    arts.sort(key=lambda art: titles.index(get_filename(art)))
    return arts
