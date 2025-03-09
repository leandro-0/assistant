from sentence_transformers import SentenceTransformer
import faiss

# import spacy
import re
import pickle
import pandas as pd
from sqlalchemy.orm import joinedload
import logging
from app.database.models import Article
from app.database.connection import Session

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

df = pd.read_csv("preprocessed.csv")
logger.info("Dataframe loaded")

# nlp = spacy.load("es_core_news_sm")
# logger.info("Spacy loaded")

model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
logger.info("SentenceTransformer model loaded")

index = faiss.read_index("index-chunks-512-50-nomic.faiss")
logger.info("FAISS index loaded")

with open("chunks-512-50-nomic.pkl", "rb") as f:
    chunks, chunks_ids = pickle.load(f)

logger.info("Chunks loaded")


def clean_query(query: str) -> str:
    query = query.replace('"', "").replace("'", "")
    query = query.replace("\n", " ").replace("\r", " ")
    query = query.replace("  ", " ").strip()
    return query.lower()
    # doc = nlp(query)
    # sentences = [sent.text.strip() for sent in doc.sents]
    # return " ".join(
    #     [sent for sent in sentences if len(sent) > 0 and sent[-1] == "."]
    # ).lower()


def get_filename(article: Article) -> str:
    return f"{article.published.year}-" + (
        " ".join(re.sub(r"[^\w\s]", "", article.title).split()).replace(" ", "-")[:100]
        + ".pdf"
    )


def search(query: str, top_k: int = 10) -> list[Article]:
    query_embedding = (
        model.encode([clean_query(query)], convert_to_tensor=True, prompt_name="query")
        .cpu()
        .numpy()
    )
    _, indices = index.search(query_embedding, k=top_k * 4)

    already_in = set()
    results = []
    for i in indices[0]:
        id = chunks_ids[i]
        if id in already_in:
            continue

        already_in.add(id)
        results.append(id)
        if len(results) == top_k:
            break

    titles = [re.sub(r"pack\\.*?\\", "", p) for p in df.iloc[results]["Path"]]
    arts = list(filter(lambda art: get_filename(art) in titles, articles))
    arts.sort(key=lambda art: titles.index(get_filename(art)))
    return arts
