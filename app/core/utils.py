import re


def get_filename(article) -> str:
    return f"{article.published.year}-" + (
        " ".join(re.sub(r"[^\w\s]", "", article.title).split()).replace(" ", "-")[:100]
        + ".pdf"
    )
