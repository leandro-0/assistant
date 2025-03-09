from dataclasses import dataclass
from sqlalchemy import Column, Integer, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


@dataclass
class Origin:
    name: str
    url: str

    def from_string(string: str):
        name, url = string.split(",")
        return Origin(name, url)


class Issue(Base):
    __tablename__ = "issues"
    id = Column(Integer, primary_key=True, autoincrement=True)
    origin = Column(Text)
    title = Column(Text)
    series = Column(Text)
    description = Column(Text)
    cover_image_url = Column(Text)
    articles_page = Column(Text)
    articles = relationship(
        "Article", back_populates="issue", cascade="all, delete-orphan"
    )


class Article(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    issue_id = Column(Integer, ForeignKey("issues.id"))
    title = Column(Text)
    authors = Column(Text)
    pages = Column(Text)
    download_url = Column(Text)
    page_url = Column(Text)
    published = Column(DateTime)
    issue = relationship("Issue", back_populates="articles")
