import logging
from app.chat.models.message import Message
from app.core.utils import get_filename
from app.database.connection import Session
from app.database.models import Article
from app.state.documents_store_state import get_documents_store_state

logger = logging.getLogger("uvicorn.error")


class Chat:
    def __init__(self, article_id: int) -> None:
        self.__article_id = article_id
        self.__messages: list[Message] = []
        self.__df_id = None
        self.__load_df_id()

    def __load_df_id(self) -> None:
        try:
            store_state = get_documents_store_state()
            db = Session()
            current = db.query(Article).filter(Article.id == self.__article_id).first()
            if current is None:
                raise ValueError(f"Article with id {self.__article_id} not found")

            filename = get_filename(current)
            self.__df_id = (
                store_state.store.df["Path"].str.contains(filename, na=False).idxmax()
            )
        except Exception as e:
            logger.error(f"Error getting DF id for article {self.__article_id}: {e}")
        finally:
            db.close()

    def add_message(self, message: str, sender: str) -> None:
        self.__messages.append(Message(message, sender))

    @property
    def messages(self) -> list[Message]:
        return self.__messages

    @property
    def article_id(self) -> int:
        return self.__article_id

    @property
    def df_id(self) -> int:
        return self.__df_id
