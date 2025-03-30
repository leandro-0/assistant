import faiss
from app.engine.documents_store import DocumentsStore
from spacy import load as load_spacy_corpus
from spacy import Language


class DocumentsStoreState:
    def __init__(self) -> None:
        self.__store: DocumentsStore = DocumentsStore.load_from_pickle(
            "store-better-preprocess.pkl"
        )
        self.__index: faiss.IndexFlatL2 = None
        self.__nlp: Language = load_spacy_corpus("es_core_news_sm")
        self.__init_index()

    @property
    def store(self) -> DocumentsStore:
        return self.__store

    @property
    def index(self) -> faiss.IndexFlatL2:
        return self.__index

    @property
    def nlp(self) -> Language:
        return self.__nlp

    def __init_index(self) -> None:
        self.__index = faiss.IndexFlatL2(self.__store.embeddings.shape[1])
        self.__index.add(self.__store.embeddings)


__global_state = DocumentsStoreState()


def get_documents_store_state() -> DocumentsStoreState:
    return __global_state
