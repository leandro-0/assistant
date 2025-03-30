import logging
import os
import pickle
import pandas as pd
import pickle

logger = logging.getLogger("uvicorn.error")


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


class DocumentsStore:
    def __init__(
        self,
        pickle_df_path: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embeddings_pickle_path: str = None,
    ):
        self.__df = pd.read_pickle(pickle_df_path)
        self.__df = self.__df.reset_index(drop=True)
        self.__embeddings = None
        self.__chunks: list[str] = []
        self.__chunks_ids: list[int] = []
        self.__chunk_size = chunk_size
        self.__chunk_overlap = chunk_overlap
        self.__pickle_df_path = pickle_df_path
        self.__embeddings_pickle_path = embeddings_pickle_path
        self.__generate_embeddings()

    @property
    def df(self):
        return self.__df

    @property
    def embeddings(self):
        return self.__embeddings

    @property
    def chunks(self):
        return self.__chunks

    @property
    def chunks_ids(self):
        return self.__chunks_ids

    @property
    def chunk_size(self):
        return self.__chunk_size

    @property
    def chunk_overlap(self):
        return self.__chunk_overlap

    def __create_chunks(self) -> None:
        text_splitter = TokenTextSplitter(
            chunk_size=self.__chunk_size, chunk_overlap=self.__chunk_overlap
        )

        try:
            for i, r in self.__df.iterrows():
                new_chunks = text_splitter.split_text(r["Text"])
                self.__chunks.extend(new_chunks)
                self.__chunks_ids.extend([i] * len(new_chunks))

            logging.info("Chunking finished")
        except Exception as e:
            logging.error(f"Error during chunking: {e}")
            raise e

    def __generate_embeddings(self):
        self.__create_chunks()

        if self.__embeddings_pickle_path and os.path.exists(
            self.__embeddings_pickle_path
        ):
            try:
                with open(self.__embeddings_pickle_path, "rb") as f:
                    self.__embeddings = pickle.load(f)
                logger.info("Embeddings loaded from pickle")
                return
            except Exception as e:
                logger.error(f"Error loading embeddings from pickle: {e}")

        if len(self.__chunks) == 0:
            logging.warning("No chunks to process")
            return

        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True
        )

        pool = model.start_multi_process_pool()
        self.__embeddings = model.encode_multi_process(
            self.__chunks, pool, prompt_name="passage"
        )
        model.stop_multi_process_pool(pool)

        # Save embeddings to disk
        try:
            with open("store-embeddings-nomic.pkl", "wb") as f:
                pickle.dump(self.__embeddings, f)
        except Exception as e:
            logging.error(f"Error saving embeddings: {e}")

        logging.info("Embeddings generated")

    def chunk_text_from_id(self, id: int) -> str | None:
        if id < 0 or id >= len(self.__chunks):
            return None
        return self.__chunks[id]

    def document_text_from_chunk_id(self, chunk_id: int) -> str | None:
        if chunk_id < 0 or chunk_id >= len(self.__chunks_ids):
            return None
        return self.__df.iloc[self.__chunks_ids[chunk_id]]["Text"]

    def save_to_pickle(self, file_name: str) -> None:
        try:
            with open(file_name, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            logging.error(f"Error saving to pickle: {e}")
            raise e

    @staticmethod
    def load_from_pickle(file_name: str) -> "DocumentsStore":
        try:
            return CustomUnpickler(open(file_name, "rb")).load()
        except Exception as e:
            logging.error(f"Error loading from pickle: {e}")
            raise e
