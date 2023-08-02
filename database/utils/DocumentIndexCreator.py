import os
import pickle
import re
import logging
import json
import time
import openai
from dotenv import load_dotenv
import pinecone
from datetime import datetime


from llama_index import (
    GPTVectorStoreIndex,
    ServiceContext,
    Document,
    set_global_service_context,
)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import OpenAI
from llama_index.llm_predictor import LLMPredictor
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    # TitleExtractor,
    QuestionsAnsweredExtractor,
    # SummaryExtractor,
    KeywordExtractor,
)
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter

openai.api_key=os.getenv('OPENAI_API_KEY')


class DocumentIndexCreator:
    def __init__(self, save_folder, index_filename, batch_size=100):
        self.save_folder = save_folder
        self.index_filename = index_filename
        self.batch_size = batch_size
        self.doc_titles = []
        self.doc_paths = []
        self.doc_ids = []
        self.doc_embeddings = {}

        self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)
        self.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
        self.llm_predictor = LLMPredictor(llm=self.llm)
        self.text_splitter = TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)
        self.metadata_extractor = MetadataExtractor(
            extractors=[
                # TitleExtractor(nodes=2),
                QuestionsAnsweredExtractor(
                    questions=3, llm_predictor=self.llm_predictor
                ),
                # SummaryExtractor(summaries=["prev", "self"]),
                KeywordExtractor(keywords=5, llm_predictor=self.llm_predictor),
            ]
        )
        self.node_parser = SimpleNodeParser(
            text_splitter=self.text_splitter, metadata_extractor=self.metadata_extractor
        )

        self.load_documents()
    
    def sanitize_filename(self, filename):
        return re.sub(r"[/\\]", "_", filename)
    
    def read_file_as_string(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def get_file_metadata(self, file_path):
        metadata_path = file_path.replace(".txt", ".json")
        md = self.read_file_as_string(metadata_path)
        md = json.loads(md)
        if md:
            return md
        return {}

    def load_last_runtimes(self):
        if os.path.exists("runtimes.json"):
            with open("runtimes.json", "r") as file:
                return json.load(file)
        return {}

    def update_last_runtime(self, doc_path):
        self.doc_embeddings[doc_path]["LastRuntime"] = time.time()

    def save_last_runtimes(self):
        with open("runtimes.json", "w") as file:
            json.dump(self.doc_embeddings, file)

    def load_documents(self):
        self.doc_embeddings = self.load_last_runtimes()
        for dirpath, dirnames, filenames in os.walk(self.save_folder):
            for filename in filenames:
                if filename.endswith(".txt"):
                    subdir_name = os.path.basename(dirpath)
                    file_name = os.path.splitext(filename)[0]

                    metadata_path = os.path.join(
                        dirpath, self.sanitize_filename(file_name) + ".json"
                    )
                    metadata = self.get_file_metadata(metadata_path)

                    last_updated_date_str = metadata.get("LastUpdatedDate", "")
                    last_updated_date = datetime.fromisoformat(last_updated_date_str[:-1]) if last_updated_date_str else None

                    doc_path = os.path.join(dirpath, filename)
                    if last_updated_date and (doc_path not in self.doc_embeddings or last_updated_date > datetime.fromtimestamp(self.doc_embeddings[doc_path].get("LastRuntime", 0))):
                        self.doc_titles.append(subdir_name + " - " + file_name)
                        self.doc_paths.append(doc_path)
                        self.doc_embeddings[doc_path] = {"LastRuntime": 0}


    def index_documents(self):
        nodes = []
        for title, path in zip(self.doc_titles, self.doc_paths):
            if path.endswith(".txt"):
                text = self.read_file_as_string(path)
                extra_info = self.get_file_metadata(path)

                nodes.append(Document(text=text, doc_id=title, extra_info=extra_info))
                print("Document added: " + title)

                if len(nodes) >= self.batch_size:
                    self.process_batch(nodes)
                    nodes = []

        if nodes:
            self.process_batch(nodes)

    def process_batch(self, nodes):
        service_context = ServiceContext.from_defaults(
            llm=self.llm, embed_model=self.embed_model, node_parser=self.node_parser
        )
        set_global_service_context(service_context)

        start = time.time()
        print(time.time())

        parsed_nodes = self.node_parser.get_nodes_from_documents(nodes, show_progress=True)

        print(time.time() - start)
        print("Nodes added: " + str(len(parsed_nodes)))

        self.update_index(parsed_nodes)

    def save_index(self):
        with open(self.index_filename, "wb") as file:
            index_data = {
                "doc_ids": self.doc_ids,
                "doc_embeddings": self.doc_embeddings,
            }
            pickle.dump(index_data, file)

    def load_index(self):
        if os.path.exists(self.index_filename):
            with open(self.index_filename, "rb") as file:
                index_data = pickle.load(file)
                self.doc_ids = index_data.get("doc_ids", [])
                self.doc_embeddings = index_data.get("doc_embeddings", [])

    def update_index(self, nodes):
        for node in nodes:
            if node.ref_doc_id not in self.doc_ids:
                self.doc_ids.append(node.ref_doc_id)
                self.doc_embeddings.append(node.embedding)
            else:
                index = self.doc_ids.index(node.ref_doc_id)
                self.doc_embeddings[index] = node.embedding

        self.save_index()


def create_and_load_index(index_name, nodes):
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )

    pinecone_index = pinecone.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002", embed_batch_size=100
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    GPTVectorStoreIndex(
        nodes,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )