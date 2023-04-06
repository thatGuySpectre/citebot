import yaml
import os

import langchain
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("keys.yaml") as f:
    os.environ.update(yaml.safe_load(f))

splitter = RecursiveCharacterTextSplitter(["\n\n", ". \n", "\n", " ", ""], chunk_size=500, chunk_overlap=100)

# local file paths also work
urls = ["https://arxiv.org/pdf/2304.01852.pdf",
        "https://arxiv.org/pdf/q-bio/0511037.pdf"]

docs = []
for url in urls:
    raw_docs = PyPDFLoader(file_path=url).load_and_split(text_splitter=splitter)
    for d in raw_docs:
        d.metadata.update({"source": f"{url} page {d.metadata['page']}"})
    docs.extend(raw_docs)

# for actual use should use a better embeddings model, for example embeddings=OpenAiEmbeddings
db = Chroma.from_documents(docs, persist_directory="db")
