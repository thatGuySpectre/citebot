import os
import yaml

from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQAWithSourcesChain

with open("keys.yaml") as f:
    os.environ.update(yaml.safe_load(f))

db = Chroma(persist_directory="db")
llm = OpenAI()

papers = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

while True:
    response = papers(inputs={"question": input()})
    print(response["answer"].strip())
    print("Sources: ", response["sources"])
