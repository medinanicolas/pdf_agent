from langchain.tools.retriever import create_retriever_tool
from app.agent.utils import retriever

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_pdf",
    "Retrieve pdf documents that could contain useful information."
)