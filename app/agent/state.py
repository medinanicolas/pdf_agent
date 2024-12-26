from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated, List
from langchain_core.messages.base import BaseMessage


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    