from app.agent.state import GraphState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.agent.tools import retriever_tool
from app.agent.utils import format_chat_history, qa_chain, answerability_chain, hallucination_chain, fallback_chain


def retrieve(state: GraphState):
    """
    Retrieves the documents from the vectorstore.
    
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    
    print(f"> 📃 Retrieving documents...")
    user_messages = [msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)]
    query = user_messages[-1] if user_messages else ""
    
    retriever_input = {"query": query}
    
    response = retriever_tool.invoke(retriever_input)
    
    
    tool_message = ToolMessage(
        content=response,
        tool_name="retriever",
        tool_call_id='id'
    )

    return {"messages": [tool_message]}


def check_answerability(state: dict):
    """
    Check if the question can be answered based on the provided documents in the state.

    Args:
        state (dict): The current state containing documents and the question.

    Returns:
        str: "answerable" if the question can be answered, otherwise "not answerable".
    """
    messages = state["messages"]
    last_message = messages[-1]

    question = messages[-2].content
    docs = last_message.content

    if not docs or not question:
        print("> ❌ \033[91mMissing documents or question in the state\033[0m")
        return "not answerable"

    result = answerability_chain.invoke({"documents": docs, "question": question})
    
    if result.grade == "yes":
        print("> ✅ \033[92mThe question can be answered with the provided documents\033[0m")
        return "answerable"
    else:
        print("> ❌ \033[91mThe question cannot be answered with the provided documents\033[0m")
        return "not answerable"

def generate(state: GraphState):
    """
    Generates an answer using the retrieved documents/query results.
    
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    
    print(f"> 🧠 Generating an answer ...")
    
    messages = state["messages"]
    question = messages[-2].content
    last_message = messages[-1]
    docs = last_message.content
    response = qa_chain.invoke({"question": question, "context": docs, "chat_history": format_chat_history(messages[:-1])})
    
    return {"messages": [AIMessage(content=response)]}


def fallback(state):
    """
    Fallback to LLM's internal knowledge.
    
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    
    print(f"> 👈 Fallback to LLM's internal knowledge ...")
    messages = state["messages"]
    question = messages[-2].content

    response = fallback_chain.invoke({"question": question, "chat_history":format_chat_history(messages[:-2])})
    
    return {"messages": [AIMessage(content=response)]}

def check_hallucination(state: GraphState):
    """
    Check if the generation is hallucinated.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    messages = state["messages"]
    last_message = messages[-1]

    docs = messages[-2].content
    generation = last_message.content
    
    grounded = hallucination_chain.invoke({"documents": docs, "generation": generation})
    
    if grounded.grade == "yes":
        print("> ✅ \033[92mAnswer addresses the question\033[0m")
        return "useful"
    
    else:
        print("> ❌ \033[91mGeneration is not grounded in the documents\033[0m")
        return "not supported"
    