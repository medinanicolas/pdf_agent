from langgraph.graph import START, END, StateGraph

from app.agent.nodes import check_answerability, check_hallucination, fallback, generate, retrieve
from app.agent.state import GraphState
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("fallback", fallback)


workflow.add_edge(START, 'retrieve')

workflow.add_conditional_edges(
    'retrieve',
    check_answerability,
    {
        "answerable": 'generate',
        "not answerable": 'fallback'
    }
)
workflow.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        "useful": END,
        "not supported": 'fallback'
    }
)
workflow.add_edge('fallback', END)

graph = workflow.compile(checkpointer=memory)