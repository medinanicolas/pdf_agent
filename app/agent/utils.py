from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.core.config import settings
from app.agent.prompts import qa_prompt_template, hallucination_prompt_template

llm = ChatOpenAI(model=settings.OPENAI_CHAT_MODEL, temperature=0)

neo4j_vector_store = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL),
    url=settings.NEO4J_URI,
    username=settings.NEO4J_USERNAME,
    password=settings.NEO4J_PASSWORD,
    index_name="ChunkEmbedding",
    node_label="Chunk",
    text_node_properties=["text"],
    embedding_node_property="textEmbedding",
)

retriever = neo4j_vector_store.as_retriever()

qa_prompt = PromptTemplate(
    input_variables=["question", "context", "chat_history"],
    template=qa_prompt_template
)


qa_chain = qa_prompt | llm | StrOutputParser()

fallback_prompt = ChatPromptTemplate.from_template(
    """
    You are an intelligent assistant that operates without any external context. Based solely on your internal knowledge, determine how to handle the user's question:

    For questions that inherently require external or societal context (e.g., historical events, political figures, or verified medical information):

    Clearly state that answering accurately is not possible without more context. Inform the user that the question requires additional information or verification.
    For simple or general knowledge questions (e.g., universal facts or personal details unrelated to societal context):

    Provide a direct and accurate answer, based only on your internal knowledge.
    Always avoid hallucination. If you cannot answer reliably due to the lack of context, explicitly state the limitation instead of making assumptions or fabricating information.

    Examples of application:

    For a question like "Who is the president of [nation]?", respond:
    "I cannot answer that reliably without more context. Please provide additional details or consult a verified source."

    For a question like "What is the color of the sky?", respond:
    "The sky is typically blue during the day, under clear conditions."

    For a question like "What is my dog's name?", respond:
    "I do not have access to that information. Could you share your dog's name?"

    Be transparent and precise in your responses, operating solely within the boundaries of your internal knowledge.



    Chat History: {chat_history}
    Question: {question}
    """
)

fallback_chain = fallback_prompt | llm | StrOutputParser()

class HallucinationEvaluator(BaseModel):
    """Binary score for hallucination present in generation answer."""

    grade: str = Field(...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
    
hallucination_llm = llm.with_structured_output(HallucinationEvaluator)
hallucination_prompt = ChatPromptTemplate.from_template(hallucination_prompt_template)

hallucination_chain = hallucination_prompt | hallucination_llm

class AnswerabilityEvaluator(BaseModel):
    """Binary score for determining if a question is answerable using the provided documents."""
    grade: str = Field(..., description="Answer can be determined using the documents, 'yes' or 'no'")

answerability_llm = llm.with_structured_output(AnswerabilityEvaluator)
answerability_prompt_template = """
You are tasked with determining if the documents explicitly provide the information needed to answer the following question.

Documents: {documents}
Question: {question}

Carefully analyze the documents and check if any part of the documents explicitly contains the answer to the question. If the documents explicitly provide the answer to the question, respond with "yes". If the documents do not contain the exact answer to the question, respond with "no". 

Do not infer, assume, or provide any answers based on indirect information. Only respond with "yes" or "no", and make sure your answer is based on the exact content of the documents.
"""


answerability_prompt = ChatPromptTemplate.from_template(answerability_prompt_template)
answerability_chain = answerability_prompt | answerability_llm

def format_chat_history(messages):
    """
    Formats the chat history for inclusion in the prompt.

    Args:
        messages (list): A list of message objects (HumanMessage, ToolMessage, AIMessage).

    Returns:
        str: Formatted chat history as a string.
    """
    formatted_history = []
    for message in messages:
        if isinstance(message, HumanMessage):
            formatted_history.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_history.append(f"AI: {message.content}")
        elif isinstance(message, ToolMessage):
            formatted_history.append(f"Tool ({message.tool_name}): {message.content}")
    return "\n".join(formatted_history)
