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
    You are an assistant for question-answering tasks. Your task is to respond appropriately based on the nature of the question.

    If the question involves a business topic, is of significant importance, or you believe more context is needed to provide an accurate answer:
    1. Indicate that the context provided is insufficient to answer the question definitively.
    2. Mention that more specific information or additional documents are needed to answer the question accurately.
    3. Clearly state that you are providing a response based on general knowledge, and your answer may not be fully tailored to the provided documents.
    4. Encourage the user to provide more documents or clarify the question for a more accurate response.

    For questions that don't require additional context:
    1. Respond directly with the information you know, based on the context or general knowledge.

    Keep your response concise and no longer than three sentences. Be mindful to provide a tailored answer only when relevant, and when in doubt, ask for clarification or more details.\n\n
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
