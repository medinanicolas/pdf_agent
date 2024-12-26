qa_prompt_template = """You are an assistant for question-answering tasks. """ \
"""Use the following pieces of retrieved context to answer the question. """ \
"""If you don't know the answer, explain that you need more context. """ \
"""Use three sentences maximum and keep the answer concise. \
Chat History: {chat_history}
Question: {question} 
Context: {context} 
Answer:"""

hallucination_prompt_template = """
     You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.\n\n
     Set of facts: {documents} \n\n LLM generation: {generation}
     """