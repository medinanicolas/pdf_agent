from app.core.config import settings
from langchain_neo4j import Neo4jGraph
from functools import lru_cache

@lru_cache(maxsize=1)
def get_neo4j_graph():
    return Neo4jGraph(
        url=settings.NEO4J_URI, 
        username=settings.NEO4J_USERNAME, 
        password=settings.NEO4J_PASSWORD, 
        database=settings.NEO4J_DATABASE
    )