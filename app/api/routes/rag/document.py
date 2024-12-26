from fastapi import APIRouter, File, UploadFile, Form, Depends, Response
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import os, hashlib, shutil
from app.core.config import settings
from app.core.database.neo4j import get_neo4j_graph

router = APIRouter()

@router.post('/upload')
async def upload_document(
    file: UploadFile = File(...),
    ):
    try:
        if file.content_type != 'application/pdf':
            return {'error': 'The file must be pdf'}
        file_path = os.path.join(settings.RAG_UPLOAD_DIR, file.filename)

        if not os.path.isdir(settings.RAG_UPLOAD_DIR):
            os.makedirs(settings.RAG_UPLOAD_DIR, exist_ok=True)
        file_hash = hashlib.sha256()
        with open(file_path, 'wb') as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
            while True:
                chunk = await file.read(1024)  # Read in chunks of 1024 bytes
                if not chunk:
                    break
                file_hash.update(chunk)
        pdf_loader = PyPDFLoader(file_path)
        pages = pdf_loader.load()
        
        print(f"generating chunks in {len(pages)} pages")

        text_splitter = SemanticChunker(
            OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL),
            sentence_split_regex ='(?<=[.? !])\\s+'
        )
        chunks = text_splitter.split_documents(pages)

        graphdb = get_neo4j_graph()
        document_id = file_hash.hexdigest()
        graphdb.query(
            '''
            MERGE (d:Document {documentId: $document_id, name: $name})
            ''',
            params={'document_id': document_id, 'name': file.filename}
        )

        graphdb.query(
            '''
            CREATE CONSTRAINT unique_chunk IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
            '''
        )

        graphdb.query(
            '''
            CREATE VECTOR INDEX `embeddingChunks` IF NOT EXISTS
            FOR (c:Chunk) ON (c.textEmbedding)
            OPTIONS { indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                } 
            }
            '''
        )

        for i, chunk in enumerate(chunks):
            chunk_id = f'{file.filename}_{i}'
            graphdb.query(
                '''
                MERGE (c:Chunk {chunkId: $chunk_id}) ON CREATE SET c.text = $chunk
                WITH c MATCH (d:Document {documentId: $document_id}) MERGE (d)-[:HAS_CHUNK]-(c)
                ''',
                params={
                    'document_id': document_id, 
                    'chunk_id': chunk_id, 
                    'chunk': chunk.page_content
                }
            )
        
            graphdb.query("""
            MATCH (d:Document {documentId: $document_id})-[:HAS_CHUNK]->(chunk:Chunk) WHERE chunk.textEmbedding IS NULL
            WITH chunk, genai.vector.encode(
            chunk.text, 
            "OpenAI", 
            {
                token: $openAiApiKey,
                model: $openAiEmbeddingModel,
                endpoint: $openAiEndpoint
            }) AS vector
            CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
            """, 
            params={
                'document_id': document_id,
                'openAiApiKey': settings.OPENAI_API_KEY,
                'openAiEndpoint': settings.OPENAI_EMBEDDINGS_URI,
                'openAiEmbeddingModel': settings.OPENAI_EMBEDDING_MODEL
            }
        )

        return {'filename': file.filename, 'document_id': document_id, 'status': 'Uploaded successfully'}

    except Exception as e:
        # Handle any other errors
        print(f"An unexpected error occurred: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

@router.get('/')
def list_documents():
    try:
        graphdb = get_neo4j_graph()

        result = graphdb.query(
            '''
            MATCH (d:Document)
            RETURN d.documentId AS document_id, d.name AS filename
            '''
        )

        if not result:
            return {"message": "No documents found in the database"}

        return {"documents": result}
    
    except Exception as e:
        return {"error": f"An error occurred while fetching the documents: {str(e)}"}

    


