from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.chat import Chat
from app.agent import graph
import pprint
from fastapi import HTTPException

router = APIRouter()

@router.post("/chat")
async def chat(chat: Chat):
    async def token_stream():
        try:
            events = list(graph.stream({'messages': chat.message}, config={"configurable": {"thread_id": "1", "recursion_limit": 3}}))
            for event in events:
                if 'fallback' in event and 'messages' in event['fallback']:
                    yield event['fallback']['messages'][0].content
                elif 'generate' in event and 'messages' in event['generate']:
                    yield event['generate']['messages'][0].content
                else:
                    print("Not in generation nodes")
                    print(event)
        except Exception as e:
            yield f"Error: {str(e)}\n" 

    return StreamingResponse(token_stream(), media_type="text/plain")
