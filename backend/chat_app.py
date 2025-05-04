from __future__ import annotations as _annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import fastapi
from fastapi import Depends, Request, WebSocket, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import LiteralString, ParamSpec, TypedDict
from datetime import datetime
from httpx import AsyncClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Any, Dict, Union
from agents import Agent, Runner, TResponseInputItem, FileSearchTool
from openai.types.responses import ResponseTextDeltaEvent
from openai import OpenAI
import requests
from io import BytesIO
import os
import asyncio
from starlette.websockets import WebSocketDisconnect
from markitdown import MarkItDown

load_dotenv()

THIS_DIR = Path(__file__).parent
UPLOADS_DIR = THIS_DIR / "uploads"


app = fastapi.FastAPI()


origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_credentials= True,
    allow_headers=["*"],
)


client = OpenAI()

Vector_store_id = os.getenv("VECTOR_STORE_ID")

agent = Agent(
    name="Assistant", 
    instructions="""You are a helpful assistant. User's can upload files to your system. If there aren't any 'You uploaded a file called...' messages in your conversation history, then you should assume that the user hasn't uploaded any files yet. 
    You can encourage the user to upload files if they desire, and they can ask questions about their documents. If the user uploads a file, you can answer questions based on it. Otherwise, maintain normal conversation.""",
    tools = [
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=[Vector_store_id],
        )
    ])


events_queue = asyncio.Queue() # Queue to handle file upload events

@app.websocket('/async_chat')
async def async_chat(websocket: WebSocket):
    await websocket.accept()

    convo: list[TResponseInputItem] = [] #For having conversational memeory

    async def message_handler():
        while True:
            # Wait for a new prompt from the client.
            msg = await websocket.receive_text()
            await events_queue.put({"type": "message", "content": msg})
    
    #Start the message handler task in the background
    message_task = asyncio.create_task(message_handler())

    try:

        while True:
            event = await events_queue.get()

            if event["type"] == "file":
                filename = event["filename"]
                file_uploaded_msg = f"You uploaded a file called '{filename}'"
                await websocket.send_text(
                    json.dumps({
                        'role': 'file_upload',
                        'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                        'content': file_uploaded_msg,
                    })
                )
                convo.append({"content": file_uploaded_msg, "role": "assistant"})
            
            elif event["type"] == "message":
                prompt = event["content"]

                # Immediately send back the user prompt.
                await websocket.send_text(
                    json.dumps({
                        'role': 'user',
                        'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                        'content': prompt,
                    })
                )
            
                async with AsyncClient() as client:
                    
                    convo.append({"content": prompt, "role": "user"})

                    result = Runner.run_streamed(agent, convo)

                    response_parts = ""
                    async for event in result.stream_events():
                        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                            # print(event.data.delta, end="", flush=True)
                            response_parts += event.data.delta
                            await websocket.send_text(
                                json.dumps({
                                    'role': 'model',
                                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                                    'content': response_parts,
                                })
                            )
                            
                    convo = result.to_input_list()
    
    except WebSocketDisconnect:
        await websocket.close()

    except Exception as e:
        await websocket.send_text(
            json.dumps({
                'role': 'model',
                'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                'content': f"An error occurred: {e}",
            })
        )
        message_task.cancel()
    
        


@app.post('/uploadflie/')
async def create_upload_file(file_upload: UploadFile):

    data = await file_upload.read()
    file_saved_at = UPLOADS_DIR / file_upload.filename
    with open(file_saved_at, 'wb') as f:
        f.write(data)


    file_to_upload = Path(file_saved_at)

    # Convert Excel file to Markdown if the file is an Excel file
    if file_to_upload.suffix == ".xlsx" or file_to_upload.suffix == ".xls":
        md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
        result = md.convert(file_saved_at)
        md_file_saved_at = UPLOADS_DIR / (file_to_upload.stem + ".md") 
        with open(md_file_saved_at, 'w') as f:
            f.write(result.text_content)

        file_to_upload = Path(md_file_saved_at)

    # Upload file to vector store
    with open(UPLOADS_DIR / file_to_upload.name, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="assistants"
        )
    
    client.vector_stores.files.create(
        vector_store_id=Vector_store_id,
        file_id=result.id
    )
    
    # Add the file upload event to the asyncio queue
    await events_queue.put({"type": "file", "filename": file_upload.filename})

    return {"filenames": file_upload.filename}



if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)]
    )