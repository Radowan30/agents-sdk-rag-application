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
from agents import Agent, Runner, TResponseInputItem
from openai.types.responses import ResponseTextDeltaEvent
from openai import OpenAI

load_dotenv()


THIS_DIR = Path(__file__).parent


# @asynccontextmanager
# async def lifespan(_app: fastapi.FastAPI):
#     async with Database.connect() as db:
#         _app.state.db = db
#         yield {'db': db}


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

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

# result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
# print(result.final_output)


@app.websocket('/async_chat')
async def async_chat(websocket: WebSocket):
    await websocket.accept()
    # # Get the DB instance from the app state
    # database: Database = websocket.app.state.db

    # # Send previous chat messages to the client.
    # previous_msgs = await database.get_messages()
    # for m in previous_msgs:
    #     await websocket.send_text(json.dumps(to_chat_message(m)))
    convo: list[TResponseInputItem] = [] #For having conversational memeory
    
    while True:
        try:
            # Wait for a new prompt from the client.
            prompt = await websocket.receive_text()
            
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

                # Stream messages from the agent.
                # async with agent.run_stream(prompt, message_history=messages, deps=deps) as result:
                    
                #     async for text in result.stream(debounce_by=0.01):
                #         m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                #         await websocket.send_text(
                #             json.dumps(to_chat_message(m))
                #         )
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
                #     await database.add_messages(result.new_messages_json())

    
                # await database.add_messages(result.new_messages_json())
        except Exception as e:
            await websocket.send_text(
                json.dumps({
                    'role': 'model',
                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    'content': f"An error occurred: {e}",
                })
            )
            break

    await websocket.close()

@app.post('/uploadflie/')
async def create_upload_file(file_upload: UploadFile):

    data = await file_upload.read()
    save_to = THIS_DIR / 'uploads' / file_upload.filename
    with open(save_to, 'wb') as f:
        f.write(data)
    
    return {"filenames": file_upload.filename}

# async def get_db(request: Request) -> Database:
#     return request.state.db



# class ChatMessage(TypedDict):
#     """Format of messages sent to the browser."""

#     role: Literal['user', 'model', 'video']
#     timestamp: str
#     content: str


# # def to_chat_message(m: ModelMessage) -> ChatMessage:
# #     first_part = m.parts[0]
# #     if isinstance(m, ModelRequest):
# #         if isinstance(first_part, UserPromptPart):
# #             assert isinstance(first_part.content, str)
# #             return {
# #                 'role': 'user',
# #                 'timestamp': first_part.timestamp.isoformat(),
# #                 'content': first_part.content,
# #             }
# #     elif isinstance(m, ModelResponse):
# #         if isinstance(first_part, TextPart):
# #             content = first_part.content.strip()

# #             return {
# #                 'role': 'model',
# #                 'timestamp': m.timestamp.isoformat(),
# #                 'content': content,
# #             }
 

# P = ParamSpec('P')
# R = TypeVar('R')


# @dataclass
# class Database:
#     """Rudimentary database to store chat messages in SQLite.

#     The SQLite standard library package is synchronous, so we
#     use a thread pool executor to run queries asynchronously.
#     """

#     con: sqlite3.Connection
#     _loop: asyncio.AbstractEventLoop
#     _executor: ThreadPoolExecutor

#     @classmethod
#     @asynccontextmanager
#     async def connect(
#         cls, file: Path = THIS_DIR / '.chat_app_messages.sqlite'
#     ) -> AsyncIterator[Database]:
#         with logfire.span('connect to DB'):
#             loop = asyncio.get_event_loop()
#             executor = ThreadPoolExecutor(max_workers=1)
#             con = await loop.run_in_executor(executor, cls._connect, file)
#             slf = cls(con, loop, executor)
#         try:
#             yield slf
#         finally:
#             await slf._asyncify(con.close)

#     @staticmethod
#     def _connect(file: Path) -> sqlite3.Connection:
#         con = sqlite3.connect(str(file))
#         con = logfire.instrument_sqlite3(con)
#         cur = con.cursor()
#         cur.execute(
#             'CREATE TABLE IF NOT EXISTS messages (id INT PRIMARY KEY, message_list TEXT);'
#         )
#         con.commit()
#         return con

#     async def add_messages(self, messages: bytes):
#         await self._asyncify(
#             self._execute,
#             'INSERT INTO messages (message_list) VALUES (?);',
#             messages,
#             commit=True,
#         )
#         await self._asyncify(self.con.commit)

#     async def get_messages(self) -> list[ModelMessage]:
#         c = await self._asyncify(
#             self._execute, 'SELECT message_list FROM messages order by id'
#         )
#         rows = await self._asyncify(c.fetchall)
#         messages: list[ModelMessage] = []
#         for row in rows:
#             messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
#         return messages

#     def _execute(
#         self, sql: LiteralString, *args: Any, commit: bool = False
#     ) -> sqlite3.Cursor:
#         cur = self.con.cursor()
#         cur.execute(sql, args)
#         if commit:
#             self.con.commit()
#         return cur

#     async def _asyncify(
#         self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
#     ) -> R:
#         return await self._loop.run_in_executor(  # type: ignore
#             self._executor,
#             partial(func, **kwargs),
#             *args,  # type: ignore
#         )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)]
    )