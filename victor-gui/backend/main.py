from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import random

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/victor")
async def victor_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = {
                "load": random.randint(0, 100),
                "tps": random.randint(5, 20),
                "sentiment": random.choice(["calm", "alert", "insight"])
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.2)
    except Exception:
        pass
