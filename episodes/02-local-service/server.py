"""Single-process teaching service; not a complete production deployment."""
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from quickstart import QUESTIONS, TICKET, choose_action, load_agent


@asynccontextmanager
async def lifespan(app):
    # Startup work finishes before this process becomes ready.
    app.state.ready = False
    app.state.agent = load_agent()
    app.state.gate = Lock()
    app.state.agent.predict(TICKET, QUESTIONS)
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        del app.state.agent


app = FastAPI(lifespan=lifespan)


class Ticket(BaseModel):
    # Character limit is not a tokenizer limit; test model truncation separately.
    text: str = Field(min_length=1, max_length=1000)


@app.get("/healthz")
def healthz():
    return {"alive": True}


@app.get("/readyz")
def readyz():
    if not getattr(app.state, "ready", False):
        raise HTTPException(503, "Model not ready")
    return {"ready": True}


@app.post("/triage")
def triage(ticket: Ticket):
    # Sync def runs in FastAPI's threadpool; do not block the async event loop.
    if not getattr(app.state, "ready", False):
        raise HTTPException(503, "Model not ready")
    if not app.state.gate.acquire(blocking=False):
        raise HTTPException(503, "Model busy; retry later", headers={"Retry-After": "1"})
    try:
        result = app.state.agent.predict(ticket.text, QUESTIONS)
        return {"answers": result["answers"],
                "action": choose_action(result["answers"])}
    finally:
        app.state.gate.release()
