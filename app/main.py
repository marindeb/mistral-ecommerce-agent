import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import agent, rag_pipeline


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


class QueryRequest(BaseModel):
    question: str
    mode: str | None = None  # optional now


@app.post("/query")
def query_endpoint(request: QueryRequest):
    question = request.question.strip()
    mode = request.mode

    # --- Automatic mode selection ---
    if not mode:
        analytical_keywords = [
            "average",
            "rate",
            "compare",
            "trend",
            "correlation",
            "return",
            "sales",
        ]
        mode = (
            "agent"
            if any(k in question.lower() for k in analytical_keywords)
            else "rag"
        )

    IS_TEST = os.getenv("APP_ENV") == "test"
    try:
        if mode == "rag":
            answer = rag_pipeline.query(question)
        elif mode == "agent":
            answer = agent.ask_agent(question)
        else:
            if not IS_TEST:
                raise HTTPException(status_code=400, detail="Invalid mode")
            else:
                raise ValueError("Invalid mode")  # simple exception for unit tests
        return {"mode": mode, "question": question, "answer": answer}

    except Exception as e:
        if not IS_TEST:
            logging.info(e)
            raise HTTPException(status_code=500, detail=str(e))
        else:
            raise e
