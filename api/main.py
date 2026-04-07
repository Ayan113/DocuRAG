import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from agent.agent import DocuRAGAgent

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("langchain_google_genai.chat_models").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ey_dataset")
INDEX_DIR = os.path.join(BASE_DIR, "data", "index")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

agent: Optional[DocuRAGAgent] = None
agent_ready = False


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=2000)


class SourceInfo(BaseModel):
    file: str = ""
    type: str = ""
    reference: str = ""
    snippet: str = ""


class QueryResponse(BaseModel):
    answer: str
    violations_detected: bool = False
    missing_data: str = "None detected"
    sources: list[SourceInfo] = Field(default_factory=list)
    reasoning: str = ""
    confidence_score: float = 0.0


def build_response(
    *,
    answer: str,
    reasoning: str,
    missing_data: str = "None detected",
    status_code: int = 200,
    violations_detected: bool = False,
    sources: Optional[list] = None,
    confidence_score: float = 0.0,
) -> JSONResponse:
    payload = QueryResponse(
        answer=answer,
        violations_detected=violations_detected,
        missing_data=missing_data,
        sources=sources or [],
        reasoning=reasoning,
        confidence_score=confidence_score,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


app = FastAPI(
    title="DocuRAG API",
    description="Compliance-focused document agent over PDF policies and CSV records.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global agent, agent_ready

    print("[SYSTEM] Initializing agent...")
    logger.info("Starting DocuRAG server")
    agent_ready = False
    agent = None

    try:
        agent = DocuRAGAgent(
            data_dir=DATA_DIR,
            index_dir=INDEX_DIR,
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.1")),
        )
        agent.initialize()
        agent_ready = True
        print("[SYSTEM] Agent fully initialized")
        logger.info("Agent initialized")
    except Exception as exc:
        print("[ERROR] Agent initialization failed:", str(exc))
        logger.exception("Agent initialization failed: %s", exc)
        agent = None
        agent_ready = False

    print("[DEBUG] agent_ready =", agent_ready)


@app.on_event("shutdown")
async def shutdown():
    global agent, agent_ready
    logger.info("Shutting down DocuRAG server")
    agent = None
    agent_ready = False


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    question = request.question.strip()
    if not question:
        return build_response(
            answer="Please provide a question before submitting.",
            reasoning="The /query endpoint received an empty query.",
            missing_data="Question text is missing.",
            status_code=400,
        )

    if not agent_ready or agent is None:
        return build_response(
            answer="System is initializing or failed to start. Please check logs.",
            reasoning="Initialization incomplete or failed",
            missing_data="Agent not ready",
            status_code=503,
        )

    logger.info("Received query: %s", question)

    try:
        result = agent.query(question)
        return JSONResponse(status_code=200, content=QueryResponse(**result).model_dump())
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        return build_response(
            answer="Unable to complete the request.",
            reasoning=f"The agent raised an unexpected error: {exc}",
            missing_data="Result generation failed.",
            status_code=500,
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if agent_ready and agent else "initializing",
        "agent_initialized": bool(agent_ready and agent),
        "llm_available": bool(agent and agent.llm),
        "llm_provider": agent.llm_provider if agent else "deterministic",
        "agent_ready": agent_ready,
    }


@app.get("/sources")
async def list_sources():
    if not agent_ready or agent is None:
        return {"pdf_files": [], "csv_files": [], "index_stats": {}, "llm_available": False}
    return agent.get_sources_info()


@app.post("/rebuild-index")
async def rebuild_index():
    if not agent_ready or agent is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Agent is not initialized."},
        )

    try:
        agent.rebuild_index()
        return {"status": "success", "message": "Index rebuilt successfully."}
    except Exception as exc:
        logger.exception("Index rebuild failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Rebuild failed: {exc}"},
        )


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>DocuRAG API is running.</h1>")

    with open(index_path, "r", encoding="utf-8") as handle:
        return HTMLResponse(handle.read())
