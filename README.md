# Agentic Compliance Review System (DocuRAG)

## Overview

DocuRAG is an AI-powered compliance review agent that analyzes enterprise policy PDFs and structured screening data together. It identifies policy violations, approved exceptions, missing information, and supporting evidence in a way that is easier to audit than a basic retrieval-only system.

The current setup is configured for the EY evaluation dataset under `data/ey_dataset/`.

## Key Features

- Cross-document reasoning across PDF and CSV evidence
- Policy versus record validation
- Violation versus exception classification
- Missing data detection for incomplete decisions
- Source-backed explanations with document references
- LLM reasoning with deterministic fallback

## Architecture

This system is designed as an agentic pipeline rather than a simple RAG flow.

- The agent understands the query and decides which tools to use.
- The PDF tool performs semantic retrieval across indexed policy content.
- The CSV tool analyzes structured records for violations, exceptions, anomalies, and missing fields.
- FAISS stores semantic vectors for fast policy retrieval with metadata preserved.
- The synthesis layer combines tool outputs into a grounded compliance answer.

## Project Layout

```text
agent/         Agent orchestration and synthesis
api/           FastAPI app and response contract
data/          EY dataset plus the saved FAISS index
frontend/      Minimal browser UI
loaders/       PDF and CSV ingestion
processing/    Chunking and embeddings
tools/         PDF search and CSV analysis tools
vectorstore/   FAISS wrapper with metadata and dedupe
run.py         Local server entrypoint
```

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` from the template and add your keys:

```bash
cp .env.example .env
```

Example:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-pro
AGENT_TEMPERATURE=0.1
```

3. Run the server:

```bash
uvicorn api.main:app --reload
```

4. Open:

```text
http://127.0.0.1:8000
```

## Data and Indexing

At startup the system rebuilds `data/index/` from `data/ey_dataset/` only.

Included folders:

- `policies/` for policy PDFs
- `structured/` for CSV records
- `candidate_pack/` for PDFs if present
- `evidence/` for supporting PDFs

Ignored content:

- `expected_outputs/`
- `reference/`
- `README.md`
- `.docx`
- `.xlsx`
- other non-indexed files

## API

Useful endpoints:

- `POST /query`
- `GET /health`
- `GET /sources`
- `POST /rebuild-index`

The `/query` endpoint returns:

```json
{
  "answer": "plain-text compliance summary",
  "violations_detected": true,
  "missing_data": "plain-text missing data summary",
  "sources": [
    {
      "file": "BPSS_Screening_Policy_v3.pdf",
      "type": "pdf",
      "reference": "page 1",
      "snippet": "Policy excerpt..."
    }
  ],
  "reasoning": "plain-text explanation",
  "confidence_score": 0.84
}
```

## Example Queries

- `Which candidates have compliance issues?`
- `Is CAND-101 compliant?`
- `Are there any policy violations or approved exceptions?`
- `What required information is missing from the screening records?`
- `Are there contradictions between policy requirements and adjudication decisions?`

## Design Approach

This system is intentionally built as an agentic AI pipeline rather than a simple retrieval layer.

It dynamically selects tools, reasons across structured and unstructured data, distinguishes policy violations from approved exceptions, and identifies missing information required for decision-making. The result is a workflow that feels closer to a real compliance review assistant than a document search demo.

## Provider Strategy

The system supports multiple LLM providers.

- Groq is the default provider for stable evaluation runs.
- Gemini remains available as a secondary provider.
- If both providers are unavailable or fail, the system falls back to deterministic synthesis instead of crashing.

Note: The system is designed to run locally due to FAISS-based indexing and multi-document processing requirements. Deployment on serverless platforms may not support persistent vector storage.

Current default Groq model:

- `llama-3.1-8b-instant`

Optional larger supported Groq model:

- `llama-3.3-70b-versatile`
