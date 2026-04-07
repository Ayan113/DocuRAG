import json
import logging
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.tools import Tool

from loaders.csv_loader import CSVLoader
from loaders.pdf_loader import PDFLoader
from processing.chunking import TextChunker
from processing.embeddings import EmbeddingGenerator
from tools.csv_analysis_tool import create_csv_analysis_tool
from tools.pdf_search_tool import create_pdf_search_tool
from vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
logging.getLogger("langchain_google_genai.chat_models").setLevel(logging.ERROR)


SYSTEM_PROMPT = """
You are an enterprise compliance agent reviewing EY policy documents and operational records.

Use the provided evidence to answer the user question with grounded reasoning.

You work in four steps:
1. Understand what the user is asking.
2. Read the evidence from policy PDFs and CSV records.
3. Distinguish policy rules from candidate or operational data.
4. Explain the outcome, including exceptions, missing data, and contradictions.

Rules:
- Base factual claims only on the provided evidence.
- Use multiple sources when available and explain how they fit together.
- Separate policy language from record evidence. Do not treat a candidate record as policy.
- Distinguish valid documented exceptions from unresolved violations.
- If the evidence is incomplete or a requested record is missing, say "Insufficient data to answer conclusively."
- Do not invent citations, policy text, row details, or entity identifiers.

Return valid JSON with this shape:
{
  "answer": "short direct answer",
  "violations_detected": true,
  "missing_data": "summary or None detected",
  "sources": [
    {
      "file": "file name",
      "type": "pdf or csv",
      "reference": "page 2 or row 14",
      "snippet": "short evidence snippet"
    }
  ],
  "reasoning": "concise explanation of how the conclusion follows from the evidence",
  "confidence_score": 0.0
}

Final requirements:
- Mention contradictions and missing data explicitly when present.
- If the question asks about violations versus exceptions, structure the answer as:
  1. Policy Violations
  2. Approved Exceptions
  3. Overall Assessment
- Return the response in plain text only. Do not use markdown formatting, asterisks, or bullet symbols like "*". Use simple "-" for lists.
- If no reliable answer is possible, return "Insufficient data to answer conclusively."
- Output JSON only.
""".strip()


class DocuRAGAgent:
    """Orchestrates retrieval and answer synthesis across PDFs and CSVs."""

    def __init__(
        self,
        data_dir: str,
        index_dir: Optional[str] = None,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.1,
    ):
        self.data_dir = data_dir
        self.index_dir = index_dir or os.path.join(data_dir, "index")
        self.model_name = model
        self.temperature = temperature
        self.groq_model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.llm_provider = "deterministic"

        self.pdf_loader: Optional[PDFLoader] = None
        self.csv_loader: Optional[CSVLoader] = None
        self.chunker: Optional[TextChunker] = None
        self.embedding_gen: Optional[EmbeddingGenerator] = None
        self.faiss_store: Optional[FAISSStore] = None
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.tools: Dict[str, Tool] = {}

        self._is_initialized = False

    def initialize(self) -> None:
        if self._is_initialized:
            return

        logger.info("Initializing DocuRAG agent")

        self.pdf_loader = PDFLoader(self.data_dir)
        self.csv_loader = CSVLoader(self.data_dir)
        self.chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        self.embedding_gen = EmbeddingGenerator()
        self.faiss_store = FAISSStore(dimension=384)
        self._reset_index()
        self._build_index()

        pdf_tool = create_pdf_search_tool(self.faiss_store, self.embedding_gen)
        csv_tool = create_csv_analysis_tool(self.csv_loader)
        self.tools = {
            pdf_tool.name: pdf_tool,
            csv_tool.name: csv_tool,
        }

        self.llm = self._create_llm()

        self._is_initialized = True
        logger.info("Agent ready: %s", self.faiss_store.get_stats())

    def _create_llm(self):
        if self.groq_api_key:
            self.llm_provider = "groq"
            return ChatGroq(
                api_key=self.groq_api_key,
                model=self.groq_model_name,
                temperature=0,
                max_retries=0,
            )

        if self.google_api_key:
            self.llm_provider = "gemini"
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=self.google_api_key,
                max_retries=0,
            )

        self.llm_provider = "deterministic"
        return None

    def _reset_index(self) -> None:
        if os.path.isdir(self.index_dir):
            shutil.rmtree(self.index_dir)
        self.faiss_store = FAISSStore(dimension=384)

    def _build_index(self) -> None:
        if not all([self.pdf_loader, self.csv_loader, self.chunker, self.embedding_gen, self.faiss_store]):
            raise RuntimeError("Agent components are not initialized")

        pdf_documents = self.pdf_loader.load_all()
        csv_documents = self.csv_loader.rows_to_documents()
        documents = pdf_documents + csv_documents
        chunks = self.chunker.chunk_documents(documents)

        if not chunks:
            logger.warning("No documents available for indexing")
            return

        texts = [chunk["text"] for chunk in chunks]
        metadata = [{key: value for key, value in chunk.items() if key != "text"} for chunk in chunks]
        embeddings = self.embedding_gen.embed_texts(texts)

        self.faiss_store.add_documents(texts, embeddings, metadata)
        self.faiss_store.save(self.index_dir)
        print("[INDEX] Total documents indexed:", len(documents))
        self._print_index_summary(documents)

    def _print_index_summary(self, documents: List[Dict[str, Any]]) -> None:
        type_counts = Counter(document.get("document_type", "unknown") for document in documents)
        folder_counts = Counter(document.get("folder_source", "unknown") for document in documents)
        print("[INDEX] Count by type:", dict(type_counts))
        print("[INDEX] Count by folder:", dict(folder_counts))

    def query(self, question: str) -> Dict[str, Any]:
        if not self._is_initialized:
            raise RuntimeError("Agent not initialized")

        cleaned_question = question.strip()
        if not cleaned_question:
            return self._error_response("Please provide a non-empty query.")

        started_at = time.time()
        understanding = self._understand_query(cleaned_question)
        selected_tools = self._select_tools(understanding)
        contexts: Dict[str, Dict[str, Any]] = {}

        logger.info("Question: %s", cleaned_question)
        logger.info("Planned tools: %s", selected_tools)

        for tool_name in selected_tools:
            tool_input = self._build_tool_input(tool_name, cleaned_question, understanding, contexts)
            contexts[tool_name] = self._run_tool(tool_name, tool_input)

        if self._needs_additional_retrieval(contexts):
            for tool_name in self.tools:
                if tool_name in contexts:
                    continue
                tool_input = self._build_tool_input(tool_name, cleaned_question, understanding, contexts)
                contexts[tool_name] = self._run_tool(tool_name, tool_input)

        response = self.synthesize_answer(contexts, cleaned_question)
        _ = round(time.time() - started_at, 2)
        return self._finalize_response(response)

    def _understand_query(self, question: str) -> Dict[str, Any]:
        query = question.lower()

        focus: List[str] = []
        if any(token in query for token in ["cnd-", "cand-"]):
            focus.append("candidate")
        if "exp-" in query:
            focus.append("expense")
        if "aud-" in query:
            focus.append("audit")
        if any(word in query for word in ["expense", "receipt", "meal", "travel", "vendor", "reimburse"]):
            focus.append("expense")
        if any(word in query for word in ["candidate", "hiring", "adjudication", "background", "drug test", "offer", "interview"]):
            focus.append("candidate")
        if any(word in query for word in ["privacy", "audit", "training", "gdpr", "incident", "risk", "access"]):
            focus.append("audit")
        if not focus:
            focus = ["expense", "candidate", "audit"]
        else:
            focus = list(dict.fromkeys(focus))

        asks_for_policy = any(
            word in query
            for word in ["policy", "policies", "rule", "rules", "limit", "limits", "allowed", "exception"]
        )
        asks_for_records = any(
            word in query
            for word in ["record", "records", "candidate", "expense", "audit", "row", "data", "dataset"]
        )
        asks_for_compliance = any(
            word in query
            for word in ["compliance", "compliant", "violation", "violations", "breach", "risk", "missing", "contradiction", "issue", "issues"]
        )

        needs_policy = asks_for_policy or asks_for_compliance
        needs_records = asks_for_records or asks_for_compliance or not asks_for_policy
        needs_cross_source = needs_policy and needs_records

        policy_hints = {
            "expense": [],
            "candidate": ["bpss screening policy", "screening operations sop", "adjudication register"],
            "audit": [],
        }

        hints: List[str] = []
        for area in focus:
            hints.extend(policy_hints.get(area, []))

        return {
            "focus": focus,
            "needs_policy": needs_policy,
            "needs_records": needs_records,
            "needs_cross_source": needs_cross_source,
            "policy_hints": sorted(set(hints)),
        }

    def _select_tools(self, understanding: Dict[str, Any]) -> List[str]:
        tools: List[str] = []

        if understanding["needs_records"]:
            tools.append("csv_analysis")
        if understanding["needs_policy"]:
            tools.append("pdf_search")

        if not tools:
            tools = ["csv_analysis", "pdf_search"]

        if understanding["needs_cross_source"]:
            return ["csv_analysis", "pdf_search"]

        return tools

    def _build_tool_input(
        self,
        tool_name: str,
        question: str,
        understanding: Dict[str, Any],
        contexts: Dict[str, Dict[str, Any]],
    ) -> str:
        if tool_name != "pdf_search":
            return question

        hints = set(understanding.get("policy_hints", []))
        csv_context = contexts.get("csv_analysis", {})
        for hint in csv_context.get("policy_hints", []):
            hints.add(hint)

        if not hints:
            return question

        return f"{question}\nRelevant policies: {', '.join(sorted(hints))}"

    def _run_tool(self, tool_name: str, tool_input: str) -> Dict[str, Any]:
        tool = self.tools[tool_name]
        print("[AGENT] Tool:", tool_name)
        raw_output = tool.func(tool_input)
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return {"status": "error", "message": f"Invalid JSON from {tool_name}", "raw_output": raw_output}

    def _needs_additional_retrieval(self, contexts: Dict[str, Dict[str, Any]]) -> bool:
        if not contexts:
            return False

        pdf_matches = len(contexts.get("pdf_search", {}).get("matches", []))
        csv_hits = (
            len(contexts.get("csv_analysis", {}).get("findings", []))
            + len(contexts.get("csv_analysis", {}).get("search_results", []))
        )
        return pdf_matches == 0 and csv_hits == 0

    def synthesize_answer(self, contexts: Dict[str, Any], question: str) -> Dict[str, Any]:
        normalized = self._normalize_contexts(contexts)
        target_tokens = self._extract_target_tokens(question)

        if not normalized["sources"]:
            return self._insufficient_data_response("No relevant data was retrieved for this query.")
        if target_tokens and not normalized["findings"] and not normalized["search_results"]:
            requested = ", ".join(token.upper() for token in target_tokens)
            return self._insufficient_data_response(f"No matching record was found for {requested}.")

        print("[REASONING] Generating answer")

        if self.llm is None:
            return self._fallback_synthesis(question, normalized, api_key_missing=True)

        try:
            prompt = self._build_synthesis_prompt(question, normalized)
            result = self.llm.invoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)
            parsed = self._parse_json_block(content)
            if parsed is None:
                raise ValueError("Model returned non-JSON output")
            return self._coerce_response(parsed, normalized, question)
        except Exception as exc:
            logger.warning("LLM synthesis failed: %s", self._safe_error_message(exc))
            return self._fallback_synthesis(question, normalized, api_key_missing=False)

    def _normalize_contexts(self, contexts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        csv_context = contexts.get("csv_analysis", {})
        pdf_context = contexts.get("pdf_search", {})

        findings = csv_context.get("findings", [])
        search_results = csv_context.get("search_results", [])
        missing_data = csv_context.get("missing_data", [])
        anomalies = csv_context.get("anomalies", [])
        policy_matches = pdf_context.get("matches", [])

        sources: List[Dict[str, str]] = []
        for match in policy_matches:
            sources.append(
                self._build_source(
                    file_name=match.get("file_name"),
                    document_type=match.get("document_type", "pdf"),
                    page_number=match.get("page_number"),
                    row_index=match.get("row_index"),
                    snippet=match.get("snippet", ""),
                )
            )

        for finding in findings:
            sources.append(
                self._build_source(
                    file_name=finding.get("file_name"),
                    document_type=finding.get("document_type", "csv"),
                    page_number=finding.get("page_number"),
                    row_index=finding.get("row_index"),
                    entity_name=finding.get("entity"),
                    snippet=finding.get("summary", ""),
                )
            )

        for item in search_results:
            sources.append(
                self._build_source(
                    file_name=item.get("file_name"),
                    document_type=item.get("document_type", "csv"),
                    page_number=item.get("page_number"),
                    row_index=item.get("row_index"),
                    entity_name=item.get("entity_name"),
                    snippet=item.get("snippet", ""),
                )
            )

        deduped_sources = self._dedupe_sources(sources)[:10]
        contradiction_notes = self._detect_contradictions(findings)

        return {
            "findings": findings,
            "search_results": search_results,
            "missing_data": missing_data,
            "anomalies": anomalies,
            "policy_matches": policy_matches,
            "sources": deduped_sources,
            "contradictions": contradiction_notes,
        }

    def _build_source(
        self,
        *,
        file_name: Optional[str],
        document_type: str,
        page_number: Optional[int],
        row_index: Optional[int],
        entity_name: Optional[str] = None,
        snippet: str,
    ) -> Dict[str, str]:
        reference = self._format_reference(
            file_name=file_name,
            document_type=document_type,
            page_number=page_number,
            row_index=row_index,
            entity_name=entity_name,
        )

        return {
            "file": file_name or "unknown",
            "type": document_type,
            "reference": reference,
            "snippet": snippet[:240],
        }

    def _format_reference(
        self,
        *,
        file_name: Optional[str],
        document_type: str,
        page_number: Optional[int],
        row_index: Optional[int],
        entity_name: Optional[str],
    ) -> str:
        if page_number:
            return f"page {page_number}"

        if not row_index:
            return "n/a"

        clean_name = str(entity_name).strip() if entity_name else ""
        if not clean_name:
            return f"row {row_index}"

        if file_name in {"bpps_tracker_export.csv", "document_inventory.csv", "employment_history.csv"}:
            return f"Candidate {clean_name} (row {row_index})"
        return f"{clean_name} (row {row_index})"

    def _finding_reference(self, finding: Dict[str, Any]) -> str:
        return self._format_reference(
            file_name=finding.get("file_name"),
            document_type=finding.get("document_type", "csv"),
            page_number=finding.get("page_number"),
            row_index=finding.get("row_index"),
            entity_name=finding.get("entity"),
        )

    def _dedupe_sources(self, sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        deduped: List[Dict[str, str]] = []
        for source in sources:
            key = (source["file"], source["type"], source["reference"], source["snippet"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        return deduped

    def _detect_contradictions(self, findings: List[Dict[str, Any]]) -> List[str]:
        grouped: Dict[str, set[str]] = {}
        for finding in findings:
            entity = finding.get("entity")
            finding_type = finding.get("finding_type")
            if not entity or not finding_type:
                continue
            grouped.setdefault(entity, set()).add(finding_type)

        contradictions = []
        for entity, types in grouped.items():
            if {"violation", "exception"}.issubset(types):
                contradictions.append(
                    f"{entity} has both exception handling evidence and unresolved violations."
                )
        return contradictions

    def _build_synthesis_prompt(self, question: str, normalized: Dict[str, Any]) -> str:
        evidence = {
            "policy_matches": normalized["policy_matches"][:6],
            "findings": normalized["findings"][:12],
            "search_results": normalized["search_results"][:6],
            "missing_data": normalized["missing_data"][:10],
            "anomalies": normalized["anomalies"][:8],
            "contradictions": normalized["contradictions"],
        }

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{json.dumps(evidence, indent=2, default=str)}\n"
        )

    def _parse_json_block(self, content: Any) -> Optional[Dict[str, Any]]:
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)

        text = str(content)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    def _coerce_response(
        self,
        response: Dict[str, Any],
        normalized: Dict[str, Any],
        question: str,
    ) -> Dict[str, Any]:
        answer = self._answer_summary(question, normalized)
        missing_summary = response.get("missing_data", "").strip() or self._missing_data_summary(normalized["missing_data"])

        return {
            "answer": answer,
            "violations_detected": bool(response.get("violations_detected", self._has_violations(normalized["findings"]))),
            "missing_data": missing_summary,
            "sources": self._normalize_sources(response.get("sources") or normalized["sources"]),
            "reasoning": response.get("reasoning", "").strip() or self._fallback_reasoning(normalized),
            "confidence_score": float(response.get("confidence_score", self._confidence_score(normalized))),
        }

    def _fallback_synthesis(
        self,
        question: str,
        normalized: Dict[str, Any],
        *,
        api_key_missing: bool,
    ) -> Dict[str, Any]:
        violations = [item for item in normalized["findings"] if item.get("finding_type") == "violation"]
        missing_summary = self._missing_data_summary(normalized["missing_data"])
        query = question.lower()

        if any(term in query for term in ["missing", "incomplete", "decision making"]):
            answer = (
                f"Key missing data: {missing_summary}."
                if missing_summary != "None detected"
                else "No material missing data was detected in the retrieved records."
            )
            if violations:
                answer += " The records also contain policy issues, but the missing fields are the main blocker for confident decisions."
            if api_key_missing:
                answer += " Response was synthesized with the deterministic fallback because no LLM provider was available."
            return {
                "answer": answer,
                "violations_detected": self._has_violations(normalized["findings"]),
                "missing_data": missing_summary,
                "sources": normalized["sources"],
                "reasoning": self._fallback_reasoning(normalized),
                "confidence_score": self._confidence_score(normalized),
            }

        if "contradiction" in query:
            if normalized["contradictions"]:
                answer = "Contradictions were found between exception handling evidence and unresolved violations."
            else:
                answer = "No direct contradiction pattern was confirmed in the retrieved evidence."
                if violations:
                    answer += " The records still contain policy violations that need review."
            if api_key_missing:
                answer += " Response was synthesized with the deterministic fallback because no LLM provider was available."
            return {
                "answer": answer,
                "violations_detected": self._has_violations(normalized["findings"]),
                "missing_data": missing_summary,
                "sources": normalized["sources"],
                "reasoning": self._fallback_reasoning(normalized),
                "confidence_score": self._confidence_score(normalized),
            }

        answer = self._answer_summary(question, normalized)

        if api_key_missing:
            answer += " Response was synthesized with the deterministic fallback because no LLM provider was available."

        return {
            "answer": answer,
            "violations_detected": self._has_violations(normalized["findings"]),
            "missing_data": missing_summary,
            "sources": normalized["sources"],
            "reasoning": self._fallback_reasoning(normalized),
            "confidence_score": self._confidence_score(normalized),
        }

    def _fallback_reasoning(self, normalized: Dict[str, Any]) -> str:
        parts = []

        if normalized["policy_matches"]:
            parts.append(
                f"Reviewed {len(normalized['policy_matches'])} policy excerpt(s) from the PDF corpus."
            )
        if normalized["findings"]:
            top_findings = "; ".join(
                f"{item['title']} for {self._finding_reference(item)}"
                for item in normalized["findings"][:4]
            )
            parts.append(f"Structured record analysis surfaced: {top_findings}.")
        if normalized["missing_data"]:
            parts.append(self._missing_data_summary(normalized["missing_data"]))
        if normalized["contradictions"]:
            parts.append("Potential contradictions: " + " ".join(normalized["contradictions"]))

        return " ".join(parts) or "Insufficient data to answer conclusively."

    def _answer_summary(self, question: str, normalized: Dict[str, Any]) -> str:
        _ = question
        return self._structured_answer_summary(normalized)

    def _is_violation_exception_query(self, question: str) -> bool:
        query = question.lower()
        return any(
            phrase in query
            for phrase in [
                "violation or exception",
                "violation vs exception",
                "policy violation",
                "violations vs exceptions",
                "violations or exceptions",
                "policy violations or exceptions",
                "approved exceptions",
                "policy violations",
                "exception",
                "exceptions",
            ]
        )

    def _structured_answer_summary(self, normalized: Dict[str, Any]) -> str:
        violations = [item for item in normalized["findings"] if item.get("finding_type") == "violation"]
        exceptions = [item for item in normalized["findings"] if item.get("finding_type") == "exception"]

        if not violations and not exceptions and not normalized["missing_data"]:
            return "Insufficient data to answer conclusively."

        lines = [
            "Answer:",
            "",
            self._answer_overview_line(violations, exceptions, normalized),
            "",
            "1. Policy Violations:",
            "",
        ]

        if violations:
            lines.append(f"- {self._violation_section_summary(violations)}")
            for example in self._finding_assessments(violations, limit=3):
                lines.append(f"- {example}")
        else:
            lines.append("- No unresolved policy breaches were supported by the retrieved records.")

        lines.extend(
            [
                "",
                "2. Approved Exceptions:",
                "",
            ]
        )

        if exceptions:
            lines.append(f"- {self._exception_section_summary(exceptions)}")
            for example in self._finding_assessments(exceptions, limit=2):
                lines.append(f"- {example}")
        else:
            lines.append("- No approved exception cases were supported by the retrieved records.")

        lines.extend(
            [
                "",
                "3. Overall Assessment:",
                "",
                f"- {self._overall_violation_exception_assessment(violations, exceptions, normalized)}",
            ]
        )

        if normalized["missing_data"]:
            lines.append(f"- Key missing information: {self._missing_data_summary(normalized['missing_data'])}.")

        return "\n".join(lines).strip()

    def _answer_overview_line(
        self,
        violations: List[Dict[str, Any]],
        exceptions: List[Dict[str, Any]],
        normalized: Dict[str, Any],
    ) -> str:
        violation_count = len(violations)
        exception_count = len(exceptions)

        if violation_count and exception_count:
            return (
                f"Yes - the review identified {violation_count} unresolved policy issue"
                f"{'s' if violation_count != 1 else ''} and {exception_count} approved exception"
                f"{'s' if exception_count != 1 else ''}."
            )
        if violation_count:
            return (
                f"Yes - the review identified {violation_count} unresolved policy issue"
                f"{'s' if violation_count != 1 else ''}."
            )
        if exception_count:
            return (
                f"No confirmed open policy breach was established, but the review identified {exception_count} approved exception"
                f"{'s' if exception_count != 1 else ''}."
            )
        if normalized["missing_data"]:
            return "No confirmed policy violation or approved exception was established, but material data gaps limit the assessment."

        return "No - the retrieved evidence did not establish a policy violation or an approved exception."

    def _violation_exception_summary(self, normalized: Dict[str, Any]) -> str:
        return self._structured_answer_summary(normalized)

    def _violation_section_summary(self, violations: List[Dict[str, Any]]) -> str:
        issue_counter = Counter(item.get("title", "policy gap") for item in violations)
        top_issue, _ = issue_counter.most_common(1)[0]
        return (
            f"The main policy exposure comes from {top_issue.lower()}, which leaves mandatory screening controls open "
            "when the policy expects completion before clearance."
        )

    def _exception_section_summary(self, exceptions: List[Dict[str, Any]]) -> str:
        if len(exceptions) == 1:
            return (
                "A smaller set of records follows the exception path, where the case appears to have been escalated "
                "and handled through documented risk acceptance rather than treated as a standard pass."
            )

        return (
            "A smaller group of records follows the exception path, suggesting formal escalation and risk acceptance "
            "rather than routine screening closure."
        )

    def _finding_assessments(self, findings: List[Dict[str, Any]], limit: int) -> List[str]:
        ranked_items = sorted(
            findings,
            key=lambda item: (
                self._severity_rank(item.get("severity")),
                item.get("row_index", 0),
            ),
            reverse=True,
        )

        assessments: List[str] = []
        seen = set()

        for item in ranked_items:
            if len(assessments) >= limit:
                break

            reference = self._finding_reference(item)
            summary = str(item.get("summary", "")).strip().rstrip(".")
            rule = str(item.get("rule", "")).strip().rstrip(".")

            if item.get("finding_type") == "exception":
                assessment = (
                    f"{reference}: {summary}. This is treated as an exception path because {rule.lower()}."
                )
            else:
                assessment = (
                    f"{reference}: {summary}. This conflicts with policy because {rule.lower()}."
                )

            if assessment in seen:
                continue

            seen.add(assessment)
            assessments.append(assessment)

        return assessments

    def _overall_violation_exception_assessment(
        self,
        violations: List[Dict[str, Any]],
        exceptions: List[Dict[str, Any]],
        normalized: Dict[str, Any],
    ) -> str:
        violation_count = len(violations)
        exception_count = len(exceptions)
        highest_violation_severity = max(
            (self._severity_rank(item.get("severity")) for item in violations),
            default=0,
        )

        if violation_count and exception_count:
            if violation_count >= exception_count:
                assessment = (
                    f"Policy violations are more frequent than approved exceptions ({violation_count} versus {exception_count}) "
                    "and they carry the higher operational risk because they leave mandatory controls incomplete."
                )
            else:
                assessment = (
                    f"Approved exceptions are more frequent than open violations ({exception_count} versus {violation_count}), "
                    "but the unresolved violations remain the riskier population because they do not show the same level of formal escalation."
                )
        elif violation_count:
            assessment = (
                f"The risk profile is driven by policy violations ({violation_count} case{'s' if violation_count != 1 else ''}), "
                "with no supported approved exceptions in the retrieved evidence."
            )
        else:
            assessment = (
                f"The retrieved evidence points to approved exceptions ({exception_count} case{'s' if exception_count != 1 else ''}) "
                "rather than open policy breaches."
            )

        if highest_violation_severity >= 3:
            assessment += " The most serious items involve mandatory checks that should have been complete before clearance."
        if normalized["contradictions"]:
            assessment += " Some records still show mixed signals between exception handling and unresolved issues, so they warrant manual review."

        return assessment

    def _executive_summary(
        self,
        normalized: Dict[str, Any],
        *,
        exceptions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        categories = self._group_findings(normalized)
        non_empty_categories = [
            (name, items)
            for name, items in categories.items()
            if items
        ]

        if not non_empty_categories:
            return "Insufficient data to answer conclusively."

        section_order = [
            "Expense Policy Violations",
            "Hiring Compliance Issues",
            "Data/Privacy Issues",
            "Missing Data Issues",
        ]
        ordered_categories = [
            (name, categories[name])
            for name in section_order
            if categories.get(name)
        ]

        opening = (
            f"The biggest compliance issues across the dataset fall into {len(ordered_categories)} main categories:"
            if len(ordered_categories) > 1
            else "The main issue surfaced in the dataset is:"
        )

        lines = [opening, ""]
        example_budget = 4
        category_sizes = {
            name: len(items)
            for name, items in ordered_categories
        }

        for index, (category_name, items) in enumerate(ordered_categories, start=1):
            lines.append(f"{index}. {category_name}")
            lines.append("")
            lines.append(f"   * {self._category_summary(category_name, items)}")

            if example_budget > 0:
                example_count = 1 if len(ordered_categories) >= 3 else min(2, example_budget)
                examples = self._representative_examples(items, limit=example_count)
                if examples:
                    lines.append(f"   * Example: {examples[0]}")
                    example_budget -= 1
                    for extra_example in examples[1:]:
                        if example_budget <= 0:
                            break
                        lines.append(f"   * Example: {extra_example}")
                        example_budget -= 1

            lines.append("")

        largest_category = max(category_sizes, key=category_sizes.get)
        closing = f"Overall, {largest_category.lower()} are the most frequent issue area."
        if exceptions:
            closing += " A small subset of hiring records also appear to be documented exceptions rather than unresolved violations."
        if normalized["contradictions"]:
            closing += " Some records contain mixed signals, so the reasoning section should be used for adjudication detail."
        lines.append(closing)

        return "\n".join(lines).strip()

    def _group_findings(self, normalized: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {
            "Expense Policy Violations": [],
            "Hiring Compliance Issues": [],
            "Data/Privacy Issues": [],
            "Missing Data Issues": [],
        }

        for finding in normalized["findings"]:
            folder_source = finding.get("folder_source")
            if folder_source == "structured":
                grouped["Hiring Compliance Issues"].append(finding)

        for item in normalized["missing_data"]:
            grouped["Missing Data Issues"].append(item)

        return grouped

    def _category_summary(self, category_name: str, items: List[Dict[str, Any]]) -> str:
        if category_name == "Missing Data Issues":
            return self._missing_data_category_summary(items)

        issue_counter = Counter(item.get("title", "Unspecified issue") for item in items)
        top_issues = [title.lower() for title, _ in issue_counter.most_common(2)]
        if not top_issues:
            return "No material issues were identified."

        joined_issues = " and ".join(top_issues)
        if category_name == "Expense Policy Violations":
            return f"Frequent issues include {joined_issues}."
        if category_name == "Hiring Compliance Issues":
            return f"The main hiring risks involve {joined_issues}."
        if category_name == "Data/Privacy Issues":
            return f"The most common audit concerns involve {joined_issues}."
        return f"The main issues involve {joined_issues}."

    def _missing_data_category_summary(self, items: List[Dict[str, Any]]) -> str:
        field_counter = Counter()
        for item in items:
            field_counter.update(item.get("missing_fields", []))

        common_fields = [field.replace("_", " ") for field, _ in field_counter.most_common(2)]
        if not common_fields:
            return "Several records are missing mandatory fields, which limits decision-making."

        if len(common_fields) == 1:
            field_text = common_fields[0]
        else:
            field_text = f"{common_fields[0]} and {common_fields[1]}"

        return (
            f"Several records lack mandatory fields such as {field_text}. "
            "This limits decision-making and increases compliance risk."
        )

    def _representative_examples(self, items: List[Dict[str, Any]], limit: int = 1) -> List[str]:
        ranked_items = sorted(
            items,
            key=lambda item: (
                self._severity_rank(item.get("severity")),
                item.get("row_index", 0),
            ),
            reverse=True,
        )

        examples: List[str] = []
        seen = set()
        for item in ranked_items:
            if len(examples) >= limit:
                break

            if "missing_fields" in item:
                reference = self._format_reference(
                    file_name=item.get("file_name"),
                    document_type=item.get("document_type", "csv"),
                    page_number=item.get("page_number"),
                    row_index=item.get("row_index"),
                    entity_name=item.get("entity_name"),
                )
                fields = ", ".join(item.get("missing_fields", [])[:2]).replace("_", " ")
                example = f"{reference} is missing {fields}"
            else:
                reference = self._finding_reference(item)
                example = f"{reference} shows {item.get('title', 'an issue').lower()}"

            if example in seen:
                continue
            seen.add(example)
            examples.append(example)

        return examples

    def _severity_rank(self, severity: Optional[str]) -> int:
        order = {
            "Critical": 4,
            "High": 3,
            "Medium": 2,
            "Low": 1,
        }
        return order.get(str(severity), 0)

    def _missing_data_summary(self, missing_data: List[Dict[str, Any]]) -> str:
        if not missing_data:
            return "None detected"

        samples = []
        for item in missing_data[:4]:
            missing_fields = ", ".join(item.get("missing_fields", []))
            reference = self._format_reference(
                file_name=item.get("file_name"),
                document_type=item.get("document_type", "csv"),
                page_number=item.get("page_number"),
                row_index=item.get("row_index"),
                entity_name=item.get("entity_name"),
            )
            samples.append(f"{reference}: {missing_fields}")
        return "Missing required fields in " + "; ".join(samples)

    def _has_violations(self, findings: List[Dict[str, Any]]) -> bool:
        return any(item.get("finding_type") == "violation" for item in findings)

    def _confidence_score(self, normalized: Dict[str, Any]) -> float:
        source_bonus = min(len(normalized["sources"]) * 0.05, 0.25)
        cross_source_bonus = 0.2 if normalized["policy_matches"] and normalized["findings"] else 0.0
        contradiction_penalty = 0.15 if normalized["contradictions"] else 0.0
        missing_penalty = 0.1 if normalized["missing_data"] else 0.0
        score = 0.55 + source_bonus + cross_source_bonus - contradiction_penalty - missing_penalty
        return round(max(0.2, min(0.95, score)), 2)

    def _error_response(self, message: str) -> Dict[str, Any]:
        return self._finalize_response({
            "answer": message,
            "violations_detected": False,
            "missing_data": "Unable to assess",
            "sources": [],
            "reasoning": message,
            "confidence_score": 0.0,
        })

    def _insufficient_data_response(self, message: str) -> Dict[str, Any]:
        return self._finalize_response({
            "answer": "Insufficient data to answer conclusively.",
            "violations_detected": False,
            "missing_data": message,
            "sources": [],
            "reasoning": message,
            "confidence_score": 0.2,
        })

    def _normalize_sources(self, sources: Any) -> List[Dict[str, str]]:
        normalized_sources: List[Dict[str, str]] = []
        if not isinstance(sources, list):
            return normalized_sources

        for source in sources:
            if not isinstance(source, dict):
                continue
            normalized_sources.append(
                {
                    "file": self._clean_output(str(source.get("file", "unknown"))),
                    "type": self._clean_output(str(source.get("type", "unknown"))),
                    "reference": self._clean_output(str(source.get("reference", "n/a"))),
                    "snippet": self._clean_output(str(source.get("snippet", ""))[:240]),
                }
            )
        return normalized_sources

    def _extract_target_tokens(self, question: str) -> List[str]:
        return [token.lower() for token in re.findall(r"[A-Za-z]{2,}-\d+", question)]

    def _clean_output(self, text: str) -> str:
        cleaned = str(text)
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"(?<!\*)\*(.*?)\*(?!\*)", r"\1", cleaned)
        cleaned = re.sub(r"^\s*[\*\u2022]\s+", "- ", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.replace("*", "")
        cleaned = cleaned.replace("•", "-")
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _safe_error_message(self, error: Exception) -> str:
        message = str(error)
        if self.groq_api_key:
            message = message.replace(self.groq_api_key, "[REDACTED]")
        if self.google_api_key:
            message = message.replace(self.google_api_key, "[REDACTED]")
        return message

    def _finalize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "answer": self._clean_output(
                str(response.get("answer", "Insufficient data to answer conclusively.")).strip()
                or "Insufficient data to answer conclusively."
            ),
            "violations_detected": bool(response.get("violations_detected", False)),
            "missing_data": self._clean_output(
                str(response.get("missing_data", "None detected")).strip() or "None detected"
            ),
            "sources": self._normalize_sources(response.get("sources", [])),
            "reasoning": self._clean_output(
                str(response.get("reasoning", "")).strip()
                or "Insufficient data to answer conclusively."
            ),
            "confidence_score": max(0.0, min(1.0, float(response.get("confidence_score", 0.0)))),
        }

    def get_sources_info(self) -> Dict[str, Any]:
        return {
            "pdf_files": self.pdf_loader.get_file_list() if self.pdf_loader else [],
            "csv_files": self.csv_loader.get_file_list() if self.csv_loader else [],
            "index_stats": self.faiss_store.get_stats() if self.faiss_store else {},
            "llm_available": bool(self.llm),
            "llm_provider": self.llm_provider,
        }

    def rebuild_index(self) -> None:
        logger.info("Rebuilding FAISS index")
        self._reset_index()
        self._build_index()
