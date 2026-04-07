import json
import re
from typing import TYPE_CHECKING, Dict, List

import pandas as pd
from langchain.tools import Tool

if TYPE_CHECKING:
    from loaders.csv_loader import CSVLoader


FOCUS_FILE_MAP = {
    "expense": [],
    "candidate": ["bpps_tracker_export.csv", "document_inventory.csv", "employment_history.csv"],
    "audit": [],
}

POLICY_HINTS = {
    "expense": [],
    "candidate": [
        "bpss screening policy",
        "screening operations sop",
        "adjudication register",
    ],
    "audit": [],
}


def create_csv_analysis_tool(csv_loader: "CSVLoader") -> Tool:
    """Builds the structured CSV analysis tool."""

    def _analyze_csv(query: str) -> str:
        try:
            focus = _detect_focus(query)
            file_names = _select_files(csv_loader, focus)
            include_missing = _should_check_missing(query)
            include_anomalies = _should_check_anomalies(query)
            target_tokens = _extract_target_tokens(query)

            findings: List[dict] = []
            if "candidate" in focus or not focus:
                findings.extend(_detect_candidate_findings(csv_loader))

            search_results = _search_relevant_rows(csv_loader, file_names, query)
            missing_data = _collect_missing_data(csv_loader, file_names) if include_missing else []
            anomalies = _detect_anomalies(csv_loader, file_names) if include_anomalies else []

            if target_tokens:
                findings = _filter_findings_by_targets(findings, target_tokens)
                search_results = _filter_rows_by_targets(search_results, target_tokens)
                missing_data = _filter_rows_by_targets(missing_data, target_tokens)
                anomalies = _filter_rows_by_targets(anomalies, target_tokens)

            response = {
                "status": "success",
                "query": query,
                "focus": focus,
                "datasets": [_dataset_summary(csv_loader, file_name) for file_name in file_names],
                "filters_applied": _extract_filters(csv_loader, file_names, query),
                "findings": findings,
                "missing_data": missing_data[:20],
                "anomalies": anomalies[:20],
                "search_results": search_results[:20],
                "policy_hints": _collect_policy_hints(focus),
            }

            print("[RETRIEVAL] Docs:", len(findings) + len(search_results))
            return json.dumps(response, indent=2, default=str)
        except Exception as exc:
            return json.dumps(
                {
                    "status": "error",
                    "query": query,
                    "message": f"CSV analysis failed: {exc}",
                    "findings": [],
                    "missing_data": [],
                    "anomalies": [],
                    "search_results": [],
                }
            )

    return Tool(
        name="csv_analysis",
        func=_analyze_csv,
        description=(
            "Analyze EY structured BPSS screening data for candidate compliance issues, "
            "missing documents, weak employment evidence, and risk-accepted exceptions."
        ),
    )


def _detect_focus(query: str) -> List[str]:
    query_lower = query.lower()
    focus: List[str] = []

    if any(token in query_lower for token in ["cnd-", "cand-"]):
        focus.append("candidate")
    if any(
        word in query_lower
        for word in [
            "candidate",
            "hiring",
            "adjudication",
            "background",
            "screening",
            "employment",
            "right to work",
            "rtw",
            "identity",
            "document",
            "bpps",
            "bpss",
        ]
    ):
        focus.append("candidate")

    if any(
        word in query_lower
        for word in ["audit", "privacy", "gdpr", "expense", "receipt", "meal", "travel", "vendor", "reimburse"]
    ):
        focus.append("audit" if "audit" in query_lower or "privacy" in query_lower or "gdpr" in query_lower else "expense")

    if not focus:
        focus = ["candidate"]

    return list(dict.fromkeys(focus))


def _extract_target_tokens(query: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z]{2,}-\d+", query)]


def _select_files(csv_loader: "CSVLoader", focus: List[str]) -> List[str]:
    file_names: List[str] = []
    for area in focus:
        file_names.extend(FOCUS_FILE_MAP.get(area, []))

    available = set(csv_loader.get_file_list())
    selected = [file_name for file_name in file_names if file_name in available]
    return selected or csv_loader.get_file_list()


def _should_check_missing(query: str) -> bool:
    query_lower = query.lower()
    return any(word in query_lower for word in ["missing", "incomplete", "blank", "null", "required", "compliance", "decision"])


def _should_check_anomalies(query: str) -> bool:
    query_lower = query.lower()
    return any(word in query_lower for word in ["anomaly", "outlier", "unusual", "abnormal", "risk", "compliance"])


def _dataset_summary(csv_loader: "CSVLoader", file_name: str) -> Dict[str, object]:
    summary = csv_loader.get_summary(file_name)
    return {
        "file_name": summary.get("file_name", file_name),
        "folder_source": summary.get("folder_source", "structured"),
        "rows": summary.get("rows", 0),
        "columns": len(summary.get("columns", [])),
    }


def _extract_filters(csv_loader: "CSVLoader", file_names: List[str], query: str) -> List[dict]:
    query_lower = query.lower()
    filters: List[dict] = []
    tracked_columns = {
        "candidate_id",
        "candidate_name",
        "status_tracker",
        "risk_level",
        "evidence_status",
        "doc_type",
    }

    for file_name in file_names:
        dataframe = csv_loader.get_dataframe(file_name)
        if dataframe is None:
            continue

        for column in tracked_columns.intersection(set(dataframe.columns)):
            unique_values = dataframe[column].dropna().astype(str).unique().tolist()
            for value in unique_values:
                cleaned_value = value.strip()
                if cleaned_value and cleaned_value.lower() in query_lower:
                    filters.append(
                        {
                            "file_name": file_name,
                            "column": column,
                            "value": cleaned_value,
                        }
                    )

    return filters


def _search_relevant_rows(csv_loader: "CSVLoader", file_names: List[str], query: str) -> List[dict]:
    results: List[dict] = []

    for file_name in file_names:
        matches = csv_loader.search_rows(file_name, query)
        if matches.empty:
            continue

        for _, row in matches.head(6).iterrows():
            record = csv_loader.row_to_record(file_name, row)
            snippet = " | ".join(
                f"{key}: {value}"
                for key, value in record["record"].items()
                if value not in (None, "")
            )
            results.append(
                {
                    "file_name": record["file_name"],
                    "document_type": record["document_type"],
                    "folder_source": record["folder_source"],
                    "page_number": record["page_number"],
                    "row_index": record["row_index"],
                    "entity_name": record.get("entity_name") or _extract_entity_name(record["record"]),
                    "snippet": snippet[:700],
                }
            )

    return results


def _collect_missing_data(csv_loader: "CSVLoader", file_names: List[str]) -> List[dict]:
    missing: List[dict] = []
    for file_name in file_names:
        missing.extend(csv_loader.detect_missing_fields(file_name))
    return missing


def _detect_anomalies(csv_loader: "CSVLoader", file_names: List[str]) -> List[dict]:
    anomalies: List[dict] = []

    for file_name in file_names:
        dataframe = csv_loader.get_dataframe(file_name)
        if dataframe is None:
            continue

        for column in dataframe.select_dtypes(include=["number"]).columns:
            series = dataframe[column].dropna()
            if len(series) < 4:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = dataframe[(dataframe[column] < lower) | (dataframe[column] > upper)]
            for _, row in outliers.iterrows():
                anomalies.append(
                    {
                        "file_name": file_name,
                        "document_type": "csv",
                        "folder_source": csv_loader.get_folder_source(file_name),
                        "page_number": None,
                        "row_index": int(row.name) + 1,
                        "column": column,
                        "value": None if pd.isna(row[column]) else float(row[column]),
                        "expected_range": [round(float(lower), 2), round(float(upper), 2)],
                    }
                )

    return anomalies


def _filter_findings_by_targets(findings: List[dict], target_tokens: List[str]) -> List[dict]:
    filtered: List[dict] = []
    for finding in findings:
        searchable_values = [
            str(finding.get("record_id", "")).lower(),
            str(finding.get("entity", "")).lower(),
            str(finding.get("summary", "")).lower(),
        ]
        if any(target in value for target in target_tokens for value in searchable_values):
            filtered.append(finding)
    return filtered


def _filter_rows_by_targets(rows: List[dict], target_tokens: List[str]) -> List[dict]:
    filtered: List[dict] = []
    for row in rows:
        haystack = " ".join(str(value).lower() for value in row.values())
        if any(token in haystack for token in target_tokens):
            filtered.append(row)
    return filtered


def _detect_candidate_findings(csv_loader: "CSVLoader") -> List[dict]:
    findings: List[dict] = []

    tracker = csv_loader.get_dataframe("bpps_tracker_export.csv")
    if tracker is not None:
        for _, row in tracker.iterrows():
            candidate_name = row.get("candidate_name", "Unknown")
            candidate_id = row.get("candidate_id", "N/A")
            status_tracker = str(row.get("status_tracker", "")).strip()
            ready_to_join = str(row.get("ready_to_join", "")).strip()
            identity_complete = str(row.get("identity_complete", "")).strip()
            rtw_complete = str(row.get("rtw_complete", "")).strip()
            employment_complete = str(row.get("employment_complete", "")).strip()
            criminality_complete = str(row.get("criminality_complete", "")).strip()
            risk_level = str(row.get("risk_level", "")).strip() or "Medium"

            incomplete_checks = [
                label
                for label, value in [
                    ("identity", identity_complete),
                    ("right to work", rtw_complete),
                    ("employment history", employment_complete),
                    ("criminality", criminality_complete),
                ]
                if value == "No"
            ]

            if ready_to_join == "Yes" and incomplete_checks:
                findings.append(
                    _finding(
                        row=row,
                        file_name="bpps_tracker_export.csv",
                        folder_source=csv_loader.get_folder_source("bpps_tracker_export.csv"),
                        title="Ready-to-join set before checks were complete",
                        summary=f"{candidate_name} is marked ready to join while {', '.join(incomplete_checks)} checks remain incomplete.",
                        rule="Candidates should not be marked ready to join until mandatory BPSS checks are complete.",
                        severity="Critical" if risk_level == "High" else "High",
                        entity=candidate_name,
                        record_id=candidate_id,
                    )
                )

            if status_tracker == "Pending":
                findings.append(
                    _finding(
                        row=row,
                        file_name="bpps_tracker_export.csv",
                        folder_source=csv_loader.get_folder_source("bpps_tracker_export.csv"),
                        title="Screening still pending",
                        summary=f"{candidate_name} still has a pending screening status.",
                        rule="Pending BPSS screening cases require completion before a hiring decision is finalized.",
                        severity="High",
                        entity=candidate_name,
                        record_id=candidate_id,
                    )
                )

            if status_tracker == "Risk Accepted":
                findings.append(
                    _finding(
                        row=row,
                        file_name="bpps_tracker_export.csv",
                        folder_source=csv_loader.get_folder_source("bpps_tracker_export.csv"),
                        title="Risk accepted case",
                        summary=f"{candidate_name} has a documented risk-accepted status and should be treated as an exception case.",
                        rule="Risk acceptance should be explicitly documented and justified.",
                        severity="Low",
                        entity=candidate_name,
                        record_id=candidate_id,
                        finding_type="exception",
                    )
                )

    inventory = csv_loader.get_dataframe("document_inventory.csv")
    if inventory is not None:
        for _, row in inventory.iterrows():
            candidate_id = row.get("candidate_id", "N/A")
            present = str(row.get("present_in_folder", "")).strip()
            remarks = str(row.get("remarks", "")).strip()
            doc_type = str(row.get("doc_type", "")).strip()

            if present == "No":
                findings.append(
                    _finding(
                        row=row,
                        file_name="document_inventory.csv",
                        folder_source=csv_loader.get_folder_source("document_inventory.csv"),
                        title="Required document missing",
                        summary=f"{candidate_id} is missing the required document '{doc_type}'.",
                        rule="Mandatory screening documents must be present in the candidate file.",
                        severity="High",
                        entity=candidate_id,
                        record_id=candidate_id,
                    )
                )

            if any(flag in remarks.lower() for flag in ["older than 90 days", "expired"]):
                findings.append(
                    _finding(
                        row=row,
                        file_name="document_inventory.csv",
                        folder_source=csv_loader.get_folder_source("document_inventory.csv"),
                        title="Supporting document may be invalid",
                        summary=f"{candidate_id} has a {doc_type} document flagged as '{remarks}'.",
                        rule="Identity, address, and right-to-work evidence must be current and valid at review time.",
                        severity="Medium",
                        entity=candidate_id,
                        record_id=candidate_id,
                    )
                )

    employment = csv_loader.get_dataframe("employment_history.csv")
    if employment is not None:
        for _, row in employment.iterrows():
            candidate_id = row.get("candidate_id", "N/A")
            evidence_status = str(row.get("evidence_status", "")).strip()
            evidence_type = str(row.get("evidence_type", "")).strip()

            if evidence_status in {"Unexplained", "Weak"}:
                findings.append(
                    _finding(
                        row=row,
                        file_name="employment_history.csv",
                        folder_source=csv_loader.get_folder_source("employment_history.csv"),
                        title="Employment history evidence is weak",
                        summary=f"{candidate_id} has {evidence_type} evidence marked '{evidence_status}'.",
                        rule="Employment history should be corroborated with sufficient supporting evidence.",
                        severity="High" if evidence_status == "Unexplained" else "Medium",
                        entity=candidate_id,
                        record_id=candidate_id,
                    )
                )

    return findings


def _detect_audit_findings(csv_loader: "CSVLoader") -> List[dict]:
    return []


def _collect_policy_hints(focus: List[str]) -> List[str]:
    hints: List[str] = []
    for area in focus:
        hints.extend(POLICY_HINTS.get(area, []))
    return sorted(set(hints))


def _finding(
    *,
    row: pd.Series,
    file_name: str,
    folder_source: str,
    title: str,
    summary: str,
    rule: str,
    severity: str,
    entity: str | None,
    record_id: str | None,
    finding_type: str = "violation",
) -> dict:
    return {
        "finding_type": finding_type,
        "title": title,
        "summary": summary,
        "rule": rule,
        "severity": severity,
        "entity": entity,
        "record_id": record_id,
        "file_name": file_name,
        "document_type": "csv",
        "folder_source": folder_source,
        "page_number": None,
        "row_index": int(row.name) + 1,
    }


def _extract_entity_name(record: Dict[str, object]) -> str | None:
    for key in ["candidate_name", "full_name", "candidate_id", "employee_name", "department", "audit_id"]:
        value = record.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None
