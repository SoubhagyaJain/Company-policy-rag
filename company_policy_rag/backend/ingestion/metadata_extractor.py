from __future__ import annotations

import re
from typing import Any, Literal
from datetime import datetime

from backend.models.document import DocumentMetadata, ExtractedDocumentMetadata
from backend.utils.logging import logger
from src.config import settings

try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None


DEPARTMENT_HEADER_PATTERNS = [
    re.compile(r"(?i)\b(?:Department|Dept|Owner|Division|Functional Area)\s*[:=\-]\s*([A-Za-z0-9 &/\-_]+)"),
    re.compile(r"(?i)\b(?:Issued By|Authorizing Body|Administered By)\s*[:=\-]\s*([A-Za-z0-9 &/\-_]+)"),
]

DEPARTMENT_CANONICAL_MAP: dict[str, str] = {
    # HR Aliases
    "hr": "HR",
    "human resources": "HR",
    "people": "HR",
    "people operations": "HR",
    "people ops": "HR",
    "talent": "HR",
    "personnel": "HR",
    "employee relations": "HR",
    # IT Aliases
    "it": "IT",
    "information technology": "IT",
    "infosec": "IT",
    "cybersecurity": "IT",
    "cyber security": "IT",
    "information security": "IT",
    "tech ops": "IT",
    "systems administration": "IT",
    "systems admin": "IT",
    # Finance Aliases
    "finance": "Finance",
    "accounting": "Finance",
    "payroll": "Finance",
    "tax": "Finance",
    "treasury": "Finance",
    "billing": "Finance",
    "financial operations": "Finance",
    # Legal & Compliance Aliases
    "legal": "Legal",
    "compliance": "Legal",
    "regulatory": "Legal",
    "governance": "Legal",
    "contracts": "Legal",
    "ethics": "Legal",
    "legal affairs": "Legal",
    "legal & compliance": "Legal",
    # Operations Aliases
    "operations": "Operations",
    "facilities": "Operations",
    "logistics": "Operations",
    "workplace": "Operations",
    "real estate": "Operations",
    # Engineering / Product Aliases
    "engineering": "Engineering",
    "r&d": "Engineering",
    "software engineering": "Engineering",
    "product": "Engineering",
    "quality assurance": "Engineering",
    "qa": "Engineering",
    # Marketing & Sales
    "marketing": "Marketing",
    "communications": "Marketing",
    "sales": "Sales",
    "business development": "Sales",
    # General
    "general": "General",
    "company-wide": "General",
    "all employees": "General",
    "corporate": "General",
}

POLICY_ID_PATTERNS = [
    # Pattern 1: Explicit label with alphanumeric code (e.g., "Policy ID: POL-HR-001", "Policy No. 102.4")
    re.compile(r"(?i)\b(?:Policy\s*(?:ID|Number|No\.?|#)|Doc(?:ument)?\s*(?:ID|Code|Ref|Reference|No\.?))\s*[:=\-]?\s*([A-Za-z0-9\.\-_/]+)"),
    # Pattern 2: Standard Enterprise Triple-Token Code (e.g., POL-HR-001, IT-SEC-2024, FIN-EXP-04-v2)
    re.compile(r"\b([A-Z]{2,6}[-_][A-Z]{2,6}[-_]\d{2,5}(?:[-_]v?\d+)?)\b"),
    # Pattern 3: Standard Enterprise Double-Token Code (e.g., HR-101, SEC-2024, FIN-05, POL-042)
    re.compile(r"\b([A-Z]{2,5}[-_]\d{2,5}(?:[-_]v?\d+)?)\b"),
    # Pattern 4: Section-style Policy Number (e.g., "Policy 12.4.1", "Policy #5.2")
    re.compile(r"(?i)\bPolicy\s*(?:#|No\.?)\s*(\d+(?:\.\d+)+)\b"),
]

DATE_HEADER_PATTERNS = [
    re.compile(r"(?i)\b(?:Effective\s+Date|Effective|Date\s+of\s+Issue|Date\s+of\s+Effectiveness)\s*[:=\-]?\s*([A-Za-z0-9, /.\-]+)"),
    re.compile(r"(?i)\b(?:Last\s+Revised|Revised\s+Date|Revised|Last\s+Updated|Published\s+Date)\s*[:=\-]?\s*([A-Za-z0-9, /.\-]+)"),
    re.compile(r"(?i)\b(?:Date\s*[:=\-])\s*([A-Za-z0-9, /.\-]+)"),
]

ROLE_PATTERN = re.compile(
    r"\b(HR Director|Department Head|Chief [A-Za-z]+ Officer|CISO|CTO|CFO|CEO|Manager|Supervisor|Employee|"
    r"Contractor|System Administrator|Admin|Ethics Officer|Compliance Officer|Privacy Officer|"
    r"People Operations Manager|Managing Director|General Counsel)\b",
    re.IGNORECASE,
)

DOLLAR_PATTERN = re.compile(
    r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|thousand|k|M|billion))?(?:\s*\/\s*(?:day|week|month|year|meal|trip|incident|yr))?",
    re.IGNORECASE,
)

DURATION_PATTERN = re.compile(
    r"\b\d+\s*(?:business\s+|calendar\s+|working\s+)?(?:days?|weeks?|months?|years?|hours?)\b",
    re.IGNORECASE,
)

PERIOD_PATTERN = re.compile(
    r"\b(probationary period|notice period|waiting period|retention period|grace period|"
    r"annual review|quarterly review|bi-weekly|monthly|semi-annual|at-will employment)\b",
    re.IGNORECASE,
)

POLICY_TOPIC_TAXONOMY: dict[str, list[str]] = {
    "remote_work": [
        "remote work", "work from home", "wfh", "telework", "telecommuting",
        "home office", "virtual work", "remote employee", "hybrid work", "flexible work"
    ],
    "leave_pto": [
        "pto", "paid time off", "vacation", "sick leave", "sick day", "bereavement",
        "parental leave", "maternity", "paternity", "fmla", "jury duty", "unpaid leave",
        "holiday pay", "leave of absence", "sabbatical"
    ],
    "it_security": [
        "password", "mfa", "2fa", "authentication", "encryption", "data security",
        "vpn", "access control", "phishing", "incident response", "endpoint", "byod",
        "acceptable use", "clean desk", "data retention", "cybersecurity", "confidentiality"
    ],
    "expenses_travel": [
        "travel expense", "per diem", "mileage", "reimbursement", "hotel", "airfare",
        "business meals", "receipts", "lodging", "expense report", "corporate card",
        "travel policy", "entertainment expense"
    ],
    "benefits_health": [
        "health insurance", "medical", "dental", "vision", "401k", "retirement",
        "hsa", "fsa", "life insurance", "disability", "wellness program", "cobra",
        "tuition assistance", "commuter benefits"
    ],
    "code_of_conduct": [
        "code of conduct", "ethics", "harassment", "sexual harassment", "discrimination",
        "whistleblower", "conflict of interest", "gifts and entertainment", "anti-bribery",
        "equal opportunity", "confidentiality", "nda", "workplace conduct"
    ],
    "compensation_payroll": [
        "salary", "base pay", "overtime", "bonus", "payroll", "direct deposit",
        "pay frequency", "wage", "timesheet", "exempt", "non-exempt", "minimum wage",
        "severance", "commission"
    ],
    "performance_discipline": [
        "performance review", "performance appraisal", "disciplinary action", "pip",
        "performance improvement", "written warning", "termination", "resignation",
        "at-will", "notice period", "separation", "exit interview"
    ],
    "safety_workplace": [
        "workplace safety", "osha", "ergonomics", "emergency evacuation", "fire safety",
        "first aid", "workplace injury", "hazard reporting", "security badge"
    ],
    "hiring_onboarding": [
        "onboarding", "background check", "probationary period", "job description",
        "recruitment", "interviewing", "referral bonus", "orientation"
    ],
}


class DocumentMetadataExtractor:
    """
    Extracts structured metadata (department, effective date, policy ID, entities, topics)
    from policy document content using heuristic rules, regex pattern matching,
    and optional LLM extraction.
    """

    def __init__(
        self,
        mode: Literal["heuristic", "llm", "hybrid"] | None = None,
        max_entities: int | None = None,
    ) -> None:
        self.mode = mode or getattr(settings, "metadata_extraction_mode", "heuristic")
        self.max_entities = max_entities or getattr(settings, "metadata_max_entities_per_chunk", 20)

    def flatten_for_chroma(self, extracted: ExtractedDocumentMetadata) -> dict[str, Any]:
        """Convert extracted metadata into ChromaDB-compatible primitive dictionary."""
        def _to_csv(val: Any) -> str:
            if isinstance(val, list):
                return ", ".join(str(x) for x in val if x is not None)
            return str(val) if val is not None else ""

        return {
            "department": str(extracted.department or "General"),
            "category": str(extracted.category or "general"),
            "effective_date": str(extracted.effective_date or ""),
            "policy_id": str(extracted.policy_id or ""),
            "key_entities": _to_csv(extracted.key_entities),
            "topic_tags": _to_csv(extracted.topic_tags),
        }

    def extract_department(self, text: str, default: str = "General") -> tuple[str, float]:
        """Extract departmental ownership from header patterns or body keyword density."""
        if not text:
            return default, 0.5

        # 1. Header scan (first 1000 characters)
        header_text = text[:1000]
        for pattern in DEPARTMENT_HEADER_PATTERNS:
            match = pattern.search(header_text)
            if match:
                raw_dept = match.group(1).strip().strip(".,;:|()[]{}")
                clean_dept = raw_dept.lower()
                # Direct check for Information Technology / IT
                if clean_dept == "information technology":
                    return "Information Technology", 0.95
                if clean_dept == "it":
                    return "IT", 0.95
                # Direct lookup in canonical map
                if clean_dept in DEPARTMENT_CANONICAL_MAP:
                    if clean_dept == DEPARTMENT_CANONICAL_MAP[clean_dept].lower():
                        return DEPARTMENT_CANONICAL_MAP[clean_dept], 0.95
                    return raw_dept.title(), 0.95
                for alias, canonical in DEPARTMENT_CANONICAL_MAP.items():
                    if alias in clean_dept:
                        return canonical, 0.90
                # If non-empty alphanumeric phrase under header
                if len(raw_dept) >= 2 and len(raw_dept) <= 40:
                    return raw_dept.title(), 0.85

        # 2. Body keyword scoring
        scores: dict[str, float] = {}
        text_lower = text.lower()
        header_lower = header_text.lower()

        for alias, canonical in DEPARTMENT_CANONICAL_MAP.items():
            if alias in ("general", "corporate", "company-wide", "all employees"):
                continue
            # Header occurrences weighted 3x, rest weighted 1x
            header_count = len(re.findall(r"\b" + re.escape(alias) + r"\b", header_lower))
            body_count = len(re.findall(r"\b" + re.escape(alias) + r"\b", text_lower))
            score = (header_count * 3.0) + body_count
            if score > 0:
                scores[canonical] = scores.get(canonical, 0.0) + score

        if scores:
            best_dept, best_score = max(scores.items(), key=lambda x: x[1])
            if best_score >= 3.0:
                return best_dept, min(0.85, 0.60 + (best_score * 0.05))
            if best_score >= 1.0:
                return best_dept, 0.65

        return default, 0.50

    def extract_policy_id(self, text: str) -> str | None:
        """Extract alphanumeric policy identifier/code."""
        if not text:
            return None

        # Look in first 2000 characters first (usually in header/title)
        scan_text = text[:2000]
        for pattern in POLICY_ID_PATTERNS:
            matches = pattern.finditer(scan_text)
            for match in matches:
                raw_id = match.group(1).strip().strip(".,;:|()[]{}")
                # Filter out pure numbers unless preceded by Policy keyword
                if raw_id.isdigit():
                    if len(raw_id) == 4 and (raw_id.startswith("19") or raw_id.startswith("20")):
                        continue  # Discard standalone year
                    if "policy" in match.group(0).lower():
                        return f"Policy {raw_id}"
                    continue
                if len(raw_id) >= 3:
                    return raw_id

        # Secondary search across entire text
        for pattern in POLICY_ID_PATTERNS:
            match = pattern.search(text)
            if match:
                raw_id = match.group(1).strip().strip(".,;:|()[]{}")
                if not raw_id.isdigit() and len(raw_id) >= 3:
                    return raw_id

        return None

    def normalize_date(self, raw_date_str: str) -> str | None:
        """Normalize raw date string into ISO 8601 format (YYYY-MM-DD)."""
        if not raw_date_str:
            return None

        clean_str = raw_date_str.strip().strip(".,;:|()[]{}")
        # Truncate anything after newline or delimiter
        clean_str = re.split(r"[\n\r|]", clean_str)[0].strip()

        # 1. Regex check for ISO format YYYY-MM-DD
        iso_match = re.search(r"\b(20\d{2}|19\d{2})[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])\b", clean_str)
        if iso_match:
            y, m, d = iso_match.groups()
            return f"{y}-{int(m):02d}-{int(d):02d}"

        # 2. Regex check for MM/DD/YYYY or MM-DD-YYYY
        us_match = re.search(r"\b(0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])[-/.](20\d{2}|19\d{2})\b", clean_str)
        if us_match:
            m, d, y = us_match.groups()
            return f"{y}-{int(m):02d}-{int(d):02d}"

        # 3. Use dateutil parser if available
        if date_parser is not None:
            try:
                dt = date_parser.parse(clean_str, fuzzy=True)
                # Sanity check year
                if 1980 <= dt.year <= 2100:
                    return dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        # 4. Fallback manual standard formats (e.g. Month DD, YYYY)
        month_names = {
            "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
            "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
            "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
            "nov": 11, "november": 11, "dec": 12, "december": 12,
        }
        word_date_match = re.search(
            r"\b([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2}|19\d{2})\b",
            clean_str,
            re.IGNORECASE,
        )
        if word_date_match:
            mon_str, day_str, yr_str = word_date_match.groups()
            mon_key = mon_str.lower()
            if mon_key in month_names:
                return f"{yr_str}-{month_names[mon_key]:02d}-{int(day_str):02d}"

        day_mon_yr_match = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\.?,?\s+(20\d{2}|19\d{2})\b",
            clean_str,
            re.IGNORECASE,
        )
        if day_mon_yr_match:
            day_str, mon_str, yr_str = day_mon_yr_match.groups()
            mon_key = mon_str.lower()
            if mon_key in month_names:
                return f"{yr_str}-{month_names[mon_key]:02d}-{int(day_str):02d}"

        mon_yr_match = re.search(
            r"\b([A-Za-z]+)\.?,?\s+(20\d{2}|19\d{2})\b",
            clean_str,
            re.IGNORECASE,
        )
        if mon_yr_match:
            mon_str, yr_str = mon_yr_match.groups()
            mon_key = mon_str.lower()
            if mon_key in month_names:
                return f"{yr_str}-{month_names[mon_key]:02d}-01"

        return None

    def extract_effective_date(self, text: str) -> str | None:
        """Extract effective date or last revised date from header."""
        if not text:
            return None

        scan_text = text[:1500]
        for pattern in DATE_HEADER_PATTERNS:
            match = pattern.search(scan_text)
            if match:
                raw_date = match.group(1)
                normalized = self.normalize_date(raw_date)
                if normalized:
                    return normalized

        # Fallback date search in header
        normalized_general = self.normalize_date(scan_text)
        return normalized_general

    def extract_key_entities(self, text: str, max_entities: int | None = None) -> list[str]:
        """Extract roles, dollar amounts, durations, and policy periods."""
        if not text:
            return []

        limit = max_entities or self.max_entities
        entities: list[str] = []
        seen: set[str] = set()

        def _add(item: str) -> None:
            cleaned = item.strip().strip(".,;:|()[]{}")
            # Replace internal commas to ensure safe ChromaDB serialization
            cleaned = cleaned.replace(",", "")
            key = cleaned.lower()
            if cleaned and key not in seen and len(cleaned) >= 2:
                seen.add(key)
                entities.append(cleaned)

        # 0. Core Roles in lowercase / plural
        core_roles = [
            "full-time employees", "part-time employees", "system administrators", "system administrator",
            "contractors", "contractor", "employees", "employee", "managers", "manager",
            "directors", "director", "supervisors", "supervisor", "executives", "executive",
            "sysadmin", "ciso", "cto", "cfo", "ceo", "general counsel", "hr director",
            "department head", "compliance officer", "ethics officer", "privacy officer",
        ]
        for role in core_roles:
            pattern = r"\b" + re.escape(role) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                _add(role)

        # 1. Dollar amounts
        for match in DOLLAR_PATTERN.finditer(text):
            _add(match.group(0))

        # 2. Durations / deadlines
        for match in DURATION_PATTERN.finditer(text):
            _add(match.group(0))

        # 3. Roles / authority figures
        for match in ROLE_PATTERN.finditer(text):
            _add(match.group(0).title())

        # 4. Key policy periods
        for match in PERIOD_PATTERN.finditer(text):
            _add(match.group(0).title())

        return entities[:limit]

    def extract_topic_tags(self, text: str) -> list[str]:
        """Classify document content into taxonomy tags and specific domain keywords."""
        if not text:
            return []

        text_lower = text.lower()
        matched_tags: list[str] = []
        seen: set[str] = set()

        def _add_tag(t: str) -> None:
            clean = t.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                matched_tags.append(clean)

        # 1. High-frequency domain keyword tokens
        domain_keyword_tokens = [
            "security", "vpn", "access control", "password", "mfa", "encryption",
            "pto", "vacation", "sick leave", "leave", "parental leave", "benefits",
            "travel", "expenses", "reimbursement", "per diem", "mileage",
            "remote work", "work from home", "wfh", "hybrid work",
            "code of conduct", "ethics", "harassment", "confidentiality",
            "health insurance", "medical", "dental", "vision", "401k",
            "salary", "payroll", "overtime", "bonus",
            "performance review", "discipline", "termination",
            "workplace safety", "osha", "onboarding", "background check"
        ]

        for kw in domain_keyword_tokens:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_lower):
                _add_tag(kw)

        # 2. Taxonomy category classifications
        for topic, keywords in POLICY_TOPIC_TAXONOMY.items():
            for kw in keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, text_lower):
                    _add_tag(topic)
                    break

        return matched_tags

    def extract(
        self,
        text: str,
        doc_metadata: DocumentMetadata | None = None,
    ) -> ExtractedDocumentMetadata:
        """
        Execute full metadata extraction on text and return ExtractedDocumentMetadata.
        """
        default_dept = "General"
        category = "general"
        if doc_metadata:
            if doc_metadata.department:
                default_dept = doc_metadata.department
            if doc_metadata.category:
                category = doc_metadata.category

        department, dept_conf = self.extract_department(text, default=default_dept)
        policy_id = self.extract_policy_id(text)
        effective_date = self.extract_effective_date(text)
        key_entities = self.extract_key_entities(text)
        topic_tags = self.extract_topic_tags(text)

        confidence_scores: dict[str, float] = {
            "department": dept_conf,
            "policy_id": 0.95 if policy_id else 0.0,
            "effective_date": 0.90 if effective_date else 0.0,
            "key_entities": 0.85 if key_entities else 0.0,
            "topic_tags": 0.85 if topic_tags else 0.0,
        }

        overall_conf = (
            dept_conf * 0.35
            + (0.95 if policy_id else 0.5) * 0.20
            + (0.90 if effective_date else 0.5) * 0.15
            + (0.85 if topic_tags else 0.5) * 0.30
        )

        return ExtractedDocumentMetadata(
            department=department,
            category=category,
            effective_date=effective_date,
            policy_id=policy_id,
            key_entities=key_entities,
            topic_tags=topic_tags,
            confidence=round(overall_conf, 2),
            confidence_scores=confidence_scores,
            extraction_method="heuristic",
            extra={},
        )

    def flatten_for_chroma(self, extracted: ExtractedDocumentMetadata) -> dict[str, Any]:
        """Convert extracted metadata into ChromaDB-compatible primitive dictionary."""
        return {
            "department": str(extracted.department or "General"),
            "category": str(extracted.category or "general"),
            "effective_date": str(extracted.effective_date or ""),
            "policy_id": str(extracted.policy_id or ""),
            "key_entities": ", ".join(extracted.key_entities) if isinstance(extracted.key_entities, list) else str(extracted.key_entities or ""),
            "topic_tags": ", ".join(extracted.topic_tags) if isinstance(extracted.topic_tags, list) else str(extracted.topic_tags or ""),
        }

