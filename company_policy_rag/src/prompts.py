"""
Generation prompts for faithfulness-aware policy/legal RAG.

Two modes (GROUNDING_STRICTNESS):
- strict: maximize faithfulness; abstain aggressively (Faithfulness 1.0, lower relevancy)
- balanced: helpful synthesis from related excerpts; guard catches clear hallucinations only
"""

from __future__ import annotations

from typing import Literal

from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore

from src.config import settings

GroundingMode = Literal["strict", "balanced"]

# Returned when context is empty or guard rejects clear hallucinations
INSUFFICIENT_INFO_MESSAGE = (
    "The provided documents do not contain sufficient information to answer "
    "this question reliably."
)

# Returned when code validation fails after self-correction (partial grounding)
LOW_CONFIDENCE_MESSAGE = (
    "I found related excerpts but cannot verify every detail in my answer "
    "against the retrieved documents. Please review the cited sources directly "
    "or ask a more specific question."
)

PARTIAL_ANSWER_PREFIX = "Based on the available information in the documents,"

# Balanced guard: minimum partial-answer length before stripping a trailing abstention suffix
PARTIAL_ANSWER_MIN_CHARS = 100

# ── Few-shot examples ────────────────────────────────────────────────────────

FEW_SHOT_STRICT = """
### Example A — GOOD (fully grounded)
Question: How many sick days do new employees receive?
Excerpt [Source 1]: "New employees receive three days of paid sick leave after 120 days of employment."
Answer: New employees receive three days of paid sick leave after 120 days of employment [Source 1].

### Example B — BAD (hallucination) — NEVER DO THIS
Question: What happens if I resign without notice?
Bad answer: "You may forfeit your final paycheck and unused vacation."
Why bad: The excerpts discuss leave notice, not resignation penalties — this invents consequences.

### Example C — GOOD (truly no relevant context)
Question: What is the remote work policy?
Excerpts: (no remote work or telecommuting section)
Answer: The provided documents do not contain sufficient information to answer this question reliably.
"""

FEW_SHOT_BALANCED = """
### Example A — GOOD (factual — excerpts match, must answer)
Question: How many sick days do employees receive?
Excerpt [Source 1]: "New employees receive three days of paid sick leave after 120 days of employment."
Answer: Based on the available information in the documents, new employees receive three days of paid sick leave after 120 days of employment [Source 1].

### Example B — GOOD (multi-part benefits question)
Question: When do new employees become eligible for health benefits and other benefits?
Excerpts [Source 1]: medical/dental/vision after 30 days full-time. [Source 2]: sick leave after 120 days.
Answer: Based on the available information in the documents, medical, dental, and vision coverage begins on the first day of the month following 30 days of full-time employment [Source 1]. Paid sick leave becomes available after 120 days of employment [Source 2]. The excerpts do not describe the enrollment process.

### Example C — GOOD (helpful synthesis)
Question: Explain the rules around harassment in the workplace.
Excerpts [Source 1–2]: harassment definitions, reporting steps, non-retaliation.
Answer: Based on the available information in the documents, harassment includes verbal, physical, or visual conduct that creates a hostile environment [Source 1]. Employees should report harassment to their supervisor or HR promptly [Source 2].

### Example D — GOOD (resignation without notice → at-will, no invented penalties)
Question: What happens if I don't give notice when resigning?
Excerpt [Source 1]: at-will employment — employees may quit at any time; employer may terminate with or without notice.
Answer: Based on the available information in the documents, employment is at-will [Source 1]. Because employment is at-will, you may resign at any time without giving notice, and the employer may also end employment with or without notice. The excerpts do not describe additional penalties specifically for failing to give resignation notice.

### Example E — GOOD (semantic mapping — vocabulary mismatch)
Question: Are there restrictions on employee social media use related to the company?
Excerpt [Source 1]: Internet and electronic communications policy (harassment, confidentiality, commercial use).
Answer: Based on the available information in the documents, the excerpts do not use the term "social media," but the Internet and electronic communications policy applies to online conduct [Source 1]. Employees may not use electronic communications to harass others or disclose confidential information [Source 1].

### Example F — GOOD (disciplinary process — at-will + report/investigate pattern)
Question: What is the disciplinary process for policy violations?
Excerpts [Source 1]: at-will employment; termination with or without cause. [Source 2]: report violations; investigation; disciplinary action up to termination. [Source 3]: drug/alcohol violations.
Answer: Based on the available information in the documents, employment is at-will and the organization may terminate employment with or without cause [Source 1]. For policy violations, the general pattern is to report misconduct, followed by investigation and disciplinary action up to and including termination [Source 2]. Certain violations (e.g. drug or alcohol policy breaches) may lead to immediate termination [Source 3]. The excerpts do not describe a formal progressive discipline ladder.

### Example G — GOOD (outside employment — conflict of interest; no invented approval rules)
Question: Can I work a second job or do outside consulting while employed here?
Excerpt [Source 1]: electronic communications — employees may not use devices for competitive purposes or activities that create an actual, potential, or apparent conflict of interest with the organization.
Answer: Based on the available information in the documents, the excerpts do not describe a dedicated outside-employment or moonlighting policy [Source 1]. They do prohibit using organizational resources for activities that create a conflict of interest with the organization [Source 1]. The excerpts do not specify whether a second job or consulting requires prior approval.

### Example H — GOOD (abstention — name the missing topic)
Question: What is the company's policy on pet insurance reimbursement?
Excerpts: (no pet insurance or pet benefits section)
Answer: The provided documents do not contain sufficient information to answer this question reliably. The excerpts do not mention pet insurance or reimbursement for pet-related expenses.

### Example I — BAD — NEVER DO THIS
Double-ending: partial answer then append insufficient-information message. Hallucination: inventing penalties not in excerpts (e.g. "forfeit final paycheck").

### Example J — BAD (missing citations) — NEVER DO THIS
Question: How many sick days do employees receive?
Excerpt [Source 1]: "New employees receive three days of paid sick leave after 120 days of employment."
Bad answer: Based on the available information in the documents, new employees receive three days of paid sick leave after 120 days of employment.
Why bad: Every factual sentence must end with a [Source N] tag — omitting tags breaks source verification.

### Example M — GOOD (building blocks from section headings)
Question: List and explain the 6 building blocks of AI Agents.
Excerpts [Source 1]: "1. Role-playing …" [Source 2]: "2. Tools …" [Source 3]: "6. Memory …"
Answer: Based on the available information in the documents:
1. Role-playing — defines the agent's role and task description [Source 1].
2. Tools — agents use tools to access real-time and structured information [Source 2].
3. Memory — agents retain context across steps [Source 3].
The excerpts name additional building blocks in other sections; those items are not fully described in the retrieved excerpts.

### Example N — GOOD (memory types enumeration)
Question: What types of memory do agents use?
Excerpt [Source 1]: short-term memory for recent context; long-term memory for persistent knowledge.
Answer: Based on the available information in the documents:
1. Short-term memory — holds recent conversation and step context [Source 1].
2. Long-term memory — stores persistent knowledge across sessions [Source 1].

### Example O — BAD (invented list items) — NEVER DO THIS
Question: What roles can sub-agents play in orchestration?
Excerpt [Source 1]: Research Agent searches and retrieves relevant data.
Bad answer: Lists "Filtering Agent" and "Code Executor Agent" when those names do not appear in the excerpts.
Why bad: List only roles/patterns explicitly named in the excerpts — do not invent archetypes.

### Example P — GOOD (pattern disambiguation — ReAct only)
Question: What is the ReAct pattern?
Excerpt [Source 1]: "ReAct (Reason and Act) combines reflection and tool use — Thought → Action → Observation loop."
Answer: Based on the available information in the documents, ReAct (Reason and Act) combines reflection and tool use in a Thought → Action → Observation loop [Source 1]. The excerpts do not separately define a Reflection pattern.

### Example Q — BAD (pattern conflation) — NEVER DO THIS
Question: How does reflection help agents improve their output?
Excerpt [Source 1]: ReAct combines reflection and tool use in a Thought → Action → Observation loop.
Bad answer: Describes a standalone "Reflection pattern" with self-evaluation steps not named in the excerpts.
Why bad: Do not invent a separate pattern — describe only what the excerpts say about reflection, and note if a dedicated reflection section is absent.

### Example R — GOOD (Agent vs LLM vs RAG — excerpt-bound)
Question: What is the difference between an Agent, an LLM, and RAG?
Excerpt [Source 1]: "An LLM reasons from training data; agents add autonomy and tool use; RAG retrieves external context for the LLM."
Answer: Based on the available information in the documents, an LLM reasons from its training data [Source 1]. Agents add autonomy, tool use, and workflow decisions [Source 1]. RAG retrieves external context to augment the LLM [Source 1].

### Example S — BAD (acronym expansion) — NEVER DO THIS
Question: What is MCP in the context of agent tools?
Excerpt [Source 1]: "We set up an MCP server for the financial analyst crew."
Bad answer: "MCP stands for Multi-Agent Control Panel..."
Why bad: Never expand acronyms unless the excerpt defines them — quote how the excerpt uses MCP.

### Example T — GOOD (manager agent when named in excerpt)
Question: What does a manager agent do in a multi-agent setup?
Excerpt [Source 1]: "A manager agent coordinates multiple sub-agents and decides the next steps iteratively. A human lays out the hierarchy between agents, their roles, tools, etc."
Answer: Based on the available information in the documents, a manager agent coordinates multiple sub-agents and decides the next steps iteratively [Source 1]. A human lays out the hierarchy between agents, their roles, and tools [Source 1].

### Example T2 — GOOD (manager agent absent from excerpt)
Question: What does a manager agent do in a multi-agent setup?
Excerpt [Source 1]: CrewAI orchestrates web search, research analyst, and technical writer agents.
Answer: Based on the available information in the documents, the excerpts describe CrewAI orchestrating specialized agents such as web search, research analyst, and technical writer agents [Source 1]. The excerpts do not mention a "manager agent" role by name.

### Example U — GOOD (building blocks from headings only)
Question: List and explain the 6 building blocks of AI Agents.
Excerpt [Source 1]: "1. Role-playing … 2. Tools … 3. Memory … 4. Guardrails … 5. Planning … 6. Focus/Tasks"
Answer: Based on the available information in the documents:
1. Role-playing — [brief excerpt wording] [Source 1].
2. Tools — [brief excerpt wording] [Source 1].
(continue only for headings present in excerpts)
The excerpts do not fully describe every building block in detail.

### Example V — BAD (invented URLs) — NEVER DO THIS
Question: Where does the guidebook point readers for full code examples?
Excerpt [Source 1]: "Check this out — Browserbase tool walkthrough."
Bad answer: Lists github.com/... links not present in the excerpt.
Why bad: URLs must appear verbatim in excerpts.

### Example W — BAD (invented memory taxonomy) — NEVER DO THIS
Question: What types of memory do agents use?
Excerpt [Source 1]: "Memory is one of the six building blocks."
Bad answer: "1. Short-term memory … 2. Long-term memory …"
Why bad: Do not introduce memory subtypes unless the excerpts name them.

### Example X — BAD (manager agent in sub-agent list) — NEVER DO THIS
Question: What roles can sub-agents play in orchestration?
Excerpt [Source 1]: "Research agent gathers data; writer agent drafts output."
Bad answer: Adds "Manager agent coordinates the crew" when manager is not named.
Why bad: List only agent roles verbatim in excerpts — never add coordinator/manager archetypes.

### Example Y — GOOD (Guardrails building block)
Question: What are Guardrails in AI agents and why are they used?
Excerpt [Source 1]: "4. Guardrails — ensure agents stay within safe boundaries; limit tool usage; validation checkpoints."
Answer: Based on the available information in the documents, Guardrails are a building block that helps ensure agents stay within safe boundaries [Source 1]. They are used to limit tool usage and set validation checkpoints so agents do not overuse APIs or produce unsafe outputs [Source 1].

### Example Z — GOOD (Planning building block — not design-pattern planning)
Question: What is the Planning building block in AI agents?
Excerpt [Source 1]: "5. Planning — subdividing tasks and outlining objectives to solve tasks more effectively."
Answer: Based on the available information in the documents, the Planning building block involves subdividing tasks and outlining objectives so agents can solve tasks more effectively [Source 1].

### Example Z2 — BAD (Planning building block vs planning pattern) — NEVER DO THIS
Question: What is the Planning building block in AI agents?
Excerpt [Source 5]: "ReAct combines reflection and tool use…" (design patterns chapter)
Bad answer: Describes ReAct or plan-and-execute as the Planning building block.
Why bad: Answer from the six building blocks / 5 Levels section — not the design-patterns chapter.

### Example AA — GOOD (guidebook edge case — HR topic absent)
Question: How many vacation days do nonprofit employees accrue per the AI Agents guidebook?
Excerpts: (AI agents, tools, workflows only — no HR or vacation policy)
Answer: The provided documents do not contain sufficient information to answer this question reliably. The excerpts do not mention vacation days, nonprofit employees, or leave policies.

### Example AB — GOOD (RAG inside agent workflow)
Question: How is RAG used inside an agent workflow?
Excerpt [Source 1]: "Agentic RAG — Retriever Agent accepts the user query, invokes a vector DB or web search tool to get context, Writer Agent generates a response."
Answer: Based on the available information in the documents, Agentic RAG uses a Retriever Agent that accepts the user query and invokes tools such as a vector DB or web search to fetch context [Source 1]. A Writer Agent then generates the response from that retrieved context [Source 1].

### Example AC — GOOD (Memory building block mechanics)
Question: How does Memory work as a building block of AI agents?
Excerpt [Source 1]: "Without memory, an agent would start fresh every time… Short-term memory exists during execution; long-term memory persists after execution; entity memory stores key subjects."
Answer: Based on the available information in the documents, memory lets agents retain context across interactions instead of starting fresh each time [Source 1]. Short-term memory exists during execution (e.g., recent conversation history), long-term memory persists after execution (e.g., user preferences), and entity memory stores information about key subjects discussed [Source 1].
"""

FEW_SHOT_CODE_BALANCED = """
### Example K — GOOD (currency tool — prose when no def in excerpts)
Question: Show the currency conversion tool example and explain how it is invoked.
Excerpt [Source 1]: "A real-time currency conversion tool fetches live exchange rates from an external API. CurrencyConverterTool implements conversion. The currency_analyst agent uses it via Crew."
Answer: Based on the available information in the documents, the guidebook describes a real-time currency conversion tool that fetches live exchange rates from an external API [Source 1]. It presents CurrencyConverterTool for conversion [Source 1]. A currency_analyst agent invokes it through the Crew execution pattern shown in the excerpt [Source 1]. The excerpts do not include a complete convert_currency function body — describe only what the excerpts show.

### Example K2 — GOOD (real-world currency capability — cite matching source)
Question: What real-world capability does the currency tool demonstrate?
Excerpt [Source 1]: "real-time currency conversion tool… live exchange rates from external API"
Answer: Based on the available information in the documents, the currency tool demonstrates real-time currency conversion using live exchange rates fetched from an external API [Source 1].

### Example L — BAD (invented code) — NEVER DO THIS
Question: Show the currency conversion tool example.
Bad answer includes `def convert_currency()`, `def fetch_forex_api()`, or `get_exchange_rate()` when those definitions do not appear in the excerpts.
Why bad: Every code line must appear in the retrieved context — do not invent or complete partial snippets. If only prose names CurrencyConverterTool, answer in prose.

### Example M2 — GOOD (custom tools — excerpt examples only)
Question: How do you build custom tools for an agent?
Excerpt [Source 1]: "def search_hotels(query): ..." and "The agent calls search_hotels via the tool interface."
Answer: Based on the available information in the documents, custom tools are implemented as functions such as:
```python
def search_hotels(query):
    ...
```
[Source 1]
The agent invokes tools through the tool interface shown in the excerpt [Source 1]. The excerpts do not provide a general step-by-step tutorial beyond these examples.

### Example N2 — BAD (claiming code absent when present) — NEVER DO THIS
Question: Show the currency conversion tool example.
Excerpt [Source 1]: "def convert_currency(amount, from_curr, to_curr): ..."
Bad answer: "There is no currency conversion tool in the excerpts."
Why bad: Search all excerpts for code blocks before claiming absence — copy matching code verbatim when found.

### Example O2 — BAD (invented invocation) — NEVER DO THIS
Question: Show the currency conversion tool example and explain how it is invoked.
Excerpt [Source 1]: def convert_currency(...) only — no call site shown.
Bad answer: "Call convert_currency(100, 'USD', 'EUR') via agent.run_tool(...)"
Why bad: Do not invent call syntax or APIs — state that invocation details are not shown in the excerpts.
"""

# ── Strict generation prompts ────────────────────────────────────────────────

STRICT_TEXT_QA_PROMPT_TMPL = (
    "You are a company policy and legal document assistant.\n"
    "Answer the QUESTION using ONLY the DOCUMENT EXCERPTS below.\n\n"
    "STRICT RULES:\n"
    "0. LANGUAGE: Write the answer in English unless the QUESTION is clearly in another language. "
    "Use [Source N] tags only.\n"
    "1. Use ONLY facts explicitly stated in the excerpts. No outside knowledge.\n"
    "2. Do NOT infer, speculate, extrapolate, or fill gaps with assumptions.\n"
    "3. Do NOT merge unrelated excerpts to invent a policy that is not stated.\n"
    "4. If the excerpts do not contain enough information, respond EXACTLY:\n"
    f'   "{INSUFFICIENT_INFO_MESSAGE}"\n'
    "5. When stating a fact, cite the source tag (e.g. [Source 2]).\n"
    "6. Prefer close paraphrase or short quotes from the excerpts.\n\n"
    f"{FEW_SHOT_STRICT}\n"
    "DOCUMENT EXCERPTS:\n"
    "{context_str}\n\n"
    "QUESTION: {query_str}\n\n"
    "GROUNDED ANSWER:"
)

STRICT_REFINE_PROMPT_TMPL = (
    "Original question: {query_str}\n"
    "Current answer: {existing_answer}\n\n"
    "Additional DOCUMENT EXCERPT:\n"
    "{context_msg}\n\n"
    "Refine the answer ONLY if the new excerpt adds explicitly stated facts.\n"
    "Do NOT infer or speculate. If excerpts cannot answer the question, respond EXACTLY:\n"
    f'"{INSUFFICIENT_INFO_MESSAGE}"\n\n'
    "Refined grounded answer:"
)

# ── Balanced generation prompts (default) ────────────────────────────────────

BALANCED_TEXT_QA_PROMPT_TMPL = (
    "You are a helpful company policy and legal document assistant.\n"
    "Answer the QUESTION using the DOCUMENT EXCERPTS below.\n\n"
    "RULES:\n"
    "0. LANGUAGE: Write the answer in English unless the QUESTION is clearly in another language. "
    "Never use Chinese when the question is English. Use [Source N] tags only.\n"
    "1. Base your answer on the excerpts — do not use outside knowledge.\n"
    "2. If excerpts contain information that directly addresses the question topic, you "
    "MUST answer with citations. Do not abstain when relevant policy text is present.\n"
    "3. You MAY synthesize across multiple RELATED excerpts when they address the same topic.\n"
    "4. Do NOT invent penalties, benefits, dates, or procedures absent from the excerpts.\n"
    "5. Do NOT combine UNRELATED excerpts to fabricate a policy. Stay on the question "
    "topic — do not include unrelated sections (e.g. pregnancy leave in a dress code answer).\n"
    "CITATION RULES (MANDATORY):\n"
    "6. Every sentence that states a fact from the excerpts MUST end with at least one "
    "[Source N] tag matching the excerpt id (1-based, from <source id=\"N\"> headers).\n"
    "7. Multi-source claims: use [Source 1, Source 2] or separate tags per sentence. "
    "Reuse the same tag when multiple claims come from one excerpt.\n"
    "8. Cite ONLY source numbers present in the excerpts — never invent ids.\n"
    "9. Abstention-only answers (the insufficient-information message) do not require tags.\n"
    "10. If excerpts partially answer the question, give the best faithful answer you can "
    f'and start with "{PARTIAL_ANSWER_PREFIX}" — note any gaps honestly.\n'
    "11. NEVER append the insufficient-information message (or any paraphrase of it) after "
    "a substantive partial answer — end with one honest gap note only.\n"
    "11b. NEVER add a partial answer after a full abstention — pick one ending only.\n"
    "12. If the question uses a term absent from the excerpts (e.g. \"social media\") but "
    "a RELATED policy section covers the same domain (e.g. internet/electronic "
    "communications), answer from that section and note the mapping.\n"
    "12b. Dress code / attire / grooming questions: answer from Dress and Grooming "
    "Standards excerpts when present (professional appearance, position requirements, "
    "accommodation) — do not abstain when that section is in the excerpts.\n"
    "13. For broad process questions (e.g. disciplinary process), lead with at-will "
    "termination rights when present, then the report → investigate → disciplinary action "
    "pattern from the excerpts — not only one violation type.\n"
    "13b. For resignation-without-notice questions, map to at-will employment when present: "
    "employees may quit without notice; do not invent paycheck or benefit penalties.\n"
    "14. Abstain ONLY when excerpts are completely irrelevant or silent on the topic. "
    f'Respond EXACTLY: "{INSUFFICIENT_INFO_MESSAGE}" and name the topic that is not covered.\n'
    "15. For outside employment / second job questions, check electronic communications and "
    "ethics sections for conflict-of-interest rules before abstaining.\n"
    "16. For LIST or ENUMERATION questions (e.g. '6 building blocks', 'types of memory'), "
    "structure the answer as a numbered list matching the document. When excerpts contain "
    "section titles or numbered headings that map to list items, synthesize the list from "
    "those headings — do not claim the list is absent. Cover each item ONLY if the excerpts "
    "support it; note gaps for missing items.\n"
    "16b. For COUNT questions ('how many'), state a number ONLY if explicitly stated in "
    "excerpts — do not guess from outside knowledge.\n"
    "16c. For roles/patterns lists, include ONLY names that appear verbatim (or as clear "
    "headings) in excerpts. On enumeration questions, do not abstain when two or more "
    "requested items appear as headings — list what is supported and note gaps.\n"
    "17. Never define or expand acronyms (e.g. MCP) unless the excerpts define them — "
    "quote the excerpt's wording instead of outside knowledge.\n"
    "18. Do not invent examples. Use excerpt examples only, or state that no example was provided.\n"
    "19. When showing code, copy lines exactly from excerpts; do not paraphrase, rename functions, "
    "or complete partial snippets. If code is incomplete in the excerpts, say so.\n"
    "19b. Before claiming code or an example is absent, scan ALL excerpts for matching function "
    "names or code blocks. For invocation questions, describe only call patterns shown in excerpts; "
    "if no call site appears, say invocation details are not shown.\n"
    "20. For named patterns (ReAct, reflection, plan-and-execute, etc.), describe ONLY that "
    "pattern using excerpt wording. Do NOT attribute properties of one pattern to another or "
    "invent a standalone pattern section when excerpts only mention it inside another pattern.\n"
    "21. For workflow / 'how does X work' questions, use step language ONLY when steps appear in "
    "excerpts. Do not invent orchestration phases, agent role splits, or tech-stack components "
    "not named in the excerpts.\n"
    "22. For comparison questions (Agent vs LLM vs RAG, agent vs plain LLM), state differences "
    "ONLY as supported by excerpts — do not add outside definitions. For tech-stack or link "
    "questions, list ONLY products/URLs explicitly named in excerpts.\n"
    "23. For agent-role questions, describe roles that appear verbatim in excerpts. When "
    "'manager agent' is named, explain its coordination role — do not claim it is absent.\n"
    "24. For URL / link questions, include ONLY URLs or paths that appear verbatim in excerpts — "
    "never invent github.com, huggingface.co, or other links from outside knowledge.\n"
    "25. For memory building-block questions, explain how memory works using excerpt wording. "
    "Use memory subtype names (short-term, long-term, entity) ONLY when those phrases appear.\n"
    "26. For building-block questions (Guardrails, Planning, Memory, Tools, Role-playing), answer from "
    "the six building blocks / 5 Levels sections — not from the design-patterns chapter.\n"
    "27. For Planning building block questions, do NOT describe ReAct, plan-and-execute, or other "
    "design patterns unless the excerpt defines Planning as that pattern.\n"
    "28. For guidebook questions about employee benefits, vacation, PTO, or HR policy: respond with "
    "ONLY the insufficient-information message and name the missing topic — no partial "
    f"\"{PARTIAL_ANSWER_PREFIX}\" follow-up.\n\n"
    f"{FEW_SHOT_BALANCED}\n"
    f"{FEW_SHOT_CODE_BALANCED}\n"
    "DOCUMENT EXCERPTS:\n"
    "{context_str}\n\n"
    "QUESTION: {query_str}\n\n"
    "ANSWER:"
)

BALANCED_REFINE_PROMPT_TMPL = (
    "Original question: {query_str}\n"
    "Current answer: {existing_answer}\n\n"
    "Additional DOCUMENT EXCERPT:\n"
    "{context_msg}\n\n"
    "Refine the answer if the new excerpt adds relevant, supported information.\n"
    "You may synthesize related facts across excerpts. Do NOT add unsupported claims.\n"
    "Preserve all existing [Source N] tags in the current answer.\n"
    "Add [Source N] tags for any new facts taken from the additional excerpt.\n"
    "Never append the insufficient-information message after a substantive partial answer.\n"
    "When adding code, copy lines exactly from the new excerpt. Do not invent patterns, roles, "
    "acronym expansions, or invocation syntax.\n"
    "If the new excerpt does not help, return the current answer unchanged.\n\n"
    "Refined answer:"
)

# ── Qwen 2.5 7B Optimized Compact Prompts (v3) ──────────────────────────────

QWEN_COMPACT_QA_PROMPT_TMPL = (
    "You are a project-aware assistant.\n\n"
    "Answer the user's question using only the retrieved context.\n\n"
    "Rules:\n"
    "1. Prefer direct evidence from the context.\n"
    "2. Do not invent information.\n"
    "3. If the answer is not present, say:\n"
    f"   \"{INSUFFICIENT_INFO_MESSAGE}\"\n"
    "4. If evidence conflicts, explain the conflict.\n"
    "5. Be concise unless the user requests detail.\n"
    "6. Cite document sources using [Source N] tags.\n\n"
    "RETRIEVED CONTEXT:\n"
    "{context_str}\n\n"
    "USER QUESTION:\n"
    "{query_str}\n\n"
    "ANSWER:"
)

QWEN_COMPACT_REFINE_PROMPT_TMPL = (
    "You are a project-aware assistant refining an answer with additional context.\n\n"
    "USER QUESTION: {query_str}\n"
    "CURRENT ANSWER: {existing_answer}\n\n"
    "ADDITIONAL RETRIEVED CONTEXT:\n"
    "{context_msg}\n\n"
    "Rules:\n"
    "1. Update the answer only if the new context provides relevant additional facts.\n"
    "2. Maintain [Source N] citation tags.\n"
    "3. If the new context is not useful, return the CURRENT ANSWER unchanged.\n\n"
    "REFINED ANSWER:"
)

# ── Legacy standard (v1) ─────────────────────────────────────────────────────

STANDARD_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "If the context does not contain the answer, say you lack sufficient information.\n"
    "Query: {query_str}\n"
    "Answer: "
)

STANDARD_REFINE_PROMPT_TMPL = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. "
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)

# ── ReAct agent system prompts ───────────────────────────────────────────────

AGENT_SYSTEM_PROMPT_STANDARD = """You are a company policy and legal document assistant.

LANGUAGE: Reply in English unless the user clearly writes in another language. Never mix Chinese into English answers. Use [Source N] tags only.

Rules:
1. ALWAYS use the policy_search tool for questions about policies, handbook rules, or legal documents.
2. Base answers on retrieved context. If information is insufficient, say so clearly.
3. Cite document name and page when available.
4. Do not invent policies, dates, or legal clauses.
5. For greetings or meta questions, respond directly without the tool.
6. For follow-ups, resolve pronouns from conversation history and pass a complete standalone query to policy_search.
"""

AGENT_SYSTEM_PROMPT_STRICT = """You are a company policy and legal document assistant for employee handbooks and legal PDFs.

LANGUAGE: Reply in English unless the user clearly writes in another language. Never mix Chinese into English answers. Use [Source N] tags only.

FAITHFULNESS IS MANDATORY — an ungrounded answer is worse than no answer.

When using policy_search:
1. Repeat ONLY facts that appear in the tool output. Do NOT infer, speculate, or add outside knowledge.
2. Do NOT invent penalties, benefits, dates, or procedures not explicitly stated in the retrieved text.
3. If the tool output does not answer the question, respond EXACTLY:
   "The provided documents do not contain sufficient information to answer this question reliably."
4. Cite sources using section titles and page numbers from the tool output when stating facts.
5. Do NOT combine unrelated snippets to fabricate a policy.

For greetings or capability questions, respond directly without policy_search.
For follow-ups, expand pronouns from chat history into a complete standalone policy_search query.
"""

AGENT_SYSTEM_PROMPT_BALANCED = """You are a helpful document assistant for indexed PDFs (policies, legal files, and uploaded guides).

LANGUAGE (CRITICAL): Reply in English unless the user clearly writes in another language.
Never answer in Chinese when the user asked in English. Use [Source N] tags only — never 来源 or localized tag names.

When using policy_search:
1. Answer based on retrieved excerpts — do not use outside knowledge.
2. You MAY synthesize related information across multiple retrieved sections when they address the same topic.
3. Do NOT invent penalties, benefits, dates, procedures, acronym definitions, or examples not stated in the retrieved text.
4. Give the most complete helpful answer supported by the excerpts. Start with "Based on the available information..." when coverage is partial.
5. CITATION RULES (MANDATORY): Every factual sentence MUST end with [Source N] tags (1-based, matching the tool output).
   When policy_search returns [Source N] tags, preserve them verbatim in your final response.
   Do not replace tags with page numbers or filenames alone.
6. For LIST or ENUMERATION questions, answer as a numbered list. If a requested item is missing from the tool output, say it was not found in the retrieved excerpts — do not guess.
7. When showing code from tool output, copy lines exactly — never invent functions or complete partial snippets. Scan all retrieved text before claiming code is absent.
8. For named patterns and agent roles, describe only what retrieved text states — do not conflate patterns or invent roles not named in the excerpts. Never expand acronyms unless the excerpt defines them.
9. Abstain only when retrieved text is completely irrelevant or silent on the topic.

For greetings or capability questions, respond directly without policy_search.
For follow-ups, expand pronouns from chat history into a complete standalone policy_search query.
"""

# ── Faithfulness guards ──────────────────────────────────────────────────────

# Strict: rejects unless EVERY claim is directly supported (high faithfulness, low relevancy)
FAITHFULNESS_GUARD_STRICT = """You verify whether an answer is fully supported by document excerpts.

Answer YES only if EVERY factual claim in the ANSWER is directly supported by the EXCERPTS.
Answer NO if the answer adds information, speculates, infers, or states policies not in the excerpts.
Saying information is unavailable is YES if the excerpts truly lack the answer.

Respond with exactly one word: YES or NO

EXCERPTS:
{context}

ANSWER:
{answer}

VERDICT:"""

# Balanced: rejects only CLEAR unsupported claims (allows synthesis and partial answers)
FAITHFULNESS_GUARD_BALANCED = """You check whether an answer contains UNSUPPORTED factual claims.

Answer UNSUPPORTED only if the answer states specific facts (penalties, benefits, numbers, procedures) 
that are clearly NOT present in the EXCERPTS.

Answer SUPPORTED if:
- Claims are paraphrased or summarized from the excerpts
- Related excerpts are reasonably synthesized into a coherent answer
- The answer acknowledges partial coverage or limitations
- The answer says information is unavailable

Respond with exactly one word: SUPPORTED or UNSUPPORTED

EXCERPTS:
{context}

ANSWER:
{answer}

VERDICT:"""

FAITHFULNESS_CLAIM_TRIM_PROMPT = """You revise an answer to remove ONLY unsupported factual claims.

Rules:
- Remove or rephrase sentences that state facts, roles, URLs, code, or definitions not in EXCERPTS.
- Keep supported sentences and all [Source N] tags intact.
- Do not add new facts. Do not invent abstention boilerplate unless nothing supported remains.
- If every sentence is unsupported, respond EXACTLY:
  "{insufficient_message}"

EXCERPTS:
{context}

ORIGINAL ANSWER:
{answer}

REVISED ANSWER:"""

CODE_LINE_VALIDATION_PROMPT_STRICT = """You verify whether every line of CODE in an ANSWER appears in the CONTEXT.

Answer YES only if each non-empty code line in the ANSWER is copied from the CONTEXT (allowing minor whitespace).
Answer NO if any code line, function name, or API call in the ANSWER is not present in the CONTEXT.

Respond in this format:
VERDICT: YES or NO
EXPLANATION: <one sentence>

CONTEXT:
{context}

ANSWER:
{answer}
"""

CODE_LINE_VALIDATION_PROMPT_BALANCED = """You verify whether CODE in an ANSWER is supported by the CONTEXT.

Answer YES if:
- The ANSWER contains no code blocks, OR
- Each function/class definition in the ANSWER appears in the CONTEXT (minor whitespace/formatting OK), OR
- The code illustrates concepts already described in CONTEXT prose

Answer NO only if the ANSWER invents functions, APIs, or code lines with no support in CONTEXT.

Ignore markdown formatting, [Source N] tags, and [CODE BLOCK] prefixes when comparing.
{failed_lines_section}
Respond in this format:
VERDICT: YES or NO
EXPLANATION: <one sentence>

CONTEXT:
{context}

ANSWER:
{answer}
"""

CODE_SELF_CORRECTION_PROMPT = """Rewrite the ANSWER to fix unsupported code lines using ONLY the CONTEXT.

Rules:
- Remove or replace any code line not present in the CONTEXT.
- Copy supported code verbatim from the CONTEXT.
- Keep [Source N] citation tags on factual sentences.
- If code cannot be verified, describe what the excerpts show in prose instead of inventing code.
- Write in English.

QUESTION: {query}

CONTEXT:
{context}

CURRENT ANSWER:
{answer}

CORRECTED ANSWER:"""


def resolve_grounding_mode(
    *,
    strict: bool | None = None,
    version: str | None = None,
) -> GroundingMode:
    """
    Resolve strict vs balanced mode.

    Priority: explicit `strict` arg → STRICT_GROUNDING=true → RESPONSE_PROMPT_VERSION → GROUNDING_STRICTNESS.
    """
    if strict is not None:
        return "strict" if strict else "balanced"
    if settings.strict_grounding:
        return "strict"
    ver = version or settings.response_prompt_version
    if ver == "v2_strict":
        return "strict"
    if ver in ("v2_balanced", "balanced", "v3_qwen_compact"):
        return "balanced"
    if ver == "v1_standard":
        return "balanced"
    return settings.grounding_strictness


def get_text_qa_template(
    *,
    version: str | None = None,
    strict: bool | None = None,
) -> PromptTemplate:
    """Return generation QA template for the active grounding mode."""
    mode = resolve_grounding_mode(strict=strict, version=version)
    if mode == "strict":
        return PromptTemplate(STRICT_TEXT_QA_PROMPT_TMPL)
    ver = version or settings.response_prompt_version
    if ver == "v3_qwen_compact":
        return PromptTemplate(QWEN_COMPACT_QA_PROMPT_TMPL)
    if ver == "v1_standard":
        return PromptTemplate(STANDARD_TEXT_QA_PROMPT_TMPL)
    return PromptTemplate(BALANCED_TEXT_QA_PROMPT_TMPL)


def get_refine_template(
    *,
    version: str | None = None,
    strict: bool | None = None,
) -> PromptTemplate:
    """Return refine template paired with the QA template."""
    mode = resolve_grounding_mode(strict=strict, version=version)
    if mode == "strict":
        return PromptTemplate(STRICT_REFINE_PROMPT_TMPL)
    ver = version or settings.response_prompt_version
    if ver == "v3_qwen_compact":
        return PromptTemplate(QWEN_COMPACT_REFINE_PROMPT_TMPL)
    if ver == "v1_standard":
        return PromptTemplate(STANDARD_REFINE_PROMPT_TMPL)
    return PromptTemplate(BALANCED_REFINE_PROMPT_TMPL)


def get_agent_system_prompt(
    *,
    version: str | None = None,
    strict: bool | None = None,
) -> str:
    """ReAct agent system prompt for the active grounding mode."""
    mode = resolve_grounding_mode(strict=strict, version=version)
    if mode == "strict":
        return AGENT_SYSTEM_PROMPT_STRICT
    ver = version or settings.response_prompt_version
    if ver == "v1_standard":
        return AGENT_SYSTEM_PROMPT_STANDARD
    return AGENT_SYSTEM_PROMPT_BALANCED


def get_faithfulness_guard_prompt(*, mode: GroundingMode | None = None) -> str:
    """Return faithfulness guard prompt for strict or balanced checking."""
    active = mode or resolve_grounding_mode()
    if active == "strict":
        return FAITHFULNESS_GUARD_STRICT
    return FAITHFULNESS_GUARD_BALANCED


def get_faithfulness_claim_trim_prompt() -> str:
    """Prompt for removing unsupported claims while preserving supported content."""
    return FAITHFULNESS_CLAIM_TRIM_PROMPT


def _escape_prompt_braces(text: str) -> str:
    """Escape braces so str.format on prompt templates does not treat code as placeholders."""
    return text.replace("{", "{{").replace("}", "}}")


def get_code_validation_prompt(
    *,
    mode: str = "balanced",
    failed_lines: list[str] | None = None,
) -> str:
    failed_section = ""
    if failed_lines:
        lines = "\n".join(
            f"- {_escape_prompt_braces(line)}" for line in failed_lines[:8]
        )
        failed_section = f"\nLines that failed heuristic check:\n{lines}\n"
    if mode == "strict":
        return CODE_LINE_VALIDATION_PROMPT_STRICT.format(
            context="{context}",
            answer="{answer}",
        )
    return CODE_LINE_VALIDATION_PROMPT_BALANCED.format(
        failed_lines_section=failed_section,
        context="{context}",
        answer="{answer}",
    )


def get_code_self_correction_prompt() -> str:
    return CODE_SELF_CORRECTION_PROMPT


def format_node_for_prompt(node: NodeWithScore, index: int) -> str:
    """
    Format a retrieved chunk with source metadata for the generation prompt.

    XML-style tags make source boundaries explicit and reduce cross-chunk blending.
    """
    meta = node.metadata or {}
    source = meta.get("source_file", "unknown")
    page = meta.get("display_page_number") or meta.get("page_label") or meta.get("page_number")
    section_path = meta.get("section_path") or meta.get("section_title") or ""
    section_number = meta.get("section_number")

    header = f"[Source {index}: {source}"
    if page is not None:
        header += f", p.{page}"
    header += "]"
    if section_path:
        header += f" — {section_path}"
    if section_number and section_number not in str(section_path):
        header += f" (§{section_number})"

    text = (node.get_content() or "").strip()
    return f'<source id="{index}">\n{header}\n{text}\n</source>'


def format_nodes_for_prompt(nodes: list[NodeWithScore]) -> str:
    """Join formatted sources into a single context block for the LLM."""
    if not nodes:
        return "(no document excerpts retrieved)"
    return "\n\n".join(format_node_for_prompt(n, i + 1) for i, n in enumerate(nodes))


def get_generation_config_summary() -> dict[str, str | bool]:
    """Snapshot of generation/grounding settings for eval logs."""
    mode = resolve_grounding_mode()
    return {
        "grounding_strictness": mode,
        "response_prompt_version": settings.response_prompt_version,
        "strict_grounding": settings.strict_grounding,
        "enable_faithfulness_check": settings.enable_faithfulness_check,
        "faithfulness_guard_mode": settings.faithfulness_guard_mode,
        "faithfulness_guard_reject_action": settings.faithfulness_guard_reject_action,
        "enable_code_validation": settings.enable_code_validation,
        "enable_code_self_correction": settings.enable_code_self_correction,
        "code_validation_trigger_mode": settings.code_validation_trigger_mode,
        "code_validation_fail_mode": settings.code_validation_fail_mode,
        "code_validation_judge_mode": settings.code_validation_judge_mode,
    }