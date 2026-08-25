# -*- coding: utf-8 -*-
"""
Full repository of 100 Technical Interview Questions and Answers with exact code snippets from the codebase
"""
import html
from full_qa_data import ALL_MODULES
from module_2_data import MOD2
from modules_3_to_10_data import MOD3_TO_10_LIST
from modules_5_to_10_data import MOD5_TO_10_LIST

# Combine all 10 modules cleanly
MODULES_DATA = [
    ALL_MODULES[0],           # Module 1 (Q01-Q10)
    MOD2,                     # Module 2 (Q11-Q20)
    MOD3_TO_10_LIST[0],       # Module 3 (Q21-Q30)
    MOD3_TO_10_LIST[1],       # Module 4 (Q31-Q40)
    MOD5_TO_10_LIST[0],       # Module 5 (Q41-Q50)
    MOD5_TO_10_LIST[1],       # Module 6 (Q51-Q60)
    MOD5_TO_10_LIST[2],       # Module 7 (Q61-Q70)
    MOD5_TO_10_LIST[3],       # Module 8 (Q71-Q80)
    MOD5_TO_10_LIST[4],       # Module 9 (Q81-Q90)
    MOD5_TO_10_LIST[5]        # Module 10 (Q91-Q100)
]

def get_all_modules_html():
    html_parts = []
    
    html_parts.append("""
<!-- ═══════════════════════════════════════════════════════════
     SECTION 09 — TOP 100 TECHNICAL INTERVIEW QUESTIONS & ANSWERS
     ═══════════════════════════════════════════════════════════ -->
<section class="section" id="interview">
  <div class="section-number">09</div>
  <div class="section-eyebrow">Complete Defense Repository</div>
  <h2 class="section-title">Top 100 Technical Interview Questions & Answers</h2>
  <p class="section-subtitle">Comprehensive, battle-tested answers covering every layer of the architecture, algorithms, failure modes, and code references. Filter by module or search keywords.</p>

  <span class="annotation--warning annotation">Click any question to reveal Short Answer + Deep Technical Answer + Actual Codebase Snippet.</span>

  <!-- Interactive Explorer Toolbar -->
  <div class="qa-controls mt-2xl">
    <div class="qa-search-row">
      <input type="text" class="qa-search-input" placeholder="🔍 Search across all 100 questions, answers, formulas, and files (e.g. 'RRF', '4D verifier', 'QLoRA', 'Ollama', 'ChromaDB')...">
      <button class="qa-action-btn qa-action-btn--toggle">Expand All</button>
    </div>
    
    <div class="qa-filter-pills">
      <button class="qa-pill active" data-filter="all">All (100)</button>
      <button class="qa-pill" data-filter="mod1">1. Architecture (10)</button>
      <button class="qa-pill" data-filter="mod2">2. Ingestion & Chunking (10)</button>
      <button class="qa-pill" data-filter="mod3">3. Metadata & Indexing (10)</button>
      <button class="qa-pill" data-filter="mod4">4. Hybrid Retrieval & RRF (10)</button>
      <button class="qa-pill" data-filter="mod5">5. Cross-Encoder Rerank (10)</button>
      <button class="qa-pill" data-filter="mod6">6. Generation & SSE (10)</button>
      <button class="qa-pill" data-filter="mod7">7. 4D Verifier & Retry (10)</button>
      <button class="qa-pill" data-filter="mod8">8. Cache & Concurrency (10)</button>
      <button class="qa-pill" data-filter="mod9">9. QLoRA & Fine-Tuning (10)</button>
      <button class="qa-pill" data-filter="mod10">10. Testing & Scale (10)</button>
    </div>

    <div class="qa-stats-bar">
      <span class="qa-visible-count">Showing 100 of 100 questions</span>
      <span>Tip: Press ESC to collapse open questions</span>
    </div>
  </div>

  <div class="qa-questions-container">
""")

    for mod in MODULES_DATA:
        mod_id = mod["id"]
        mod_title = mod["title"]
        mod_badge = mod["badge"]
        
        html_parts.append(f"""
    <!-- {mod_title} -->
    <div class="qa-module-header" data-module="{mod_id}">
      <h3 class="qa-module-title">{mod_title}</h3>
      <span class="qa-module-badge">{mod_badge}</span>
    </div>
""")
        
        for q in mod["questions"]:
            num = q["num"]
            level = q["level"]
            level_text = q["level_text"]
            question_text = html.escape(q["q"])
            short_ans = html.escape(q["short"])
            deep_ans = q["deep"].replace("\n", "<br>")
            code_ref = html.escape(q["code"])
            
            snippet_code = q.get("snippet", "")
            snippet_file = q.get("file", "company_policy_rag/backend/rag/pipeline.py")
            snippet_lang = q.get("lang", "python")
            
            snippet_html = ""
            if snippet_code:
                escaped_snippet = html.escape(snippet_code.strip())
                snippet_html = f"""
          <div class="code-snippet-box">
            <div class="code-snippet-header">
              <span class="code-snippet-file">{html.escape(snippet_file)}</span>
              <span class="code-snippet-lang">{snippet_lang}</span>
            </div>
            <pre class="code-snippet-pre"><code>{escaped_snippet}</code></pre>
          </div>"""
            
            html_parts.append(f"""
    <div class="question-block" data-module="{mod_id}">
      <div class="question-header">
        <span class="question-num">{num}</span>
        <span class="question-level question-level--{level}">{level_text}</span>
        <span class="question-text">{question_text}</span>
        <span class="question-toggle">+</span>
      </div>
      <div class="question-answer">
        <div class="answer-content">
          <div class="answer-section">
            <div class="answer-label">30-Second Verbal Answer (The Interview Pitch)</div>
            <div class="answer-text">{short_ans}</div>
          </div>
          <div class="answer-section">
            <div class="answer-label">Deep Technical Answer (Architecture, Formulas & Tradeoffs)</div>
            <div class="answer-text" style="line-height:1.8">{deep_ans}</div>
          </div>
          <div class="answer-section" style="margin-top:var(--space-md);padding-top:var(--space-sm);border-top:1px dashed var(--border-light)">
            <div class="answer-label" style="color:var(--text-muted)">Code References & Implementation Details</div>
            <div class="answer-code">{code_ref}</div>
            {snippet_html}
          </div>
        </div>
      </div>
    </div>
""")

    html_parts.append("""
  </div><!-- .qa-questions-container -->
</section>
""")
    
    return "\n".join(html_parts)
