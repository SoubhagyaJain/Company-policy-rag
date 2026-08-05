"""One-shot: move interview-notes.html to repo root with refreshed content."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "company_policy_rag" / "docs" / "interview-notes.html"
PLANS = ROOT / "project-plans.html"
OUT = ROOT / "interview-notes.html"
REDIRECT = ROOT / "company_policy_rag" / "docs" / "interview-notes.html"

PITCH = (
    "Employees need policy answers they can trust and verify — not confident hallucinations. "
    "As a solo AI/ML engineer, I built an evaluation-first RAG system across two corpora (308 chunks): "
    "hybrid BM25+dense retrieval, topic pipelines, rerank, grounded generation, faithfulness guard, "
    "code validation, and citation-filtered UI. On the policy benchmark I recovered relevancy from 0.40 "
    "to 0.747 (+87%, run 104356). On the guidebook track I pushed relevancy to 0.766 and hit rate 0.886 "
    "(run 101844, topic pipelines 6508f60). Phase 4 CI and Docker CD are green on GitHub "
    "(runs 27804469869, 27820859129 → soubhagya007/rag-chatbot). Faithfulness prompt tuning (dd40b86) "
    "taught me prompt-only fixes are insufficient when retrieval misses code chunks — I measured that "
    "honestly (run 055058: faith 0.543) instead of hiding regressions."
    "<em>Speak naturally in ~60 seconds. Do not say \"solved hallucinations\" — say reduced unsupported "
    "claims on golden set.</em>"
)

NEW_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover, maximum-scale=5">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="format-detection" content="telephone=no">
  <meta name="theme-color" content="#e4eaf4">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="default">
  <meta name="mobile-web-app-capable" content="yes">
  <title>Rag-chatbot — Interview Notes · Anthropic</title>
  <script>
    (function () {
      var r = document.documentElement;
      r.classList.add('js');
      if (location.protocol === 'file:') r.classList.add('is-file');
      if (/Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || '')) r.classList.add('is-mobile');
    })();
  </script>
  <style>
"""

ENHANCED_JS = r"""
  <button type="button" class="back-to-top" id="back-to-top" aria-label="Back to top" title="Back to top">↑</button>

  <script>
    (function () {
      var toggle = document.getElementById('menu-toggle');
      var closeBtn = document.getElementById('toc-close');
      var panel = document.getElementById('toc-panel');
      var backdrop = document.getElementById('toc-backdrop');
      var backToTop = document.getElementById('back-to-top');
      var links = panel ? panel.querySelectorAll('nav a[href^="#"]') : [];
      var mobileMq = window.matchMedia('(max-width: 900px)');

      function isMobile() { return mobileMq.matches; }

      function openMenu() {
        if (!panel) return;
        panel.classList.add('open');
        if (backdrop) {
          backdrop.classList.add('visible');
          backdrop.setAttribute('aria-hidden', 'false');
        }
        document.body.classList.add('menu-open');
        if (toggle) {
          toggle.setAttribute('aria-expanded', 'true');
          toggle.setAttribute('aria-label', 'Close navigation');
        }
      }

      function closeMenu() {
        if (!panel) return;
        panel.classList.remove('open');
        if (backdrop) {
          backdrop.classList.remove('visible');
          backdrop.setAttribute('aria-hidden', 'true');
        }
        document.body.classList.remove('menu-open');
        if (toggle) {
          toggle.setAttribute('aria-expanded', 'false');
          toggle.setAttribute('aria-label', 'Open navigation');
        }
      }

      if (toggle) {
        toggle.addEventListener('click', function () {
          panel.classList.contains('open') ? closeMenu() : openMenu();
        });
      }
      if (closeBtn) closeBtn.addEventListener('click', closeMenu);
      if (backdrop) backdrop.addEventListener('click', closeMenu);

      links.forEach(function (link) {
        link.addEventListener('click', function () {
          if (isMobile()) closeMenu();
        });
      });

      document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && panel && panel.classList.contains('open')) closeMenu();
      });

      function onBreakpointChange() {
        if (!isMobile() && panel && panel.classList.contains('open')) closeMenu();
      }
      if (mobileMq.addEventListener) mobileMq.addEventListener('change', onBreakpointChange);
      else if (mobileMq.addListener) mobileMq.addListener(onBreakpointChange);
      window.addEventListener('resize', onBreakpointChange);

      function refreshScrollableTables() {
        document.querySelectorAll('.table-wrap').forEach(function (wrap) {
          var scrollable = wrap.scrollWidth > wrap.clientWidth + 2;
          wrap.classList.toggle('is-scrollable', scrollable);
        });
      }
      refreshScrollableTables();
      window.addEventListener('resize', refreshScrollableTables);
      window.addEventListener('load', refreshScrollableTables);

      var sections = [];
      links.forEach(function (link) {
        var id = link.getAttribute('href');
        if (!id || id.charAt(0) !== '#') return;
        var el = document.querySelector(id);
        if (el) sections.push({ link: link, el: el });
      });

      function updateActiveLink() {
        var marker = isMobile() ? 120 : 80;
        var current = sections[0] || null;
        sections.forEach(function (s) {
          if (s.el.getBoundingClientRect().top <= marker) current = s;
        });
        sections.forEach(function (s) {
          s.link.classList.toggle('active', s === current);
        });
      }
      if (sections.length) {
        window.addEventListener('scroll', updateActiveLink, { passive: true });
        window.addEventListener('resize', updateActiveLink);
        updateActiveLink();
      }

      if (backToTop) {
        backToTop.addEventListener('click', function () {
          window.scrollTo({ top: 0, behavior: 'smooth' });
        });
        window.addEventListener('scroll', function () {
          backToTop.classList.toggle('visible', window.scrollY > 480);
        }, { passive: true });
      }

      var copyPitch = document.getElementById('copy-pitch');
      if (copyPitch) {
        copyPitch.addEventListener('click', function () {
          var text = document.getElementById('pitch-text').innerText.split('Speak naturally')[0].trim();
          navigator.clipboard.writeText(text);
          var btn = this;
          btn.textContent = 'Copied!';
          setTimeout(function () { btn.textContent = 'Copy 60s Pitch'; }, 2000);
        });
      }

      var expandAll = document.getElementById('expand-all');
      if (expandAll) expandAll.addEventListener('click', function () {
        document.querySelectorAll('.q-card').forEach(function (d) { d.open = true; });
      });
      var collapseAll = document.getElementById('collapse-all');
      if (collapseAll) collapseAll.addEventListener('click', function () {
        document.querySelectorAll('.q-card').forEach(function (d) { d.open = false; });
      });

      document.querySelectorAll('.fu-link').forEach(function (a) {
        a.addEventListener('click', function () {
          var el = document.querySelector(this.getAttribute('href'));
          if (el) setTimeout(function () { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 50);
        });
      });

      var expandFu = document.getElementById('expand-fu-groups');
      if (expandFu) expandFu.addEventListener('click', function () {
        document.querySelectorAll('.fu-group').forEach(function (g) { g.classList.add('fu-open'); });
      });
      var collapseFu = document.getElementById('collapse-fu-groups');
      if (collapseFu) collapseFu.addEventListener('click', function () {
        document.querySelectorAll('.fu-group').forEach(function (g) { g.classList.remove('fu-open'); });
      });
    })();
  </script>
"""


def extract_diagram(plans: str, diagram_id: str, next_id: str | None = None) -> str:
    start_marker = f'<div class="diagram-card b3b" id="{diagram_id}">'
    start = plans.find(start_marker)
    if start < 0:
        raise RuntimeError(f"diagram {diagram_id} not found in project-plans.html")
    if next_id:
        end_marker = f'<div class="diagram-card b3b" id="{next_id}">'
        end = plans.find(end_marker, start + 1)
        if end < 0:
            raise RuntimeError(f"next diagram {next_id} not found after {diagram_id}")
        block = plans[start:end].rstrip()
    else:
        # Close at next sibling diagram-card without id or section end
        m = re.search(
            rf'{re.escape(start_marker)}.*?\n        </div>\n\n',
            plans[start:],
            re.DOTALL,
        )
        if not m:
            raise RuntimeError(f"could not close diagram {diagram_id}")
        block = m.group(0).rstrip()
    block = block.replace('diagram-hint', 'scroll-hint')
    return block


MOBILE_SHARING_CSS = """
    html.is-file,
    html.is-mobile {
      --glass-bg: #ffffff;
      --glass-bg-strong: #ffffff;
    }

    html.is-file .liquid-bg,
    html.is-mobile .liquid-bg {
      display: none;
    }

    html.is-file .card,
    html.is-file .hero,
    html.is-file .toc,
    html.is-file .mobile-header,
    html.is-file .q-card,
    html.is-file .fu-answer-block,
    html.is-file .glass,
    html.is-file .diagram-card,
    html.is-file .plan-meta,
    html.is-file .pitch-box,
    html.is-file .stat-pill,
    html.is-file .comp-card,
    html.is-file .source-card,
    html.is-file .code-panel,
    html.is-mobile .card,
    html.is-mobile .hero,
    html.is-mobile .toc,
    html.is-mobile .mobile-header,
    html.is-mobile .q-card,
    html.is-mobile .fu-answer-block,
    html.is-mobile .glass,
    html.is-mobile .diagram-card,
    html.is-mobile .plan-meta,
    html.is-mobile .pitch-box,
    html.is-mobile .stat-pill,
    html.is-mobile .comp-card,
    html.is-mobile .source-card,
    html.is-mobile .code-panel {
      backdrop-filter: none !important;
      -webkit-backdrop-filter: none !important;
      background: #ffffff !important;
    }

    @supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
      :root {
        --glass-bg: #ffffff;
        --glass-bg-strong: #ffffff;
      }
      .liquid-bg { display: none; }
      .card, .hero, .toc, .mobile-header, .q-card, .fu-answer-block,
      .glass, .diagram-card, .plan-meta, .pitch-box, .stat-pill, .comp-card,
      .source-card, .code-panel {
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        background: #ffffff !important;
      }
    }

    .file-open-hint {
      display: none;
      position: sticky;
      top: 0;
      z-index: 300;
      padding: 0.65rem 1rem;
      padding-top: max(0.65rem, env(safe-area-inset-top));
      background: #0f172a;
      color: #f8fafc;
      font-size: 0.82rem;
      line-height: 1.45;
      text-align: center;
    }

    html.is-file .file-open-hint {
      display: block;
    }
"""

MOBILE_MENU_OLD = """      .layout {
        grid-template-columns: 1fr;
        display: block;
      }

      .toc {
        position: fixed;
        top: 0;
        left: 0;
        width: min(320px, 88vw);
        max-width: 88vw;
        height: 100vh;
        height: 100dvh;
        z-index: 160;
        transform: translateX(-105%);
        transition: transform 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        border-right: 1px solid var(--glass-border);
        border-bottom: none;
        padding: 1.25rem 1rem 2rem;
        padding-top: max(1.25rem, env(safe-area-inset-top));
        box-shadow: var(--glass-shadow-hover);
      }

      .toc.open { transform: translateX(0); }

      .toc-close { display: flex; align-items: center; justify-content: center; }

      .toc-title { padding-right: 3rem; }

      .toc nav { display: block; }

      .toc nav .toc-section {
        display: block;
        margin-top: 0.75rem;
      }"""

MOBILE_MENU_NEW = """      .layout {
        grid-template-columns: 1fr;
        display: block;
        position: static;
        z-index: auto;
      }

      body.menu-open .layout {
        position: relative;
        z-index: 170;
      }

      .toc-backdrop.visible {
        backdrop-filter: none;
        -webkit-backdrop-filter: none;
        background: rgba(15, 23, 42, 0.5);
      }

      .toc {
        position: fixed;
        top: 0;
        left: 0;
        width: min(320px, 88vw);
        max-width: 88vw;
        height: 100vh;
        height: 100dvh;
        z-index: 180;
        transform: translateX(-105%);
        transition: transform 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        border-right: 1px solid #e2e8f0;
        border-bottom: none;
        padding: 1.25rem 1rem 2rem;
        padding-top: max(1.25rem, env(safe-area-inset-top));
        box-shadow: 4px 0 24px rgba(15, 23, 42, 0.15);
        background: #ffffff !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        color: var(--text-primary);
        visibility: hidden;
        pointer-events: none;
      }

      .toc.open {
        transform: translateX(0);
        visibility: visible;
        pointer-events: auto;
      }

      .toc-close { display: flex; align-items: center; justify-content: center; }

      .toc-title {
        padding-right: 3rem;
        color: #0f172a;
      }

      .toc nav { display: block; }

      .toc nav a {
        color: #334155;
      }

      .toc nav a:hover,
      .toc nav a.active {
        backdrop-filter: none;
        -webkit-backdrop-filter: none;
        background: #ccfbf1;
        color: #0d9488;
      }

      .toc nav .toc-section {
        display: block;
        margin-top: 0.75rem;
        color: #64748b;
      }"""


def patch_css(css: str) -> str:
    css = css.replace(
        "html { scroll-behavior: smooth; }",
        """html {
      scroll-behavior: smooth;
      -webkit-text-size-adjust: 100%;
      text-size-adjust: 100%;
    }

    :focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }

    :focus:not(:focus-visible) {
      outline: none;
    }""",
    )
    css = css.replace(
        """    .menu-toggle {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 44px;
      height: 44px;
      border: 1px solid var(--glass-border-subtle);""",
        """    .menu-toggle {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 44px;
      height: 44px;
      min-width: 44px;
      min-height: 44px;
      border: 1px solid var(--glass-border-subtle);
      touch-action: manipulation;""",
    )
    css = css.replace(
        """    .toc nav a:hover {
      color: var(--accent);
      background: var(--accent-light);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border-color: var(--glass-border-subtle);
      box-shadow: var(--glass-inset);
    }""",
        """    .toc nav a:hover,
    .toc nav a.active {
      color: var(--accent);
      background: var(--accent-light);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border-color: var(--glass-border-subtle);
      box-shadow: var(--glass-inset);
    }

    .toc nav a.active {
      font-weight: 600;
    }""",
    )
    css = css.replace(
        """    .hero-stats {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
    }""",
        """    .hero-stats {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 0.75rem;
    }""",
    )
    css = css.replace(
        "    .stat-pill.metric strong { color: var(--metric); }",
        "    .stat-pill.metric strong { color: var(--metric); }\n    .stat-pill.success strong { color: var(--success); }",
    )
    css = css.replace(
        """    .table-wrap {
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      overscroll-behavior-x: contain;
      margin: 1rem 0 1.5rem;
      border-radius: var(--radius);
      background: var(--glass-bg);
      backdrop-filter: var(--blur);
      -webkit-backdrop-filter: var(--blur);
      border: 1px solid var(--glass-border);
      box-shadow: var(--glass-shadow), var(--glass-inset);
    }""",
        """    .table-wrap {
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      overscroll-behavior-x: contain;
      margin: 1rem 0 1.5rem;
      border-radius: var(--radius);
      background: var(--glass-bg);
      backdrop-filter: var(--blur);
      -webkit-backdrop-filter: var(--blur);
      border: 1px solid var(--glass-border);
      box-shadow: var(--glass-shadow), var(--glass-inset);
      max-width: 100%;
      position: relative;
    }

    .table-wrap.is-scrollable {
      cursor: grab;
    }

    .table-wrap.is-scrollable:active {
      cursor: grabbing;
    }""",
    )
    css = css.replace(
        """    td {
      padding: 0.75rem 1rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.25);
      color: var(--text-secondary);
    }""",
        """    td {
      padding: 0.75rem 1rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.25);
      color: var(--text-secondary);
      vertical-align: top;
    }

    td:last-child {
      min-width: 8rem;
      word-break: break-word;
    }""",
    )
    css = css.replace(
        """    .diagram-viewport {
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      overscroll-behavior-x: contain;
      touch-action: pan-x pan-y;
    }

    .diagram-viewport.tall .b3b-svg {
      min-width: 0;
      width: 100%;
    }

    .diagram-viewport.wide .b3b-svg {
      min-width: 720px;
      width: 100%;
    }""",
        """    .diagram-viewport {
      overflow-x: auto;
      overflow-y: hidden;
      -webkit-overflow-scrolling: touch;
      overscroll-behavior-x: contain;
      touch-action: pan-x pan-y;
      max-width: 100%;
      scrollbar-width: thin;
    }

    .diagram-viewport.tall .b3b-svg {
      min-width: 0;
      width: 100%;
      max-width: 100%;
    }

    .diagram-viewport.wide .b3b-svg {
      min-width: 720px;
      width: 100%;
    }

    @media (max-width: 900px) {
      .diagram-viewport.wide .b3b-svg {
        min-width: 600px;
      }
    }""",
    )
    if ".back-to-top" not in css:
        css = css.replace(
            "    footer a:hover { text-decoration: underline; }",
            """    footer a:hover { text-decoration: underline; }

    .back-to-top {
      display: none;
      position: fixed;
      right: max(1rem, env(safe-area-inset-right));
      bottom: max(1rem, env(safe-area-inset-bottom));
      z-index: 180;
      width: 48px;
      height: 48px;
      border-radius: 50%;
      border: 1px solid var(--glass-border);
      background: var(--glass-bg-strong);
      backdrop-filter: var(--blur);
      -webkit-backdrop-filter: var(--blur);
      box-shadow: var(--glass-shadow);
      color: var(--accent);
      font-size: 1.25rem;
      line-height: 1;
      cursor: pointer;
      touch-action: manipulation;
      -webkit-tap-highlight-color: transparent;
      opacity: 0;
      transform: translateY(12px);
      transition: opacity 0.25s ease, transform 0.25s ease;
    }

    .back-to-top.visible {
      opacity: 1;
      transform: translateY(0);
    }""",
        )
    css = css.replace(
        """      .hero-stats { gap: 0.6rem; }

      .stat-pill {
        font-size: 0.82rem;
        padding: 0.45rem 0.9rem;
      }""",
        """      .hero-stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.5rem;
      }

      .stat-pill {
        font-size: 0.8rem;
        padding: 0.5rem 0.65rem;
        text-align: center;
        justify-content: center;
        border-radius: 12px;
        min-height: 44px;
        display: flex;
        align-items: center;
      }

      .back-to-top { display: flex; align-items: center; justify-content: center; }

      footer p {
        word-break: break-word;
        line-height: 1.75;
      }""",
    )
    css = css.replace(
        "        min-width: 36rem;",
        "        min-width: 32rem;",
    )
    css = css.replace(
        "      .diagram-viewport.wide .b3b-svg { min-width: 640px; }",
        "      .diagram-viewport.wide .b3b-svg { min-width: 520px; }",
    )
    css = css.replace(
        "@media print {\n      .liquid-bg, .mobile-header, .toc-backdrop { display: none !important; }",
        "@media print {\n      .liquid-bg, .mobile-header, .toc-backdrop, .back-to-top { display: none !important; }",
    )
    css = css.replace(
        """    .toc-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.4);
      backdrop-filter: blur(4px);
      -webkit-backdrop-filter: blur(4px);
      z-index: 150;
      opacity: 0;
      transition: opacity 0.25s ease;
    }""",
        """    .toc-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.45);
      z-index: 150;
      opacity: 0;
      transition: opacity 0.25s ease;
      -webkit-tap-highlight-color: transparent;
    }""",
    )
    css = css.replace(
        "--font-serif: 'DM Serif Display', Georgia, serif;",
        "--font-serif: Georgia, 'Times New Roman', serif;",
    )
    css = css.replace(
        "--font-sans: 'Inter', system-ui, -apple-system, sans-serif;",
        "--font-sans: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;",
    )
    if "html.is-file" not in css:
        css = css.replace(
            "    }\n\n    *, *::before, *::after",
            "    }\n" + MOBILE_SHARING_CSS + "\n    *, *::before, *::after",
            1,
        )
    if MOBILE_MENU_OLD in css:
        css = css.replace(MOBILE_MENU_OLD, MOBILE_MENU_NEW)
    return css


def patch_s2_success_table(html: str) -> str:
    extra_rows = """
            <tr><td>Guidebook Relevancy</td><td>≥ 0.75</td><td class="high-score">0.766</td><td>101844</td></tr>
            <tr><td>Guidebook Hit Rate</td><td>&gt; 0.85</td><td class="high-score">0.886</td><td>101844</td></tr>
            <tr><td>Guidebook Faithfulness</td><td>≥ 0.90</td><td>0.594</td><td>101844</td></tr>
            <tr><td>Phase 4 CI + Docker CD</td><td>Green</td><td class="high-score">Done</td><td>27804469869 / 27820859129</td></tr>"""
    return html.replace(
        """            <tr><td>Hit Rate</td><td>&gt; 0.85</td><td>0.867</td><td>104356</td></tr>
          </table></div>""",
        f"""            <tr><td>Hit Rate</td><td>&gt; 0.85</td><td>0.867</td><td>104356</td></tr>{extra_rows}
          </table></div>""",
    )


def patch_s3(html: str, plans: str) -> str:
    rag = extract_diagram(plans, "arch-rag", "arch-citation")
    citation = extract_diagram(plans, "arch-citation", "arch-docker")
    rag = rag.replace(
        "System Architecture<span>RAG Pipeline — Final State</span>",
        "Pipeline<span>RAG + Trust Layer — Final State</span>",
    )
    citation = citation.replace(
        "Citation Architecture<span>Citation Pipeline — Trust Layer</span>",
        "Citations<span>Trust Layer — Source-Linked UI</span>",
    )

    # Keep alignment + eval diagrams from original, replace main pipeline block
    s3_start = html.index('<section id="s3">')
    s3_end = html.index("</section>", s3_start) + len("</section>")
    old_s3 = html[s3_start:s3_end]

    align_start = old_s3.index('<div class="diagram-card b3b">\n          <div class="diagram-title">Alignment Trade-off')
    eval_start = old_s3.index('<div class="diagram-card b3b">\n          <div class="diagram-title">Eval Loop')
    align_block = old_s3[align_start:eval_start]
    eval_block = old_s3[eval_start:]
    eval_block = eval_block.replace(
        "Policy 091001: relv 0.40→0.747 · Guidebook 164848: rel 0.700",
        "Policy 104356: relv 0.747 · Guidebook 101844: rel 0.766 · hit 0.886",
    )
    eval_block = eval_block.replace(
        '<text class="node-sub" x="260" y="158" text-anchor="middle">60 cases</text>',
        '<text class="node-sub" x="260" y="158" text-anchor="middle">60 golden cases</text>',
    )

    new_s3 = f"""<section id="s3">
        <h2>3. End-to-End Architecture</h2>
        {rag}
        {citation}
        {align_block}
        {eval_block}"""
    return html[:s3_start] + new_s3 + html[s3_end:]


def patch_s4(html: str) -> str:
    html = html.replace(
        """            <li>640/64 tokens; metadata: section_path, page_number, file_hash</li>""",
        """            <li>Hierarchical 2000/480 tokens; metadata: section_path, page_number, file_hash, content_type</li>""",
    )
    html = html.replace(
        """          <p><strong>Purpose:</strong> Hybrid BM25+dense RRF → k=30 → bge-reranker-large → top 6 → 40% score filter. Corpus scope via <code>retrieval_scope.py</code>.</p>
          <ul class="bullets">
            <li><code>_PostprocessingRetriever</code> fixes LlamaIndex postprocessor gap</li>
            <li><code>hybrid_retrieval.py</code> + <code>bm25_index.py</code> (ENABLE_HYBRID_BM25=true)</li>
            <li><strong>Trade-off:</strong> +51% precision on policy baseline, +rerank latency on CPU</li>
          </ul>
        </article>""",
        """          <p><strong>Purpose:</strong> Dense k=30 + BM25 k=30 → RRF fusion → bge-reranker-large → top 6 → 40% score filter → topic pipelines → parent expand. Corpus scope via <code>retrieval_scope.py</code>.</p>
          <ul class="bullets">
            <li><code>_PostprocessingRetriever</code> fixes LlamaIndex postprocessor gap</li>
            <li><code>hybrid_retrieval.py</code> + <code>bm25_index.py</code> (ENABLE_HYBRID_BM25=true)</li>
            <li>Comprehensive-list path: multi-query + section-diverse rerank (top_n=12)</li>
            <li><strong>Trade-off:</strong> +51% precision on policy baseline, +rerank latency on CPU</li>
          </ul>
        </article>
        <article class="card">
          <h3>Topic pipelines — Track C</h3>
          <p><strong>Purpose:</strong> Specialized retrieval/generation paths for guidebook patterns, agents, and code.</p>
          <ul class="bullets">
            <li><code>building_block_pipeline.py</code> — enumeration + building-block queries</li>
            <li><code>agent_topic_pipeline.py</code> — agent/workflow definitions (manager_agent, memory)</li>
            <li><code>tool_code_pipeline.py</code> + <code>code_retrieval.py</code> — code/currency cases</li>
            <li>Run <code>101844</code> (<code>6508f60</code>): rel 0.766, hit 0.886 — 5 weak cases remain open</li>
          </ul>
        </article>""",
    )
    html = html.replace(
        """        <article class="card">
          <h3>Evaluation — <code>src/evaluation.py</code></h3>
          <p>60 golden cases (policy + guidebook + subsets). Retrieval + LLM judge. <code>guard_modified</code> + code validation traces. 13 policy runs + Track A guidebook runs in JSON log.</p>
        </article>""",
        """        <article class="card">
          <h3>Evaluation — <code>src/evaluation.py</code></h3>
          <p>60 golden cases (policy + guidebook + subsets). Retrieval + LLM judge. <code>guard_modified</code> + code validation traces. 13 policy runs + Track A/C guidebook runs in JSON log. Eval reproducibility: <code>ENABLE_QUERY_REWRITE=false</code>.</p>
        </article>
        <article class="card">
          <h3>CI/CD — GitHub Actions + Docker Hub</h3>
          <p><strong>CI:</strong> <code>rag-ci.yml</code> — 222 pytest + retrieval smoke (run <code>27804469869</code>). <strong>CD:</strong> Docker image <code>soubhagya007/rag-chatbot</code> (run <code>27820859129</code>).</p>
        </article>""",
    )
    return html


def patch_metrics_bulk(html: str) -> str:
    replacements = [
        ("and 180 tests", "and 222 tests"),
        ("182 pytest tests", "222 pytest tests"),
        ("182 pytest across", "222 pytest across"),
        ("pin versions, 180 tests", "pin versions, 222 tests"),
        ("<strong>180</strong> Tests", "<strong>222</strong> Tests"),
        ("<strong>0.700</strong> Guidebook Rel", '<span class="stat-pill metric success"><strong>0.766</strong> Guidebook Rel</span>\n          <span class="stat-pill metric success"><strong>0.886</strong> Hit Rate'),
        (
            'passed the 35-case relevancy gate at 0.700 (run 164848) and shipped Phase 4 CI green on GitHub',
            "pushed guidebook relevancy to 0.766 and hit rate 0.886 (run 101844, topic pipelines 6508f60). Phase 4 CI and Docker CD are green on GitHub (runs 27804469869, 27820859129)",
        ),
        ("Guidebook: full rel 0.700 (run 164848, gate passed)", "Guidebook: rel 0.766, hit 0.886 (run 101844, exceeds gate)"),
        ("guidebook rel gate 0.700 (run 164848)", "guidebook rel 0.766 (run 101844)"),
        ("guidebook rel gate passed at 0.700 (run 164848)", "guidebook rel 0.766 (run 101844)"),
        ("gate 164848 rel 0.700", "best guidebook run 101844 rel 0.766"),
        ("guidebook gate 164848", "topic pipelines 101844"),
        ("enumeration (164848 rel 0.700)", "enumeration (160052 rel 0.84) + topic pipelines (101844 rel 0.766)"),
        ("→ full guidebook rel 0.700 (164848)", "→ enumeration 0.84 (160052) → topic pipelines rel 0.766 (101844)"),
        ("Run 164848: full guidebook rel 0.700 (gate passed)", "Run 101844: guidebook rel 0.766, hit 0.886 (best)"),
        ("Run 164848: full guidebook rel 0.700; enumeration", "Run 101844: guidebook rel 0.766; enumeration"),
        ("0.629→0.700 on full 35-case run", "0.629→0.766 on full 35-case run (101844)"),
        ("guidebook faith 0.629 baseline", "guidebook faith 0.594 on run 101844"),
        ("guidebook faith 0.629 vs 0.90", "guidebook faith 0.594 vs 0.90"),
        ("guidebook faith 0.629 today", "guidebook faith 0.594 today (run 101844)"),
        ("guidebook rel gate 0.700 (164848)", "guidebook rel 0.766 (101844)"),
        ("code-query rel 0.525 (run 164848)", "code/currency cases still weak (run 101844)"),
        ("640 tokens with 64 overlap", "hierarchical 2000/480 tokens"),
        ("512/64 vs 640/64", "1500/400 vs 2000/480"),
        (
            '<li>Phase 4 CI green on GitHub (run 27804469869); faithfulness tuning dd40b86 (055058)</li>',
            '<li>Phase 4 CI + Docker CD green (runs 27804469869, 27820859129); faithfulness tuning dd40b86 (055058)</li>',
        ),
        (
            "Not wired yet—honest gap. Would run scripts/evaluate.py on PRs",
            "Phase 4 CI green (run 27804469869): 222 pytest + ci_eval_gate.py retrieval smoke. Would add full evaluate.py on PRs",
        ),
        (
            "Docker Compose plus Streamlit plus 182 pytest tests. Phase 4 CI green on GitHub (run 27804469869: pytest + ci_eval_gate.py retrieval smoke). entrypoint.sh waits for Ollama and supports AUTO_INDEX_ON_START.",
            "Docker Compose + Streamlit + 222 pytest tests. Phase 4 CI green (run 27804469869). Docker CD publishes soubhagya007/rag-chatbot (run 27820859129). entrypoint.sh waits for Ollama and supports AUTO_INDEX_ON_START.",
        ),
        (
            "guidebook faithfulness 0.629 baseline on run 164848",
            "guidebook faithfulness 0.594 on run 101844",
        ),
    ]
    for old, new in replacements:
        html = html.replace(old, new)
    return html


def patch_s9(html: str) -> str:
    if "101844" in html.split('<section id="s9">')[1].split("</section>")[0]:
        return html
    return html.replace(
        """          <tr><td>055058</td><td>Faithfulness prompt tuning (dd40b86)</td><td>0.800</td><td>0.543</td><td>0.666</td></tr>
        </table></div>
        <p>Policy per-case: sick_leave relevancy 0.0→0.90. Guidebook: 0.629→0.766 on full 35-case run (101844); enumeration bucket rel 0.84.</p>""",
        """          <tr><td>055058</td><td>Faithfulness prompt tuning (dd40b86)</td><td>0.800</td><td>0.543</td><td>0.666</td></tr>
          <tr class="highlight-row"><td>101844</td><td><strong>Topic pipelines round 2 (<code>6508f60</code>)</strong></td><td class="high-score">0.886</td><td>0.594</td><td class="high-score">0.766</td></tr>
        </table></div>
        <p>Policy per-case: sick_leave relevancy 0.0→0.90. Guidebook: 0.629→0.766 on full 35-case run (101844); enumeration bucket rel 0.84. Open weak cases: pattern_plan_execute, tool_gathering_info, abstention_quantum, tools_real_world, critic_planner_mention.</p>""",
    )


def patch_footer(html: str) -> str:
    return html.replace(
        """      <footer>
        <p><strong>Company Policy RAG</strong> — Interview Notes for Anthropic<br>
        <a href="project-plans.html">Engineering Plans</a> · <a href="gen-interview-notes.html">Generator Docs</a> · <a href="../README3.md">README3</a> · <a href="https://github.com/SoubhagyaJain/Rag-chatbot">GitHub</a></p>
      </footer>""",
        """      <footer>
        <p>
          <strong>Rag-chatbot</strong> — Interview Notes for Anthropic · Updated 2026-06-19<br>
          <a href="README.md">Repo README</a> ·
          <a href="company_policy_rag/README.md">Setup &amp; Architecture</a> ·
          <a href="company_policy_rag/README3.md">README3</a> (status) ·
          <a href="project-plans.html">Engineering Plans</a> ·
          <a href="company_policy_rag/docs/gen-interview-notes.html">Generator Docs</a> ·
          Docker Hub: <a href="https://hub.docker.com/r/soubhagya007/rag-chatbot" target="_blank" rel="noopener">soubhagya007/rag-chatbot</a> ·
          GitHub: <a href="https://github.com/SoubhagyaJain/Rag-chatbot" target="_blank" rel="noopener">SoubhagyaJain/Rag-chatbot</a>
        </p>
        <p style="margin-top:0.75rem;font-style:italic;">
          Built with production-rag principles: measure first, prioritize citation trust, name trade-offs explicitly.
        </p>
      </footer>""",
    )


def main() -> None:
    src = SRC.read_text(encoding="utf-8")
    plans = PLANS.read_text(encoding="utf-8")

    # Head + CSS
    css_match = re.search(r"<style>\s*(.*?)\s*</style>", src, re.DOTALL)
    if not css_match:
        raise RuntimeError("CSS block not found")
    css = patch_css(css_match.group(1))
    html = NEW_HEAD + css + "\n  </style>\n</head>\n" + src.split("</head>", 1)[1]

    # Pitch + hero stats
    html = re.sub(
        r'<div class="pitch-box" id="pitch-text">.*?</div>',
        f'<div class="pitch-box" id="pitch-text">{PITCH}</div>',
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = html.replace(
        """        <div class="hero-stats">
          <span class="stat-pill metric"><strong>0.747</strong> Policy Rel</span>
          <span class="stat-pill metric success"><strong>0.766</strong> Guidebook Rel</span>
          <span class="stat-pill metric success"><strong>0.886</strong> Hit Rate
          <span class="stat-pill metric"><strong>0.84</strong> Enum Rel</span>
          <span class="stat-pill"><strong>308</strong> Chunks</span>
          <span class="stat-pill"><strong>222</strong> Tests</span>
          <span class="stat-pill"><strong>60</strong> Golden Cases</span>
        </div>""",
        """        <div class="hero-stats">
          <span class="stat-pill metric"><strong>0.747</strong> Policy Rel</span>
          <span class="stat-pill metric success"><strong>0.766</strong> Guidebook Rel</span>
          <span class="stat-pill metric success"><strong>0.886</strong> Hit Rate</span>
          <span class="stat-pill metric"><strong>0.84</strong> Enum Rel</span>
          <span class="stat-pill"><strong>308</strong> Chunks</span>
          <span class="stat-pill"><strong>222</strong> Tests</span>
          <span class="stat-pill success"><strong>CD</strong> Docker Green</span>
          <span class="stat-pill"><strong>60</strong> Golden Cases</span>
        </div>""",
    )
    # Fix hero stats if not yet updated (first run path)
    if "<strong>0.700</strong> Guidebook Rel" in html:
        html = html.replace(
            """        <div class="hero-stats">
          <span class="stat-pill metric"><strong>0.747</strong> Policy Rel</span>
          <span class="stat-pill metric"><strong>0.700</strong> Guidebook Rel</span>
          <span class="stat-pill metric"><strong>0.84</strong> Enum Rel</span>
          <span class="stat-pill"><strong>308</strong> Chunks</span>
          <span class="stat-pill"><strong>180</strong> Tests</span>
          <span class="stat-pill"><strong>60</strong> Golden Cases</span>
        </div>""",
            """        <div class="hero-stats">
          <span class="stat-pill metric"><strong>0.747</strong> Policy Rel</span>
          <span class="stat-pill metric success"><strong>0.766</strong> Guidebook Rel</span>
          <span class="stat-pill metric success"><strong>0.886</strong> Hit Rate</span>
          <span class="stat-pill metric"><strong>0.84</strong> Enum Rel</span>
          <span class="stat-pill"><strong>308</strong> Chunks</span>
          <span class="stat-pill"><strong>222</strong> Tests</span>
          <span class="stat-pill success"><strong>CD</strong> Docker Green</span>
          <span class="stat-pill"><strong>60</strong> Golden Cases</span>
        </div>""",
        )

    html = patch_s2_success_table(html)
    html = patch_s3(html, plans)
    html = patch_s4(html)
    html = patch_metrics_bulk(html)
    html = patch_s9(html)
    html = patch_footer(html)

    # Menu toggle aria
    html = html.replace(
        '<button class="menu-toggle" id="menu-toggle" type="button" aria-label="Open navigation">',
        '<button class="menu-toggle" id="menu-toggle" type="button" aria-label="Open navigation" aria-expanded="false" aria-controls="toc-panel">',
    )

    # Replace old script with enhanced JS
    html = re.sub(r"\s*<script>.*?</script>\s*</body>", ENHANCED_JS + "\n</body>", html, flags=re.DOTALL)

    OUT.write_text(html, encoding="utf-8")
    REDIRECT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT} ({len(html)} chars)")
    print(f"Wrote self-contained copy {REDIRECT}")


if __name__ == "__main__":
    main()