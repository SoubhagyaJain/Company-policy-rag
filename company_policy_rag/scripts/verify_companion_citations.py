import json
import re
import sys
import unicodedata
from pathlib import Path
import fitz


def norm(s):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold()))


pdf = fitz.open(sys.argv[1])
data = json.loads(Path(sys.argv[2]).read_text(encoding="utf8"))
for turn in data.get("conversation", []):
    response = turn.get("response", {})
    print("\nTURN", turn["turn"], turn["question"])
    print("ANSWER", response.get("answer", ""))
    print("REWRITE", (response.get("trace") or {}).get("rewritten_query"))
    for c in response.get("citations", []):
        p = c.get("page_number")
        snippet = c.get("snippet", "").removesuffix("...")
        valid = isinstance(p, int) and 1 <= p <= len(pdf)
        page = norm(pdf[p - 1].get_text()) if valid else ""
        sample = norm(snippet)
        direct = bool(sample and sample in page)
        first = " ".join(sample.split()[:25])
        first_found = bool(first and first in page)
        print(
            "CITATION",
            c.get("source_index"),
            "doc_ok",
            c.get("document_id") == data.get("document_id"),
            "page",
            p,
            "label",
            c.get("page_label"),
            "snippet_exact",
            direct,
            "first25",
            first_found,
            "section",
            c.get("section_title"),
        )
        print("SNIPPET", snippet[:600].replace("\n", " "))
