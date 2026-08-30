from pathlib import Path
import re

import fitz


pdf_path = Path("app/storage/uploads/doc_6853a9a01bf7_AI Engineering Guidebook (2).pdf")
document = fitz.open(pdf_path)
pattern = re.compile(r"fine.?tun|lora|qlora|prompt tuning|adapter tuning", re.IGNORECASE)

for page_index, page in enumerate(document):
    text = page.get_text("text")
    if not pattern.search(text):
        continue
    lines = text.splitlines()
    hit_lines = [index for index, line in enumerate(lines) if pattern.search(line)]
    excerpts = []
    for line_index in hit_lines[:5]:
        start = max(0, line_index - 4)
        end = min(len(lines), line_index + 13)
        excerpts.append("\n".join(lines[start:end]))
    print(f"\n===== PDF PAGE {page_index + 1} =====\n" + "\n---\n".join(excerpts))
