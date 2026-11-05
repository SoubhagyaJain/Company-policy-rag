"""Rebuild the small, fixed Apollo benchmark from the public SQuAD 1.1 dev set."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
PARAGRAPHS = (4, 8, 16, 28, 36, 44, 52, 56)


def main() -> None:
    with urlopen(SOURCE, timeout=30) as response:
        dataset = json.load(response)
    article = next(a for a in dataset["data"] if a["title"] == "Apollo_program")
    base = json.loads((ROOT / "data/eval/retrieval/handbook_corpus.json").read_text(encoding="utf-8"))[0]
    chunks = []
    for i, paragraph in enumerate(article["paragraphs"]):
        chunk = copy.deepcopy(base)
        chunk.update(id=f"apollo_{i:03}", text=paragraph["context"])
        chunk["metadata"].update(
            document_id="squad_apollo_program", source_file="Wikipedia_Apollo_program_SQuAD_v1.1",
            file_path="SQuAD/dev-v1.1/Apollo_program", file_hash="squad-dev-v1.1",
            chunk_index=i, section_title="Apollo program", section_path="Apollo program",
            page_number=None, internal_page_index=None, display_page_number=None, page_label=None,
            parent_id=None, child_ids=[],
        )
        chunks.append(chunk)
    queries = []
    for i in PARAGRAPHS:
        question = article["paragraphs"][i]["qas"][0]
        queries.append(dict(
            id=question["id"], question=question["question"], category="squad_extractive",
            relevant={f"apollo_{i:03}": 2}, keywords=[question["answers"][0]["text"]],
            should_abstain=False, answers=[a["text"] for a in question["answers"]],
        ))
    target = ROOT / "data/eval/retrieval"
    (target / "squad_apollo_corpus.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    labels = dict(corpus="squad_apollo", corpus_source="SQuAD v1.1 dev, Apollo_program; original paragraphs",
                  query_source="First question in eight spaced paragraphs; relevance labels are not exhaustive", queries=queries)
    (target / "squad_apollo_labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} passages and {len(queries)} questions")


if __name__ == "__main__":
    main()
