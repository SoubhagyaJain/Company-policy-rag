# -*- coding: utf-8 -*-
"""
Populates questions_dataset.json with all 100 questions across all 10 modules
with complete answers, exact formulas, and real codebase snippets.
"""
import json

def get_all_modules():
    mods = []
    
    # We will build all 10 modules
    # Let's import the data definitions
    from full_qa_data import ALL_MODULES
    return ALL_MODULES

if __name__ == "__main__":
    from full_qa_data import ALL_MODULES
    with open("questions_dataset.json", "w", encoding="utf-8") as f:
        json.dump(ALL_MODULES, f, indent=2, ensure_ascii=False)
    print(f"questions_dataset.json saved with {len(ALL_MODULES)} modules and {sum(len(m['questions']) for m in ALL_MODULES)} questions.")
