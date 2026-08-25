# -*- coding: utf-8 -*-
"""
Complete populator that creates questions_dataset.json with all 100 questions (Q01-Q100)
including Short Answer, Deep Technical Answer, Code References, and actual Python/Config snippets.
"""
import json

def build_dataset():
    # Module 1
    from full_qa_data import ALL_MODULES
    mod1 = ALL_MODULES[0]
    
    # Module 2
    from generate_100_questions import MODULES_DATA
    mod2 = MODULES_DATA[1]
    
    # Module 3 and 4
    from modules_3_to_10_data import MOD3_TO_10_LIST
    mod3 = MOD3_TO_10_LIST[0]
    mod4 = MOD3_TO_10_LIST[1]
    
    # Modules 5 to 10
    from modules_5_to_10_data import MOD5_TO_10_LIST
    
    all_10 = [mod1, mod2, mod3, mod4] + MOD5_TO_10_LIST
    return all_10

if __name__ == "__main__":
    from modules_5_to_10_data import MOD5_TO_10_LIST
    all_data = build_dataset()
    with open("questions_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated questions_dataset.json with {len(all_data)} modules and {sum(len(m['questions']) for m in all_data)} questions.")
