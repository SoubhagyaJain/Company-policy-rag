# -*- coding: utf-8 -*-
"""
Construct questions_dataset.json with all 100 questions and full codebase snippets
"""
import json

def get_complete_dataset():
    # Module 1
    from full_qa_data import ALL_MODULES
    dataset = list(ALL_MODULES) # has mod 1
    
    # Module 2
    from generate_100_questions import MODULES_DATA
    # In generate_100_questions, MODULES_DATA[1] is Module 2
    if len(MODULES_DATA) > 1:
        dataset.append(MODULES_DATA[1])
        
    # Modules 3 to 10
    # Let's import or append modules 3 to 10
    from modules_3_to_10_data import MOD3_TO_10_LIST
    dataset.extend(MOD3_TO_10_LIST)
    
    return dataset

if __name__ == "__main__":
    from modules_3_to_10_data import MOD3_TO_10_LIST
    from full_qa_data import ALL_MODULES
    from generate_100_questions import MODULES_DATA
    
    full_data = [ALL_MODULES[0], MODULES_DATA[1]] + MOD3_TO_10_LIST
    with open("questions_dataset.json", "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    print(f"Successfully compiled questions_dataset.json with {len(full_data)} modules and {sum(len(m['questions']) for m in full_data)} questions.")
