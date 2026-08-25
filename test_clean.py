# -*- coding: utf-8 -*-
"""
Helper to write generate_remaining_modules.py cleanly using Python dictionaries
"""
import json

# Let's read generate_100_questions.py original data structure and build the clean Python file
from generate_100_questions import MODULES_DATA

# We already have MODULES_DATA[0] (mod1) and MODULES_DATA[1] (mod2) in generate_100_questions.py
# Let's extract the modules and write them out cleanly.
