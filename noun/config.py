# config.py

# LLM
MODEL = "gpt-5.4"
MAX_RETRIES = 3
TEMPERATURE = 0.1
MAX_TOKENS = 500

# Files
HIERARCHY_JSON = "noun/nodes-data-condensed.json"
PLACEMENTS_CSV = "noun/new_placements_combo.csv"

# Performance / safety
CHECKPOINT_INTERVAL = 20
MAX_SIBLINGS = 10

# Sampling controls
SAMPLE_MODE = "first_n"
# Options: None | "first_n" | "random" | "range"

SAMPLE_SIZE  = 5   # used by first_n and random modes
RANGE_START  = 0     # used by range mode
RANGE_END    = 100
RANDOM_SEED  = 2
