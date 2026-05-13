# config.py

# LLM
MODEL = "gpt-5.4"
MAX_RETRIES = 3
TEMPERATURE = 0.1
MAX_TOKENS = 500

# Files
HIERARCHY_JSON = "physical-rearranged-by-synset.json"
PLACEMENTS_CSV = "physical-errors-new.csv"

# Performance / safety
ROOT_DISTANCE = 3
CHECKPOINT_INTERVAL = 20
MAX_SIBLINGS = 10

# Sampling controls
SAMPLE_MODE = None
# Options: None | "first_n" | "random" | "range"

SAMPLE_SIZE  = 5   # used by first_n and random modes
RANGE_START  = 20     # used by range mode
RANGE_END    = 25
RANDOM_SEED  = 2

# Depth-level auditing
# Audit only nodes at exactly this depth (1-indexed, root = depth 1).
# Set to None to audit all nodes beyond ROOT_DISTANCE (original behavior).
#
# Depth reference for physical-rearranged-by-synset.json:
#   depth 2 → artifact, matter, Natural Entities
#   depth 3 → infrastructure, [virtual] solid, [virtual] fluid, …
#   depth 4 → children of infrastructure (opening, fixture, structure, …)
#              children of [virtual] solid (glass, [virtual] crystal, …)
#              ← START HERE: first level a human hasn't fully vetted
AUDIT_DEPTH = 4