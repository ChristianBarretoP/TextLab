from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
RESULT_DIR = BASE_DIR / "result"
TEMPORAL_DIR = "tmp"
CACHE_TEM_DATASET_AES_ENEM_DIR = TEMPORAL_DIR / "aes_enem"

PROMPT_ENEM = PROMPTS_DIR / "enem"
PROMPT_COMPETENCIA_1 = PROMPT_ENEM / "competencia1.md"

DATASET_LLM_JBCS = "igorcs/LLM-JBCS"

FILE_WEAK_LABEL_QWEN = "weak_labels_QWEN.json"

MODEL_IA_QWEN = "Qwen/Qwen3-4B"
