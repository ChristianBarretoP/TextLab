from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
RESULT_DIR = BASE_DIR / "result"
TEMPORAL_DIR = Path("tmp")
CACHE_TEM_DATASET_AES_ENEM_DIR = TEMPORAL_DIR / "aes_enem"

PROMPT_ENEM = PROMPTS_DIR / "enem"
PROMPT_COMPETENCIA_1 = PROMPT_ENEM / "competencia1.md"

DATASET_LLM_JBCS = "igorcs/LLM-JBCS"

FILE_WEAK_LABEL_GPT = "weak_labels_GPT.json"
FILE_WEAK_LABEL_QWEN = "weak_labels_QWEN.json"
FILE_WEAK_LABEL_LLAMA = "weak_labels_LLAMA.json"
FILE_WEAK_LABEL_DEEP_SEEK = "weak_labels_DEEP_SEEK.json"

MODEL_IA_GPT = "gpt-5.5"
MODEL_IA_QWEN = "Qwen/Qwen3-4B"
MODEL_IA_LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_IA_DEEP_SEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
