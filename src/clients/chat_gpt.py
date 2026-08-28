import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from configs.settings import (
    PROMPT_COMPETENCIA_1,
    MODEL_IA_GPT
)
from schemas.enem.competencia1 import OUTPUT_SCHEMA

# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================
load_dotenv()
API_KEY_SECRET = os.getenv("OPEN_IA_API_KEY")

# ============================================================
# 2. PROMPT
# ============================================================
SYSTEM_PROMPT = Path(PROMPT_COMPETENCIA_1).read_text(encoding="utf-8")

# ============================================================
# 3. OBTENER A KEY API OPEN AI GPT
# ============================================================
client = OpenAI(api_key=API_KEY_SECRET)

# ============================================================
# 5. FUNÇÃO PARA GERAR UMA WEAK LABEL
# ============================================================
def generate_weak_label(essay):
    response = client.responses.create(
        model=MODEL_IA_GPT,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analise a seguinte redação.\n\n"
                    "REDAÇÃO:\n"
                    "--------------------\n"
                    f"{essay}\n"
                    "--------------------"
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "weak_concept_labels",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
    )

    return json.loads(response.output_text)