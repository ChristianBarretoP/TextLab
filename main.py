import os
import json
from datasets import load_dataset
from tqdm import tqdm
from configs.settings import (
    DATASET_LLM_JBCS,
    CACHE_TEM_DATASET_AES_ENEM_DIR,
    RESULT_DIR,
    FILE_WEAK_LABEL_GPT,
    FILE_WEAK_LABEL_QWEN,
    FILE_WEAK_LABEL_LLAMA,
    FILE_WEAK_LABEL_DEEP_SEEK,
)
from src.clients import chat_gpt

# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================
MODEL_IA = chat_gpt
FILE_OUT_RESULT = FILE_WEAK_LABEL_GPT

# ============================================================
# 1. CARREGAR DATASET
# ============================================================
dataset = load_dataset(
    DATASET_LLM_JBCS, cache_dir=CACHE_TEM_DATASET_AES_ENEM_DIR, trust_remote_code=True
)["train"]


# ============================================================
#  2. CARREGAR OU SALVAR OS RESULTADOS
# ============================================================
def load_results(output_file):
    """
    Carrega resultados existentes.
    Se o arquivo não existir, retorna lista vazia.
    """

    if not os.path.exists(output_file):
        return []

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    except json.JSONDecodeError:
        print("O arquivo JSON está corrompido.")
        print("Iniciando com resultados vazios.")
        return []


def save_results(results, output_file):
    """
    Salva os resultados imediatamente.
    """

    # cria o diretório caso não exista
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # arquivo temporário
    temp_file = output_file + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as f:

        json.dump(results, f, ensure_ascii=False, indent=4)

        # garante que os dados sejam escritos
        f.flush()
        os.fsync(f.fileno())

    # substitui o arquivo antigo somente depois
    # que o novo foi completamente escrito
    os.replace(temp_file, output_file)


# ============================================================
#  3. EXECUTA O MODELO
# ============================================================
OUTPUT_FILE = os.path.join(RESULT_DIR, FILE_OUT_RESULT)

results = load_results(OUTPUT_FILE)

print(f"Resultados já salvos: {len(results)}")

processed_ids = {item["id"] for item in results}

print(f"IDs já processados: {len(processed_ids)}")

iter = tqdm(dataset, total=len(dataset))
for i, row in enumerate(iter):
    id_essay = f"{row['id']}-{row['id_prompt']}"

    # ============================================
    # Já processado?
    # ============================================
    if id_essay in processed_ids:
        continue
    essay = row["essay_text"]
    try:
        weak_label = MODEL_IA.generate_weak_label(essay)

        result = {
            "id": id_essay,
            "weak_label": weak_label,
        }
        results.append(result)

        # adiciona ao conjunto
        processed_ids.add(id_essay)

        save_results(results, OUTPUT_FILE)

    except Exception as e:

        print(f"\nErro no ensaio {id_essay}: {e}")
        continue

    iter.set_description(f"[Weak label: Total {i+1}/{len(dataset)}]")
