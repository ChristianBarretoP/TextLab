from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from pathlib import Path
from configs.settings import (
    PROMPT_COMPETENCIA_1,
    MODEL_IA_QWEN,
)

# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

print("CUDA disponible:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
    )

# ============================================================
# 2. PROMPT
# ============================================================

SYSTEM_PROMPT = Path(PROMPT_COMPETENCIA_1).read_text(encoding="utf-8")

# ============================================================
# 3. MODELO
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_IA_QWEN)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_IA_QWEN, torch_dtype="auto", device_map="auto"
)

print("Modelo cargado correctamente")
print("Device map: ", model.hf_device_map)


# ============================================================
# 5. FUNÇÃO PARA GERAR UMA WEAK LABEL
# ============================================================
def generate_weak_label(essay):
    user_prompt = (
        "Analise a seguinte redação.\n\n"
        "REDAÇÃO:\n"
        "--------------------\n"
        f"{essay}\n"
        "--------------------"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Formata a conversa no formato esperado pelo Qwen3
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False)

    # Pegamos somente o que o modelo gerou
    generated_tokens = outputs[0, inputs["input_ids"].shape[1] :]

    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # print("=============== SAIDA ==================")
    # print(response_text)
    # print("========================================")

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        result = response_text
    return result
