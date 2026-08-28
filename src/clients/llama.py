from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from configs.settings import (
    PROMPT_COMPETENCIA_1,
    MODEL_IA_LLAMA
)

# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================
load_dotenv()
HF_TOKEN = os.getenv("LLAMA_API_KEY")  # Seu token de leitura (Read token)

print("CUDA disponible:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")

# ============================================================
# 2. PROMPT
# ============================================================

SYSTEM_PROMPT = Path(PROMPT_COMPETENCIA_1).read_text(encoding="utf-8")


# ============================================================
# 3. MODELO
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_IA_LLAMA, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_IA_LLAMA,
    token=HF_TOKEN,
    torch_dtype=torch.float16,
    device_map="auto" # Faz o modelo rodar na sua placa de vídeo local
)

print("Modelo cargado correctamente")
print("Device map: ", model.hf_device_map)

# ============================================================
# 4. FUNÇÃO PARA GERAR UMA WEAK LABEL
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
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # Formata a conversa no formato esperado pelo Qwen3
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False,
            temperature=0.0,
        )

    # Pegamos somente o que o modelo gerou
    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    # print("RESPOSTA DO MODELO:")
    # print(response_text)
    # Extrai estritamente o objeto JSON delimitado pelas chaves {}
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            result = json_str
        return result
    else:
        raise ValueError("Nenhum bloco JSON válido foi encontrado no retorno do modelo.")
