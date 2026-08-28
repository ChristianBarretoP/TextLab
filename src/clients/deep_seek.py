from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
from pathlib import Path
from configs.settings import (
    PROMPT_COMPETENCIA_1,
    MODEL_IA_DEEP_SEEK,
)

# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_IA_DEEP_SEEK)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_IA_DEEP_SEEK,
    torch_dtype=torch.float16,
    device_map="auto", # Faz o modelo rodar na sua placa de vídeo local
    trust_remote_code=True
)

print("Modelo cargado correctamente")
print(f"Modelo carregado na GPU: {model.device}")
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
            max_new_tokens=1024,
            temperature=0.6,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Pegamos somente o que o modelo gerou
    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    resposta_completa = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    resposta_final = resposta_completa[len(text):].strip()


    # print("RESPOSTA DO MODELO:")
    # print(resposta_final)
    
    # Extrai estritamente o objeto JSON delimitado pelas chaves {}
    match = re.search(r"\{.*\}", resposta_final, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            result = json_str
        return result
    else:
        raise ValueError("Nenhum bloco JSON válido foi encontrado no retorno do modelo.")