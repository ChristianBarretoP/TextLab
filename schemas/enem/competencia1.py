# ============================================================
# JSON SCHEMA
# ============================================================
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "estrutura_sintatica": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 4},
                "evidencias": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "trecho": {"type": "string"},
                            "tipo": {"type": "string"},
                            "explicacao": {"type": "string"},
                        },
                        "required": ["trecho", "tipo", "explicacao"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["score", "evidencias"],
            "additionalProperties": False,
        },
        "desvios": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 3},
                "evidencias": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "trecho": {"type": "string"},
                            "tipo": {"type": "string"},
                            "explicacao": {"type": "string"},
                        },
                        "required": ["trecho", "tipo", "explicacao"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["score", "evidencias"],
            "additionalProperties": False,
        },
    },
    "required": ["estrutura_sintatica", "desvios"],
    "additionalProperties": False,
}