import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE = Path("results/cache")
CACHE.mkdir(parents=True, exist_ok=True)

MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "512"))

# nom : (URL de base, variable de clé, clé fixe)
PRESETS = {
    "groq":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY", None),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", None),
    "ollama":     ("http://localhost:11434/v1", None, "ollama"),
    "custom":     (None, "LLM_API_KEY", None),  # URL dans LLM_BASE_URL
}


def _gemini(model):
    if model:
        raise SystemExit("Pour gemini, le modèle se règle dans .env (GEMINI_MODEL).")
    from src import llm  # module existant, non modifié
    return llm.ask, f"gemini|{llm.MODEL}|{llm.THINKING}"


def _openai_compat(name, model):
    from openai import OpenAI, APIStatusError

    base, key_env, fixed_key = PRESETS[name]
    base = base or os.getenv("LLM_BASE_URL")
    api_key = fixed_key or os.getenv(key_env)
    model = model or os.getenv("LLM_MODEL")
    if not (base and api_key and model):
        raise SystemExit(f"Config incomplète pour {name} : URL, clé ou modèle manquant.")

    client = OpenAI(base_url=base, api_key=api_key, timeout=300)
    tag = f"{name}|{base}|{model}|max{MAX_OUTPUT_TOKENS}"

    def ask(prompt, system=None, retries=5):
        key = hashlib.sha256(f"{tag}\n{system}\n{prompt}".encode("utf-8")).hexdigest()
        cache_file = CACHE / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["text"]

        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})

        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                text = resp.choices[0].message.content or ""
                cache_file.write_text(json.dumps({"text": text}, ensure_ascii=False),
                                      encoding="utf-8")
                return text
            except APIStatusError as e:
                if e.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    wait = 30 * (attempt + 1)
                    print(f"[llm] erreur {e.status_code}, nouvel essai dans {wait}s",
                          file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Trop d'erreurs du fournisseur, arrêt.")

    return ask, tag


def get_provider(name, model=None):
    """Retourne (ask, tag). Le tag identifie fournisseur, URL, modèle et limite de sortie."""
    if name == "gemini":
        return _gemini(model)
    if name in PRESETS:
        return _openai_compat(name, model)
    raise SystemExit(f"Fournisseur inconnu : {name}. Choix : gemini, {', '.join(PRESETS)}")