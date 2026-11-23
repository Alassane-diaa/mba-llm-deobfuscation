import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
THINKING = os.getenv("GEMINI_THINKING", "minimal")
CACHE = Path("results/cache")
CACHE.mkdir(parents=True, exist_ok=True)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(http_options={"timeout": 300_000})  # 5 min, en millisecondes
    return _client


def ask(prompt, system=None, retries=5, max_tokens=None, cache_namespace=None):
    """Appelle Gemini. Cache disque par (modèle, réflexion, système, prompt)."""
    cache_identity = f"{MODEL}|{THINKING}"
    if cache_namespace:
        cache_identity += f"|run={cache_namespace}"
    if max_tokens is not None:
        cache_identity += f"|max{max_tokens}"
    key = hashlib.sha256(
        f"{cache_identity}\n{system}\n{prompt}".encode("utf-8")
    ).hexdigest()
    cache_file = CACHE / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["text"]

    config_kwargs = {
        "temperature": 0,
        "system_instruction": system,
        "thinking_config": types.ThinkingConfig(thinking_level=THINKING),
    }
    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = max_tokens
    config = types.GenerateContentConfig(**config_kwargs)

    for attempt in range(retries):
        try:
            resp = _get_client().models.generate_content(
                model=MODEL, contents=prompt, config=config)
            text = resp.text or ""
            cache_file.write_text(json.dumps({"text": text}, ensure_ascii=False),
                                  encoding="utf-8")
            return text
        except genai_errors.ServerError as e:  # 5xx, temporaire : on réessaie
            wait = 15 * (attempt + 1)
            print(f"[llm] erreur serveur ({e.code}), nouvel essai dans {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
        except genai_errors.ClientError as e:  # 4xx : on arrête et on affiche
            if getattr(e, "code", None) == 429 and attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"[llm] quota atteint, nouvel essai dans {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Gemini : trop d'erreurs, arrêt.")