import os
import json
import itertools
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Jyoti Multi-Provider Sovereign Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. मॉडल्स की अलग फ़ाइल (models.json) लोड करना
MODELS_FILE = "models.json"
try:
    with open(MODELS_FILE, "r") as f:
        MODEL_CONFIG = json.load(f)
except Exception as e:
    MODEL_CONFIG = {
        "groq": {"default": "llama-3.3-70b-versatile"},
        "gemini": {"default": "gemini-2.5-flash"},
        "sambanova": {"default": "Meta-Llama-3.1-70B-Instruct"},
        "openrouter": {"default": "meta-llama/llama-3.3-70b-instruct:free"},
        "hf": {"default": "meta-llama/Llama-3.2-3B-Instruct"}
    }

# 2. Render Environment से API Keys लोड करना
GROQ_KEY = os.getenv("GROQ_API_OK", "")
GEMINI_KEY = os.getenv("GEMINI-API-OK", "")
SAMBANOVA_KEY = os.getenv("SAMBANOVA-API-OK", "")
OPENROUTER_KEY = os.getenv("OPENROUTER-API-OK", "")

# 10 Hugging Face टोकन्स को ऑटो-डिटेक्ट करके रोटेशन में डालना
hf_tokens = []
for i in range(1, 11):
    val = os.getenv(f"HF-API-{i}")
    if val:
        hf_tokens.append(val)

hf_cycle = itertools.cycle(hf_tokens) if hf_tokens else None

def get_next_hf_key():
    return next(hf_cycle) if hf_cycle else ""

class RequestPayload(BaseModel):
    prompt: str
    provider: str = "auto"  # 'auto' रखने पर बेस्ट फॉलबैक चलेगा

# इंडिविजुअल प्रोवाइडर कॉल्स
def call_groq(prompt: str):
    if not GROQ_KEY:
        return None
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL_CONFIG["groq"]["default"],
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=25
    )
    if res.status_code == 200:
        return {"provider": "groq", "data": res.json()}
    return None

def call_gemini(prompt: str):
    if not GEMINI_KEY:
        return None
    model = MODEL_CONFIG["gemini"]["default"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
    res = requests.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=25
    )
    if res.status_code == 200:
        return {"provider": "gemini", "data": res.json()}
    return None

def call_sambanova(prompt: str):
    if not SAMBANOVA_KEY:
        return None
    res = requests.post(
        "https://api.sambanova.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {SAMBANOVA_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL_CONFIG["sambanova"]["default"],
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=25
    )
    if res.status_code == 200:
        return {"provider": "sambanova", "data": res.json()}
    return None

def call_openrouter(prompt: str):
    if not OPENROUTER_KEY:
        return None
    res = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL_CONFIG["openrouter"]["default"],
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=25
    )
    if res.status_code == 200:
        return {"provider": "openrouter", "data": res.json()}
    return None

def call_hf(prompt: str):
    token = get_next_hf_key()
    if not token:
        return None
    model = MODEL_CONFIG["hf"]["default"]
    res = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": prompt},
        timeout=25
    )
    if res.status_code == 200:
        return {"provider": "hf", "data": res.json()}
    return None

@app.get("/")
def health_check():
    return {
        "status": "online",
        "engine": "Jyoti Render Multi-Provider Engine",
        "loaded_hf_keys_count": len(hf_tokens),
        "providers_active": {
            "groq": bool(GROQ_KEY),
            "gemini": bool(GEMINI_KEY),
            "sambanova": bool(SAMBANOVA_KEY),
            "openrouter": bool(OPENROUTER_KEY),
            "hf": bool(hf_tokens)
        },
        "models": MODEL_CONFIG
    }

@app.post("/v1/chat")
def process_chat(payload: RequestPayload):
    req_prov = payload.provider.lower()

    # अगर यूज़र ने सीधा प्रोवाइडर माँगा है
    if req_prov == "groq":
        res = call_groq(payload.prompt)
        if res: return res
    elif req_prov == "gemini":
        res = call_gemini(payload.prompt)
        if res: return res
    elif req_prov == "sambanova":
        res = call_sambanova(payload.prompt)
        if res: return res
    elif req_prov == "openrouter":
        res = call_openrouter(payload.prompt)
        if res: return res
    elif req_prov == "hf":
        res = call_hf(payload.prompt)
        if res: return res

    # ऑटो फॉलबैक चेन: 1 fail हुआ तो अपने-आप 2, फिर 3, फिर 4...
    providers_chain = [call_groq, call_sambanova, call_gemini, call_openrouter, call_hf]
    for provider_func in providers_chain:
        try:
            result = provider_func(payload.prompt)
            if result:
                return result
        except Exception:
            continue

    raise HTTPException(status_code=503, detail="All AI providers are currently exhausted or rate-limited.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
