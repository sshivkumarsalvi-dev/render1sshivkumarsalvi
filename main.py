import os
import json
import time
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Jyoti Sovereign Engine - Anti-Bot ZeroGPU Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}

CONFIGS = {
    "groq": load_json("groq_models.json"),
    "gemini": load_json("gemini_models.json"),
    "sambanova": load_json("sambanova_models.json"),
    "openrouter": load_json("openrouter_models.json"),
    "hf_spaces": load_json("hf_spaces.json")
}

KEYS = {
    "groq": os.getenv("GROQ_API_OK", ""),
    "gemini": os.getenv("GEMINI-API-OK", ""),
    "sambanova": os.getenv("SAMBANOVA-API-OK", ""),
    "openrouter": os.getenv("OPENROUTER-API-OK", "")
}

# 1 से 10 HF टोकन्स लोड करना
HF_TOKENS = []
for i in range(1, 11):
    val = os.getenv(f"HF-API-{i}")
    if val:
        HF_TOKENS.append(val)

# HF टोकन स्टेट ट्रैकर: GPU Seconds + Anti-Bot Cooldown Timestamp
HF_TOKEN_STATS = [
    {
        "id": i + 1,
        "token": tok,
        "used_seconds": 0.0,
        "exhausted": False,
        "last_exhausted_time": 0.0,  # 2 मिनट कूलडाउन मापने के लिए
        "last_reset": time.time()
    }
    for i, tok in enumerate(HF_TOKENS)
]

CURRENT_HF_INDEX = 0
GLOBAL_HF_COOLDOWN_UNTIL = 0.0  # ग्लोबल एंटी-बॉट सेफगार्ड

def get_active_hf_token():
    global CURRENT_HF_INDEX, GLOBAL_HF_COOLDOWN_UNTIL
    if not HF_TOKEN_STATS:
        return None, "No HF tokens configured"

    now = time.time()
    hf_cfg = CONFIGS.get("hf_spaces", {}).get("limits", {})
    max_gpu_sec = hf_cfg.get("gpu_seconds_per_token", 300)
    cooldown_sec = hf_cfg.get("cooldown_delay_seconds", 120)

    # 1. क्या ग्लोबल एंटी-बॉट 2 मिनट का ब्रेक अभी एक्टिव है?
    if now < GLOBAL_HF_COOLDOWN_UNTIL:
        remaining = int(GLOBAL_HF_COOLDOWN_UNTIL - now)
        return None, f"Anti-Bot Cooldown Active: Waiting {remaining}s before switching next HF token"

    # 2. सुरक्षित टोकन की खोज
    for _ in range(len(HF_TOKEN_STATS)):
        stat = HF_TOKEN_STATS[CURRENT_HF_INDEX]

        # 24 घंटे में कोटा रीसेट
        if now - stat["last_reset"] > 86400:
            stat["used_seconds"] = 0.0
            stat["exhausted"] = False
            stat["last_reset"] = now

        # अगर टोकन में 300 सेकंड से कम GPU यूज़ हुआ है
        if stat["used_seconds"] < max_gpu_sec:
            return stat, "OK"
        else:
            # अगर टोकन खत्म हो चुका है, तो अगले टोकन पर जाने से पहले 2 मिनट का कूलडाउन लगेगा
            if not stat["exhausted"]:
                stat["exhausted"] = True
                stat["last_exhausted_time"] = now
                GLOBAL_HF_COOLDOWN_UNTIL = now + cooldown_sec
                print(f"[ANTI-BOT] Token {stat['id']} finished 5 min GPU. Enforcing {cooldown_sec}s Cooldown Delay.")
                return None, f"Token {stat['id']} quota finished. 2-minute Anti-Bot Delay enforced."

            CURRENT_HF_INDEX = (CURRENT_HF_INDEX + 1) % len(HF_TOKEN_STATS)

    return None, "All 10 HF tokens exhausted for today"

def record_hf_gpu_time(stat_entry, duration: float):
    global CURRENT_HF_INDEX, GLOBAL_HF_COOLDOWN_UNTIL
    stat_entry["used_seconds"] += duration
    hf_cfg = CONFIGS.get("hf_spaces", {}).get("limits", {})
    max_gpu_sec = hf_cfg.get("gpu_seconds_per_token", 300)
    cooldown_sec = hf_cfg.get("cooldown_delay_seconds", 120)

    # अगर इस कॉल के बाद टोकन 300 सेकंड पार कर गया
    if stat_entry["used_seconds"] >= max_gpu_sec:
        stat_entry["exhausted"] = True
        now = time.time()
        stat_entry["last_exhausted_time"] = now
        GLOBAL_HF_COOLDOWN_UNTIL = now + cooldown_sec
        CURRENT_HF_INDEX = (CURRENT_HF_INDEX + 1) % len(HF_TOKEN_STATS)
        print(f"[ANTI-BOT] Token {stat_entry['id']} reached 300s limit! Triggering 2-min delay.")

# प्रोवाइडर रिक्वेस्ट ट्रैकर
STATS = {
    prov: {"total_requests": 0, "minute_requests": 0, "last_reset_minute": time.time()}
    for prov in ["groq", "gemini", "sambanova", "openrouter"]
}

def check_and_increment(prov: str) -> bool:
    now = time.time()
    data = STATS[prov]
    if now - data["last_reset_minute"] > 60:
        data["minute_requests"] = 0
        data["last_reset_minute"] = now

    limit_rpm = CONFIGS.get(prov, {}).get("limits", {}).get("rpm", 9999)
    if data["minute_requests"] >= limit_rpm:
        return False
    data["total_requests"] += 1
    data["minute_requests"] += 1
    return True

# HF Spaces ZeroGPU कॉल
def execute_hf_spaces(prompt: str):
    token_entry, reason = get_active_hf_token()
    if not token_entry:
        print(f"[HF SKIP] {reason}")
        return None

    cfg = CONFIGS.get("hf_spaces", {})
    endpoint = cfg.get("api_endpoint", "https://jyoti-chet.hf.space/api/predict")

    start_time = time.time()
    try:
        res = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token_entry['token']}",
                "Content-Type": "application/json"
            },
            json={"data": [prompt]},
            timeout=cfg.get("limits", {}).get("max_execution_timeout", 60)
        )
        duration = time.time() - start_time
        record_hf_gpu_time(token_entry, duration)

        if res.status_code == 200:
            return {
                "provider": "hf_zerogpu_space",
                "active_token_id": token_entry["id"],
                "gpu_time_consumed_sec": round(duration, 2),
                "token_used_total_sec": round(token_entry["used_seconds"], 2),
                "data": res.json()
            }
    except Exception as e:
        duration = time.time() - start_time
        record_hf_gpu_time(token_entry, duration)
        print(f"[ERR] HF Space Call Failed: {e}")
    return None

def execute_groq(prompt: str):
    cfg = CONFIGS.get("groq", {})
    key = KEYS["groq"]
    if not key or not cfg or not check_and_increment("groq"): return None
    res = requests.post(
        cfg.get("endpoint"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": cfg.get("default_model"), "messages": [{"role": "user", "content": prompt}]},
        timeout=25
    )
    return {"provider": "groq", "data": res.json()} if res.status_code == 200 else None

def execute_gemini(prompt: str):
    cfg = CONFIGS.get("gemini", {})
    key = KEYS["gemini"]
    if not key or not cfg or not check_and_increment("gemini"): return None
    url = cfg.get("endpoint_template").format(model=cfg.get("default_model")) + f"?key={key}"
    res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=25)
    return {"provider": "gemini", "data": res.json()} if res.status_code == 200 else None

def execute_sambanova(prompt: str):
    cfg = CONFIGS.get("sambanova", {})
    key = KEYS["sambanova"]
    if not key or not cfg or not check_and_increment("sambanova"): return None
    res = requests.post(
        cfg.get("endpoint"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": cfg.get("default_model"), "messages": [{"role": "user", "content": prompt}]},
        timeout=25
    )
    return {"provider": "sambanova", "data": res.json()} if res.status_code == 200 else None

def execute_openrouter(prompt: str):
    cfg = CONFIGS.get("openrouter", {})
    key = KEYS["openrouter"]
    if not key or not cfg or not check_and_increment("openrouter"): return None
    res = requests.post(
        cfg.get("endpoint"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": cfg.get("default_model"), "messages": [{"role": "user", "content": prompt}]},
        timeout=25
    )
    return {"provider": "openrouter", "data": res.json()} if res.status_code == 200 else None

class QueryPayload(BaseModel):
    prompt: str
    provider: str = "auto"

@app.get("/")
def dashboard():
    now = time.time()
    cooldown_rem = max(0, int(GLOBAL_HF_COOLDOWN_UNTIL - now))
    return {
        "status": "online",
        "engine": "Jyoti Sovereign Multi-Cloud Core",
        "standard_providers": STATS,
        "hf_zerogpu_engine": {
            "loaded_tokens": len(HF_TOKENS),
            "anti_bot_cooldown_active": cooldown_rem > 0,
            "cooldown_remaining_sec": cooldown_rem,
            "tokens_status": [
                {
                    "token_id": t["id"],
                    "used_gpu_sec": round(t["used_seconds"], 2),
                    "quota_max_sec": 300,
                    "exhausted": t["exhausted"]
                }
                for t in HF_TOKEN_STATS
            ]
        }
    }

@app.post("/v1/chat")
def handle_chat(payload: QueryPayload):
    p = payload.provider.lower()

    if p == "groq":
        out = execute_groq(payload.prompt)
        if out: return out
    elif p == "gemini":
        out = execute_gemini(payload.prompt)
        if out: return out
    elif p == "sambanova":
        out = execute_sambanova(payload.prompt)
        if out: return out
    elif p == "openrouter":
        out = execute_openrouter(payload.prompt)
        if out: return out
    elif p in ["hf", "hf_spaces", "space"]:
        out = execute_hf_spaces(payload.prompt)
        if out: return out

    # ऑटो-फॉलबैक चेन: अगर HF 2 मिनट के ब्रेक पर है, तो बाकी तुरंत संभाल लेंगे
    chain = [execute_groq, execute_sambanova, execute_gemini, execute_openrouter, execute_hf_spaces]
    for fn in chain:
        try:
            res = fn(payload.prompt)
            if res: return res
        except Exception:
            continue

    raise HTTPException(status_code=429, detail="All AI providers are currently exhausted or cooling down.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
