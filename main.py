import os
import json
import time
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Jyoti Dynamic JSON-Driven Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_FILES = {
    "groq": "groq_config.json",
    "gemini": "gemini_config.json",
    "sambanova": "sambanova_config.json",
    "openrouter": "openrouter_config.json",
    "hf_spaces": "hf_spaces.json"
}

def read_json(name):
    path = CONFIG_FILES.get(name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def write_json(name, data):
    path = CONFIG_FILES.get(name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def is_provider_available(cfg_name):
    cfg = read_json(cfg_name)
    if not cfg: return False, None, "File missing"
    
    state = cfg.get("runtime_state", {})
    now = time.time()
    
    # 429 या कूलडाउन चेक
    if state.get("is_rate_limited") and now < state.get("cooldown_until", 0.0):
        return False, cfg, "Under rate-limit cooldown"
    elif state.get("is_rate_limited") and now >= state.get("cooldown_until", 0.0):
        state["is_rate_limited"] = False

    # RPM व RPD सीमा जांच
    limits = cfg.get("limits", {})
    if now - state.get("last_minute_timestamp", 0.0) > 60:
        state["current_minute_requests"] = 0
        state["last_minute_timestamp"] = now

    if state.get("current_minute_requests", 0) >= limits.get("rpm", 9999):
        return False, cfg, "RPM exceeded"
    if state.get("total_requests", 0) >= limits.get("rpd", 999999):
        return False, cfg, "RPD exceeded"

    api_key = os.getenv(cfg.get("env_key", ""))
    if not api_key:
        return False, cfg, "Env key not found"

    return True, cfg, api_key

def update_provider_success(cfg_name, cfg):
    state = cfg["runtime_state"]
    state["total_requests"] += 1
    state["current_minute_requests"] += 1
    write_json(cfg_name, cfg)

def handle_provider_error(cfg_name, cfg, status_code):
    state = cfg["runtime_state"]
    if status_code in [429, 503, 500]:
        state["is_rate_limited"] = True
        state["cooldown_until"] = time.time() + 60  # 1 मिनट का ब्रेक
    write_json(cfg_name, cfg)

# HF ZeroGPU इंजन
def call_hf_space(prompt: str):
    cfg = read_json("hf_spaces")
    if not cfg: return None
    
    now = time.time()
    state = cfg.get("runtime_state", {})
    limits = cfg.get("limits", {})
    
    # 2 मिनट का एंटी-बॉट चेक
    if now < state.get("global_cooldown_until", 0.0):
        return None

    tokens = cfg.get("tokens", [])
    curr_idx = state.get("current_token_index", 0)
    
    target_token_data = None
    for _ in range(len(tokens)):
        t = tokens[curr_idx]
        if not t.get("exhausted", False) and now >= t.get("cooldown_until", 0.0):
            target_token_data = t
            break
        curr_idx = (curr_idx + 1) % len(tokens)

    if not target_token_data:
        return None

    api_key = os.getenv(target_token_data.get("env_key", ""))
    if not api_key:
        return None

    endpoint = cfg.get("api_endpoint")
    t0 = time.time()
    try:
        res = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"data": [prompt]},
            timeout=limits.get("max_execution_timeout", 60)
        )
        duration = time.time() - t0
        target_token_data["used_gpu_sec"] += duration
        
        # 300 सेकंड (5 मिनट) GPU पूरा होने पर 2 मिनट का एंटी-बॉट डिले
        if target_token_data["used_gpu_sec"] >= limits.get("gpu_seconds_per_token", 300):
            target_token_data["exhausted"] = True
            state["global_cooldown_until"] = time.time() + limits.get("cooldown_delay_seconds", 120)
            state["current_token_index"] = (curr_idx + 1) % len(tokens)
        else:
            state["current_token_index"] = curr_idx

        write_json("hf_spaces", cfg)

        if res.status_code == 200:
            return {"provider": "hf_zerogpu", "token_id": target_token_data["id"], "duration_sec": round(duration, 2), "data": res.json()}
        elif res.status_code in [429, 503]:
            target_token_data["cooldown_until"] = time.time() + 120
            write_json("hf_spaces", cfg)
    except Exception:
        write_json("hf_spaces", cfg)
    return None

class QueryPayload(BaseModel):
    prompt: str
    provider: str = "auto"

@app.get("/")
def get_cluster_status():
    return {
        "status": "online",
        "groq": read_json("groq"),
        "gemini": read_json("gemini"),
        "sambanova": read_json("sambanova"),
        "openrouter": read_json("openrouter"),
        "hf_spaces": read_json("hf_spaces")
    }

@app.post("/v1/chat")
def chat_orchestrator(payload: QueryPayload):
    req_prov = payload.provider.lower()

    # सामान्य 4 प्रोवाइडर्स का जेनेरिक निष्पादन
    def try_provider(p_name):
        ok, cfg, key_or_err = is_provider_available(p_name)
        if not ok: return None

        headers = {"Authorization": f"Bearer {key_or_err}", "Content-Type": "application/json"}
        model = cfg.get("default_model")
        
        try:
            if p_name == "gemini":
                url = cfg.get("endpoint_template").format(model=model) + f"?key={key_or_err}"
                res = requests.post(url, json={"contents": [{"parts": [{"text": payload.prompt}]}]}, timeout=25)
            else:
                res = requests.post(
                    cfg.get("endpoint"),
                    headers=headers,
                    json={"model": model, "messages": [{"role": "user", "content": payload.prompt}]},
                    timeout=25
                )

            if res.status_code == 200:
                update_provider_success(p_name, cfg)
                return {"provider": p_name, "model": model, "data": res.json()}
            else:
                handle_provider_error(p_name, cfg, res.status_code)
        except Exception:
            handle_provider_error(p_name, cfg, 500)
        return None

    # प्रोवाइडर राउटिंग
    if req_prov in ["groq", "gemini", "sambanova", "openrouter"]:
        out = try_provider(req_prov)
        if out: return out
    elif req_prov in ["hf", "hf_spaces"]:
        out = call_hf_space(payload.prompt)
        if out: return out

    # ऑटो-फॉलबैक (Groq -> SambaNova -> Gemini -> OpenRouter -> HF Spaces)
    for p in ["groq", "sambanova", "gemini", "openrouter"]:
        out = try_provider(p)
        if out: return out

    hf_out = call_hf_space(payload.prompt)
    if hf_out: return hf_out

    raise HTTPException(status_code=429, detail="All providers busy, cooling down, or limits reached in JSON state.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
