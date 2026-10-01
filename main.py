import os
import json
import time
import requests
import uvicorn
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Jyoti AI Sovereign Cluster Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIGS = {
    "groq": "groq_models.json",
    "gemini": "gemini_models.json",
    "sambanova": "sambanova_models.json",
    "openrouter": "openrouter_models.json",
    "hf_spaces": "hf_spaces.json"
}

def read_json(name: str) -> dict:
    path = CONFIGS.get(name)
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def write_json(name: str, data: dict):
    path = CONFIGS.get(name)
    if path:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

def check_and_reset_daily(cfg: dict, provider: str) -> dict:
    now = time.time()
    state = cfg.setdefault("runtime_state", {})
    if now - state.get("last_daily_reset", 0.0) > 86400:
        state["total_platform_requests_today"] = 0
        state["total_requests_today"] = 0
        state["requests_used_today"] = 0
        state["search_queries_used_today"] = 0
        state["map_queries_used_today"] = 0
        state["is_revoked"] = False
        state["revoked_until"] = 0.0
        state["all_exhausted"] = False
        state["last_daily_reset"] = now

        if provider == "groq":
            for g in cfg.get("guards", {}).values():
                g["used_today"] = 0
                g["exhausted"] = False
            for m in cfg.get("core_models", {}).values():
                m["used_today"] = 0
                m["exhausted"] = False
            for a in cfg.get("audio_models", {}).values():
                a["used_today"] = 0
                a["exhausted"] = False

        elif provider == "gemini":
            for grp in ["heavy_gemma_pool", "flash_lite_pool", "flash_pool_strict_20"]:
                for m in cfg.get(grp, {}).values():
                    m["used_today"] = 0
                    m["exhausted"] = False

        elif provider == "sambanova":
            for m in cfg.get("models", {}).values():
                m["used_today"] = 0
                m["exhausted"] = False

        elif provider == "openrouter":
            if "primary_ocr_model" in cfg:
                cfg["primary_ocr_model"]["exhausted"] = False

        elif provider == "hf_spaces":
            state["current_token_index"] = 0
            state["global_cooldown_until"] = 0.0
            for t in cfg.get("tokens", []):
                t["used_gpu_sec"] = 0.0
                t["exhausted"] = False
                t["cooldown_until"] = 0.0

        write_json(provider, cfg)
    return cfg

# ================= 1. GROQ =================
def execute_groq(prompt: str, image_url: Optional[str] = None, pdf_base64: Optional[str] = None):
    cfg = read_json("groq")
    if not cfg: return None
    cfg = check_and_reset_daily(cfg, "groq")
    state = cfg["runtime_state"]
    now = time.time()

    if state.get("is_revoked") and now < state.get("revoked_until", 0.0):
        return None

    api_key = os.getenv(cfg.get("env_key", "GROQ_API_OK"))
    if not api_key: return None

    is_multimodal = bool(image_url or pdf_base64)

    if not is_multimodal:
        active_guard = None
        for g_k in ["llama_guard_86m", "llama_guard_22m", "openai_safeguard"]:
            g = cfg.get("guards", {}).get(g_k)
            if g and not g["exhausted"] and g["used_today"] < g["rpd_limit"]:
                active_guard = g
                break

        if active_guard:
            try:
                g_res = requests.post(
                    cfg["chat_endpoint"],
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": active_guard["id"], "messages": [{"role": "user", "content": prompt}]},
                    timeout=10
                )
                if g_res.status_code == 200:
                    active_guard["used_today"] += 1
                    state["total_platform_requests_today"] += 1
                    if active_guard["used_today"] >= active_guard["rpd_limit"]:
                        active_guard["exhausted"] = True

                    txt = g_res.json()["choices"][0]["message"]["content"].lower()
                    if "unsafe" in txt or "block" in txt:
                        write_json("groq", cfg)
                        return {"provider": "groq_guard", "status": "rejected", "message": "Blocked by Safety Guard."}
            except Exception:
                pass

    core = cfg["core_models"]
    target_model = None

    if is_multimodal:
        qwen = core.get("multimodal_vision_pdf")
        if qwen and not qwen["exhausted"] and qwen["used_today"] < qwen["rpd_limit"]:
            target_model = qwen
        else:
            return None
    else:
        for k in ["gpt_oss_120b", "gpt_oss_20b", "multimodal_vision_pdf"]:
            m = core.get(k)
            if m and not m["exhausted"] and m["used_today"] < m["rpd_limit"]:
                target_model = m
                break

    if not target_model:
        state["is_revoked"] = True
        state["revoked_until"] = now + 86400
        write_json("groq", cfg)
        return None

    msgs = []
    if is_multimodal and image_url:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]})
    elif is_multimodal and pdf_base64:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}}]})
    else:
        msgs.append({"role": "user", "content": prompt})

    try:
        res = requests.post(
            cfg["chat_endpoint"],
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": target_model["id"], "messages": msgs},
            timeout=30
        )
        if res.status_code == 200:
            target_model["used_today"] += 1
            state["total_platform_requests_today"] += 1
            if target_model["used_today"] >= target_model["rpd_limit"]:
                target_model["exhausted"] = True
            write_json("groq", cfg)
            return {"provider": "groq", "model": target_model["id"], "used_today": target_model["used_today"], "data": res.json()}
        elif res.status_code in [429, 503]:
            target_model["exhausted"] = True
            write_json("groq", cfg)
    except Exception:
        write_json("groq", cfg)

    return None

# ================= 2. GEMINI (सटीक कोटा: Gemma 14.4K -> Lite 500 -> Flash 20) =================
def execute_gemini(prompt: str, image_base64: Optional[str] = None, pdf_base64: Optional[str] = None, need_search: bool = False, need_map: bool = False):
    cfg = read_json("gemini")
    if not cfg: return None
    cfg = check_and_reset_daily(cfg, "gemini")
    state = cfg["runtime_state"]
    now = time.time()

    if state.get("is_revoked") and now < state.get("revoked_until", 0.0):
        return None

    api_key = os.getenv(cfg.get("env_key", "GEMINI-API-OK"))
    if not api_key: return None

    target = None

    # प्राथमिकता क्रम:
    # 1. Gemma 4 (14.4K RPD)
    for k in ["gemma_4_31b", "gemma_4_26b"]:
        m = cfg.get("heavy_gemma_pool", {}).get(k)
        if m and not m["exhausted"] and m["used_today"] < m["rpd_limit"]:
            target = m
            break

    # 2. Flash Lite (500 RPD)
    if not target:
        for k in ["gemini_3_5_flash_lite", "gemini_3_1_flash_lite"]:
            m = cfg.get("flash_lite_pool", {}).get(k)
            if m and not m["exhausted"] and m["used_today"] < m["rpd_limit"]:
                target = m
                break

    # 3. Flash Strict (20 RPD)
    if not target:
        for m in cfg.get("flash_pool_strict_20", {}).values():
            if not m["exhausted"] and m["used_today"] < m["rpd_limit"]:
                target = m
                break

    if not target:
        state["is_revoked"] = True
        state["revoked_until"] = now + 86400
        write_json("gemini", cfg)
        return None

    parts = []
    if image_base64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_base64}})
    if pdf_base64:
        parts.append({"inline_data": {"mime_type": "application/pdf", "data": pdf_base64}})
    parts.append({"text": prompt})

    payload = {"contents": [{"parts": parts}]}
    tools = []
    if need_search and state.get("search_queries_used_today", 0) < cfg.get("tool_limits", {}).get("search_grounding_rpd", 1500):
        tools.append({"googleSearch": {}})
    if need_map and state.get("map_queries_used_today", 0) < cfg.get("tool_limits", {}).get("map_grounding_rpd", 500):
        tools.append({"googleMaps": {}})
    if tools:
        payload["tools"] = tools

    url = cfg["rest_endpoint_template"].format(model=target["id"]) + f"?key={api_key}"

    try:
        res = requests.post(url, json=payload, timeout=30)
        if res.status_code == 200:
            target["used_today"] += 1
            state["total_requests_today"] += 1
            if need_search:
                state["search_queries_used_today"] += 1
            if need_map:
                state["map_queries_used_today"] += 1

            if target["used_today"] >= target["rpd_limit"]:
                target["exhausted"] = True
            write_json("gemini", cfg)
            return {"provider": "gemini", "model": target["id"], "used_today": target["used_today"], "data": res.json()}
        elif res.status_code in [429, 503]:
            target["exhausted"] = True
            write_json("gemini", cfg)
    except Exception:
        write_json("gemini", cfg)

    return None

# ================= 3. SAMBANOVA (सटीक 120 RPD कुल कोटा) =================
def execute_sambanova(prompt: str, is_math: bool = False):
    cfg = read_json("sambanova")
    if not cfg: return None
    cfg = check_and_reset_daily(cfg, "sambanova")
    state = cfg["runtime_state"]
    now = time.time()

    if state.get("is_revoked") and now < state.get("revoked_until", 0.0):
        return None

    if state.get("total_requests_today", 0) >= cfg["limits"]["total_platform_daily_cap"]:
        state["is_revoked"] = True
        state["revoked_until"] = now + 86400
        write_json("sambanova", cfg)
        return None

    if now - state.get("last_minute_timestamp", 0.0) > 60:
        state["current_minute_requests"] = 0
        state["last_minute_timestamp"] = now

    if state.get("current_minute_requests", 0) >= cfg["limits"]["global_rpm"]:
        return None

    api_key = os.getenv(cfg.get("env_key", "SAMBANOVA-API-OK"))
    if not api_key: return None

    models = cfg["models"]
    target = None

    if is_math:
        for k in ["deepseek_r1", "deepseek_v3"]:
            m = models.get(k)
            if m and not m["exhausted"] and m["used_today"] < m["rpd_limit"]:
                target = m
                break
    else:
        for k in ["llama_70b", "llama_8b", "gemma_31b", "gpt_oss_120b"]:
            m = models.get(k)
            if m and not m["exhausted"] and m["used_today"] < m["rpd_limit"]:
                target = m
                break

    if not target:
        state["is_revoked"] = True
        state["revoked_until"] = now + 86400
        write_json("sambanova", cfg)
        return None

    try:
        res = requests.post(
            cfg["endpoint"],
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": target["id"], "messages": [{"role": "user", "content": prompt}]},
            timeout=25
        )
        if res.status_code == 200:
            target["used_today"] += 1
            state["total_requests_today"] += 1
            state["current_minute_requests"] += 1
            if target["used_today"] >= target["rpd_limit"]:
                target["exhausted"] = True
            write_json("sambanova", cfg)
            return {"provider": "sambanova", "model": target["id"], "used_today": target["used_today"], "data": res.json()}
        elif res.status_code == 429:
            target["exhausted"] = True
            write_json("sambanova", cfg)
    except Exception:
        write_json("sambanova", cfg)

    return None

# ================= 4. OPENROUTER (50 RPD कुल खाता कोटा) =================
def execute_openrouter(prompt: str, image_url: Optional[str] = None):
    cfg = read_json("openrouter")
    if not cfg: return None
    cfg = check_and_reset_daily(cfg, "openrouter")
    state = cfg["runtime_state"]
    model_cfg = cfg["primary_ocr_model"]
    now = time.time()

    if state.get("is_revoked") and now < state.get("revoked_until", 0.0):
        return None

    if state.get("requests_used_today", 0) >= cfg["account_limits"]["total_daily_rpd"]:
        state["is_revoked"] = True
        state["revoked_until"] = now + 86400
        model_cfg["exhausted"] = True
        write_json("openrouter", cfg)
        return None

    if now - state.get("last_minute_timestamp", 0.0) > 60:
        state["current_minute_requests"] = 0
        state["last_minute_timestamp"] = now

    if state.get("current_minute_requests", 0) >= cfg["account_limits"]["global_rpm"]:
        return None

    api_key = os.getenv(cfg.get("env_key", "OPENROUTER-API-OK"))
    if not api_key: return None

    msgs = [{"role": "system", "content": model_cfg.get("system_instruction", "")}]
    if image_url:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prompt or "Extract text and explain."}, {"type": "image_url", "image_url": {"url": image_url}}]})
    else:
        msgs.append({"role": "user", "content": prompt})

    try:
        res = requests.post(
            cfg["endpoint"],
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://jyotiagent.rjs-06-opc.com",
                "X-Title": "Jyoti OCR",
                "Content-Type": "application/json"
            },
            json={"model": model_cfg["id"], "messages": msgs},
            timeout=30
        )
        if res.status_code == 200:
            state["requests_used_today"] += 1
            state["current_minute_requests"] += 1
            if state["requests_used_today"] >= cfg["account_limits"]["total_daily_rpd"]:
                state["is_revoked"] = True
                state["revoked_until"] = now + 86400
                model_cfg["exhausted"] = True
            write_json("openrouter", cfg)
            return {"provider": "openrouter_ocr", "model": model_cfg["id"], "used_today": state["requests_used_today"], "data": res.json()}
        elif res.status_code == 429:
            state["is_revoked"] = True
            state["revoked_until"] = now + 86400
            model_cfg["exhausted"] = True
            write_json("openrouter", cfg)
    except Exception:
        write_json("openrouter", cfg)

    return None

# ================= 5. HF ZEROGPU (10 टोकन्स x 300s = 50 मिनट GPU, 120s Delay) =================
def execute_hf_spaces(prompt: str):
    cfg = read_json("hf_spaces")
    if not cfg: return None
    cfg = check_and_reset_daily(cfg, "hf_spaces")
    state = cfg["runtime_state"]
    now = time.time()

    if state.get("all_exhausted") or (state.get("revoked_until", 0.0) > now):
        return None

    if now < state.get("global_cooldown_until", 0.0):
        return None

    tokens = cfg.get("tokens", [])
    curr_idx = state.get("current_token_index", 0)
    target_token = None

    for _ in range(len(tokens)):
        t = tokens[curr_idx]
        if not t.get("exhausted", False) and now >= t.get("cooldown_until", 0.0):
            target_token = t
            break
        curr_idx = (curr_idx + 1) % len(tokens)

    if not target_token:
        state["all_exhausted"] = True
        state["revoked_until"] = now + 86400
        write_json("hf_spaces", cfg)
        return None

    api_key = os.getenv(target_token.get("env_key", ""))
    if not api_key: return None

    t0 = time.time()
    try:
        res = requests.post(
            cfg["api_endpoint"],
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"data": [prompt]},
            timeout=cfg["limits"].get("max_execution_timeout", 60)
        )
        duration = time.time() - t0
        target_token["used_gpu_sec"] += duration

        if target_token["used_gpu_sec"] >= cfg["limits"]["gpu_seconds_per_token"]:
            target_token["exhausted"] = True
            state["global_cooldown_until"] = time.time() + cfg["limits"]["cooldown_delay_seconds"]
            state["current_token_index"] = (curr_idx + 1) % len(tokens)
        else:
            state["current_token_index"] = curr_idx

        write_json("hf_spaces", cfg)
        return {"provider": "hf_zerogpu", "token_id": target_token["id"], "duration_sec": round(duration, 2), "data": res.json()} if res.status_code == 200 else None
    except Exception:
        target_token["cooldown_until"] = time.time() + 120
        write_json("hf_spaces", cfg)

    return None

# ================= FASTAPI ROUTING ENDPOINTS =================

class UnifiedChatRequest(BaseModel):
    prompt: str
    provider: str = "auto"
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    pdf_base64: Optional[str] = None
    need_search: bool = False
    need_map: bool = False
    is_math: bool = False

@app.get("/")
def get_dashboard():
    return {
        "status": "online",
        "groq": read_json("groq"),
        "gemini": read_json("gemini"),
        "sambanova": read_json("sambanova"),
        "openrouter": read_json("openrouter"),
        "hf_spaces": read_json("hf_spaces")
    }

@app.post("/v1/chat")
def handle_unified_chat(req: UnifiedChatRequest):
    # 1. OCR / विज़न / PDF (Qwen 3.8 27B -> Gemini Gemma/Flash OCR -> OpenRouter Gemma OCR)
    if req.image_url or req.image_base64 or req.pdf_base64:
        out = execute_groq(req.prompt, image_url=req.image_url, pdf_base64=req.pdf_base64)
        if out: return out
        gem_out = execute_gemini(req.prompt, image_base64=req.image_base64, pdf_base64=req.pdf_base64, need_search=req.need_search, need_map=req.need_map)
        if gem_out: return gem_out
        if req.image_url:
            op_out = execute_openrouter(req.prompt, image_url=req.image_url)
            if op_out: return op_out
        raise HTTPException(status_code=429, detail="All OCR/Vision models exhausted and locked for 24h.")

    # 2. डायरेक्ट प्रोवाइडर अनुरोध
    p = req.provider.lower()
    if p == "groq":
        out = execute_groq(req.prompt)
        if out: return out
    elif p == "gemini":
        out = execute_gemini(req.prompt, need_search=req.need_search, need_map=req.need_map)
        if out: return out
    elif p == "sambanova":
        out = execute_sambanova(req.prompt, is_math=req.is_math)
        if out: return out
    elif p == "openrouter":
        out = execute_openrouter(req.prompt)
        if out: return out
    elif p in ["hf", "hf_spaces"]:
        out = execute_hf_spaces(req.prompt)
        if out: return out

    # 3. ग्लोबल ऑटो-फ़ॉलबैक चेन (Groq -> Gemini -> SambaNova -> OpenRouter -> HF Spaces)
    chain = [
        lambda: execute_groq(req.prompt),
        lambda: execute_gemini(req.prompt, need_search=req.need_search, need_map=req.need_map),
        lambda: execute_sambanova(req.prompt, is_math=req.is_math),
        lambda: execute_openrouter(req.prompt),
        lambda: execute_hf_spaces(req.prompt)
    ]

    for fn in chain:
        try:
            res = fn()
            if res: return res
        except Exception:
            continue

    raise HTTPException(status_code=429, detail="CRITICAL: All 5 AI Providers exhausted their quotas and are locked for 24 hours.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
