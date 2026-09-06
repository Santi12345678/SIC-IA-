from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SIC I.A Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_FALLBACK = ["gemini-3.6-flash", "gemini-flash-latest", "gemini-3.7-flash"]

def get_api_key() -> str:
    load_dotenv(override=True)
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY no está configurada en el archivo .env del backend."
        )
    return key

def call_gemini(payload: dict, api_key: str) -> str:
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    last_err = ""
    for model in MODELS_FALLBACK:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif resp.status_code in (429, 503):
                last_err = resp.text
                continue
            else:
                try:
                    err_json = resp.json()
                except Exception:
                    err_json = resp.text
                raise HTTPException(status_code=resp.status_code, detail=f"Google API Error: {err_json}")
        except requests.RequestException as e:
            last_err = str(e)
            continue
            
    raise HTTPException(status_code=503, detail=f"Servidores de Gemini con alta demanda. Reintentá en unos segundos. Detalle: {last_err}")

class ChatRequest(BaseModel):
    message: str
    attachment: Optional[Any] = None
    model: Optional[str] = "⚡ SIC I.A · Pro"
    plan: Optional[str] = "Gratuito"

class CodeAnalysisRequest(BaseModel):
    language: str
    help_type: str
    problem_desc: Optional[str] = ""
    code: Optional[str] = ""
    model: Optional[str] = "⚡ SIC I.A · Pro"
    plan: Optional[str] = "Gratuito"

@app.get("/")
def read_root():
    return {"status": "online", "message": "SIC I.A Backend API is running with Gemini Flash"}

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    api_key = get_api_key()
    
    sys_instruction = (
        "Sos SIC I.A, un asistente de inteligencia artificial especializado en videojuegos y programación. "
        "Respondé siempre en español argentino. Podés hablar de cualquier tema relacionado con videojuegos "
        "(gameplay, lore, configuración, rendimiento, recomendaciones, game design, motores gráficos, esports) y programación "
        "(cualquier lenguaje, debugging, algoritmos, frameworks, desarrollo web, desarrollo de juegos, bases de datos, etc). "
        "Si te preguntan algo que no tiene nada que ver con videojuegos o programación, respondé amablemente pero redirigí la conversación hacia esos temas. "
        "Usá markdown para formatear tus respuestas con títulos, listas, bloques de código cuando corresponda. Sé detallado, útil y conversacional."
    )
    
    user_parts = [{"text": req.message}]
    
    if req.attachment and isinstance(req.attachment, dict) and req.attachment.get("dataUrl"):
        data_url = req.attachment["dataUrl"]
        if ";base64," in data_url:
            mime_type, b64_data = data_url.split(";base64,")
            mime_type = mime_type.replace("data:", "")
            user_parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": b64_data
                }
            })

    payload = {
        "system_instruction": {
            "parts": [{"text": sys_instruction}]
        },
        "contents": [
            {
                "role": "user",
                "parts": user_parts
            }
        ],
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 4096
        }
    }
    
    ai_text = call_gemini(payload, api_key)
    return {"reply": ai_text}

@app.post("/api/code-analysis")
def code_analysis_endpoint(req: CodeAnalysisRequest):
    api_key = get_api_key()
    
    prompt = f"""Sos un experto en programación en {req.language}. El usuario necesita ayuda con: "{req.help_type}".

Descripción del problema: {req.problem_desc or 'No especificada'}

Código a analizar:
```{req.language.lower()}
{req.code or '// No se proporcionó código'}
```

Respondé en español argentino con un informe técnico completo en markdown que incluya:
### 1. Qué está mal
### 2. Por qué está mal
### 3. Cómo solucionarlo (paso a paso)
### 4. Código Corregido (bloque de código completo y funcional)
### 5. Buenas prácticas y recomendaciones"""

    sys_instruction = "Sos SIC I.A, un asistente de programación experto. Respondé siempre en español argentino con informes técnicos detallados en markdown."

    payload = {
        "system_instruction": {
            "parts": [{"text": sys_instruction}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 4096
        }
    }
    
    ai_text = call_gemini(payload, api_key)
    return {"reply": ai_text}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

