import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL = os.getenv(
    "VLLM_MODEL",
    "meta-llama/Llama-3.2-1B-Instruct",
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int = Field(default=128, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


app = FastAPI(
    title="High-Throughput LLM Interface",
    version="0.2.0",
)


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{VLLM_BASE_URL}/v1/models")

        response.raise_for_status()

        return {
            "status": "ok",
            "model": VLLM_MODEL,
            "engine": "vLLM",
            "backend": VLLM_BASE_URL,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"vLLM backend unavailable: {exc}",
        ) from exc


@app.post("/generate")
async def generate(request: GenerateRequest):
    payload = {
        "model": VLLM_MODEL,
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{VLLM_BASE_URL}/v1/completions",
                json=payload,
            )

        response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]
        usage = data["usage"]

        return {
            "model": data["model"],
            "prompt": request.prompt,
            "text": choice["text"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        }

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"vLLM returned HTTP {exc.response.status_code}",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"vLLM backend unavailable: {exc}",
        ) from exc
