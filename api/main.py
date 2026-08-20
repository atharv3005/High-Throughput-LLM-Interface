import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams


load_dotenv()

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

llm: LLM | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int = Field(default=128, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm

    print(f"Loading {MODEL_NAME} ...")
    start = time.time()

    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=0.78,
        max_model_len=1024,
        max_num_seqs=4,
        max_num_batched_tokens=512,
    )

    print(f"Model loaded in {time.time() - start:.1f}s")
    yield

    llm = None


app = FastAPI(
    title="High-Throughput LLM Inference",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "engine": "vLLM",
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    start = time.time()

    outputs = llm.generate(
        [request.prompt],
        sampling_params,
    )

    elapsed = time.time() - start

    output = outputs[0].outputs[0]

    return {
        "model": MODEL_NAME,
        "prompt": request.prompt,
        "text": output.text,
        "completion_tokens": len(output.token_ids),
        "generation_time_seconds": round(elapsed, 4),
        "tokens_per_second": round(
            len(output.token_ids) / elapsed, 2
        ) if elapsed > 0 else 0.0,
    }
