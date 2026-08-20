import time

from dotenv import load_dotenv
from vllm import LLM, SamplingParams

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def main():
    load_dotenv()

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

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s\n")

    prompts = [
        "Explain what PagedAttention does in one paragraph."
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
    )

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - start

    for output in outputs:
        text = output.outputs[0].text
        n_tokens = len(output.outputs[0].token_ids)

        print("PROMPT:", output.prompt)
        print("OUTPUT:", text)
        print(
            f"\n[sanity check only, NOT a benchmark] "
            f"{n_tokens} tokens in {gen_time:.2f}s "
            f"-> {n_tokens / gen_time:.1f} tok/s, "
            f"single unbatched request"
        )


if __name__ == "__main__":
    main()
