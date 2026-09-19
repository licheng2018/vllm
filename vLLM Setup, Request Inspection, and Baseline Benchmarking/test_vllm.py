from vllm import LLM, SamplingParams
import time
import statistics

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.70

output_path = "/content/request_output_inspection.txt"

prompts = [
    "Explain in one sentence what a KV cache is in LLM inference.",
    "Why can continuous batching improve GPU utilization?",
]

# ============================================================
# TODO — YOU WRITE THIS
#
# Task A: Construct SamplingParams.
# Requirements:
#   - temperature = 0.0
#   - max_tokens = 64
#
# sampling_params = ...
# ============================================================

sampling_params = SamplingParams(temperature=0.0,
                                 max_tokens=64)

# ============================================================
# TODO — YOU WRITE THIS
#
# Task B: Construct the vLLM LLM object.
#
# Requirements:
#   model=MODEL
#   dtype="float16"
#   max_model_len=MAX_MODEL_LEN
#   gpu_memory_utilization=GPU_MEMORY_UTILIZATION
#
# llm = ...
# ============================================================

llm = LLM(model = MODEL,
          dtype = "float16",
          max_model_len = MAX_MODEL_LEN,
          gpu_memory_utilization = GPU_MEMORY_UTILIZATION)

# ============================================================
# TODO — YOU WRITE THIS
#
# Task C: Generate outputs for `prompts`.
#
# Ask yourself:
#   1. Which object owns the generate() method?
#   2. What two main arguments does it need here?
#
# outputs = ...
# ============================================================

outputs = llm.generate(
    prompts,
    sampling_params
)


# ============================================================
# Inspect RequestOutput
# Small offline timing experiment:
# ============================================================

short_prompt = "Briefly explain GPU memory bandwidth."
long_prompt = ("Explain the relationship between GPU compute throughput, memory "
               "bandwidth, arithmetic intensity, and kernel performance. " * 80)

def run_once(prompt):
    # ========================================================
    # TODO — YOU WRITE THIS
    #
    # Measure wall-clock latency around ONE llm.generate call.
    # Return:
    #   latency_seconds, output_token_count
    #
    # Notes:
    # - This is NOT yet a rigorous serving benchmark.
    # - llm.generate is offline/batched inference, not HTTP TTFT.
    # - We only want a Day-1 baseline and familiarity with outputs.
    # ========================================================
    # raise NotImplementedError
    start_time = time.perf_counter()
    output = llm.generate(
        [prompt],
        sampling_params
    )
    end_time = time.perf_counter()
    latency_seconds = end_time - start_time
    output_token_count = len(output[0].outputs[0].token_ids)
    return latency_seconds, output_token_count

# Warm-up: complete this after run_once() works.
# _ = run_once(short_prompt)

# ============================================================
# TODO — YOU WRITE THIS
#
# Run each prompt 3 times and report:
#   mean latency
#   mean output-token count
#
# Compare short_prompt vs long_prompt.
# ============================================================
short_results = [run_once(short_prompt) for _ in range(3)]
long_results = [run_once(long_prompt) for _ in range(3)]

short_latencies = [x[0] for x in short_results]
short_tokens = [x[1] for x in short_results]

long_latencies = [x[0] for x in long_results]
long_tokens = [x[1] for x in long_results]

with open(output_path, "w", encoding="utf-8") as f:

    # ========================================================
    # RequestOutput inspection
    # ========================================================
    f.write("=== RequestOutput Inspection ===\n\n")

    for output in outputs:
        candidate = output.outputs[0]

        f.write(f"Request ID: {output.request_id}\n")
        f.write(f"Prompt: {output.prompt}\n")
        f.write(f"Prompt token count: {len(output.prompt_token_ids)}\n")
        f.write(f"Finished: {output.finished}\n")

        f.write(f"Generated text: {candidate.text}\n")
        f.write(f"Generated token IDs: {candidate.token_ids}\n")
        f.write(f"Generated token count: {len(candidate.token_ids)}\n")
        f.write(f"Finish reason: {candidate.finish_reason}\n")

        f.write("-" * 60 + "\n")

    # ========================================================
    # Latency benchmark
    # ========================================================
    f.write("\n=== Offline Latency Benchmark ===\n\n")

    f.write("Short prompt:\n")
    f.write(f"Mean latency: {statistics.mean(short_latencies):.4f} s\n")
    f.write(f"Mean output tokens: {statistics.mean(short_tokens):.2f}\n")
    f.write(f"Raw latencies: {short_latencies}\n")
    f.write("\n")

    f.write("Long prompt:\n")
    f.write(f"Mean latency: {statistics.mean(long_latencies):.4f} s\n")
    f.write(f"Mean output tokens: {statistics.mean(long_tokens):.2f}\n")
    f.write(f"Raw latencies: {long_latencies}\n")
