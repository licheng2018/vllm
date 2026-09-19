# vLLM Scheduler and KV-Cache Study

## Day 1 — vLLM Setup, Request Inspection, and Baseline Benchmarking

### Overview

This project studies vLLM at the source-code and runtime level, with a focus on the request lifecycle, scheduling, KV-cache management, continuous batching, and chunked prefill.

The goal of Day 1 was to establish a reproducible inference environment, validate the basic vLLM execution path, inspect request-level outputs, and collect an initial latency baseline before moving into scheduler and KV-cache internals.

---

## Environment

* **GPU:** NVIDIA Tesla T4
* **GPU Memory:** 14.56 GiB
* **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
* **Precision:** FP16
* **Maximum model length:** 4096 tokens
* **GPU memory utilization:** 0.70
* **vLLM:** 0.29.0
* **PyTorch:** 2.13.0
* **CUDA:** 13.0
* **Python:** 3.13

Because vLLM initialization inside the Colab/Jupyter kernel caused an `ipykernel stdout fileno()` incompatibility, offline inference was executed through a standalone Python process.

The resulting execution structure was:

```text
Colab Notebook
      ↓
offline Python process
      ↓
vLLM LLM Engine
      ↓
EngineCore
      ↓
Scheduler
      ↓
KV Cache / Model Runner
      ↓
GPU
```

---

## Offline Inference

Two requests were submitted together through:

```python
outputs = llm.generate(
    prompts,
    sampling_params
)
```

The experiment used:

```python
SamplingParams(
    temperature=0.0,
    max_tokens=64
)
```

This provides deterministic generation behavior and limits each request to at most 64 generated tokens.

---

## RequestOutput Inspection

vLLM returns a list of `RequestOutput` objects rather than plain generated strings.

Each request contains request-level information such as:

```text
RequestOutput
├── request_id
├── prompt
├── prompt_token_ids
├── finished
└── outputs
     └── CompletionOutput
          ├── text
          ├── token_ids
          └── finish_reason
```

### Request 0

**Prompt**

```text
Explain in one sentence what a KV cache is in LLM inference.
```

**Prompt tokens:** 15
**Generated tokens:** 42
**Finished:** True
**Finish reason:** `stop`

The model completed generation naturally before reaching the configured 64-token limit.

### Request 1

**Prompt**

```text
Why can continuous batching improve GPU utilization?
```

**Prompt tokens:** 8
**Generated tokens:** 64
**Finished:** True
**Finish reason:** `length`

This request reached the configured `max_tokens=64` limit, which is why its generated answer was truncated.

This demonstrates the difference between:

```text
finish_reason = stop
```

and:

```text
finish_reason = length
```

The former indicates natural generation termination, while the latter indicates termination because the configured generation limit was reached.

---

## Offline Latency Benchmark

A simple wall-clock benchmark was performed using `time.perf_counter()`.

Each workload was executed three times after warm-up.

### Results

| Workload     | Mean Latency | Mean Output Tokens |
| ------------ | -----------: | -----------------: |
| Short Prompt |     0.9389 s |                 64 |
| Long Prompt  |     1.1869 s |                 64 |

### Raw Latencies

**Short prompt**

```text
0.9361 s
0.9422 s
0.9385 s
```

**Long prompt**

```text
1.4734 s
1.0434 s
1.0440 s
```

The short-prompt measurements were highly stable, while the long-prompt workload showed a slower first measured iteration.

The mean latency increased from:

```text
0.9389 s → 1.1869 s
```

which is approximately a **26.4% increase**.

Because both workloads generated exactly 64 output tokens, the additional latency is primarily associated with processing the larger input rather than producing more output tokens.

At this stage, this measurement represents **offline end-to-end latency**, not TTFT.

---

## Initial Observations

Several useful runtime behaviors were observed during vLLM initialization:

* vLLM automatically enabled **chunked prefill**.
* The Tesla T4 has compute capability 7.5 and therefore cannot use FlashAttention-2 in this configuration.
* vLLM selected the **Triton attention backend** instead.
* Model weights consumed approximately 3 GiB of GPU memory.
* A substantial portion of the remaining GPU memory was allocated for the KV cache.
* vLLM performed CUDA graph capture and model warm-up before serving requests.

These observations will become important when studying scheduler behavior and KV-cache allocation in later stages of the project.

---

## What Day 1 Established

Day 1 established the basic runtime path:

```text
Prompt
  ↓
LLM.generate()
  ↓
vLLM Engine
  ↓
Request scheduling
  ↓
Model execution
  ↓
GPU
  ↓
RequestOutput
```

It also established the first reproducible performance baseline against which later scheduler modifications can be compared.

---

## Day 1 Summary

**Built a reproducible vLLM inference environment on an NVIDIA T4, validated offline inference, inspected `RequestOutput` metadata, and established an initial short-vs-long prompt latency baseline for later scheduler and KV-cache analysis.**

---

## Next Step — Day 2

Day 2 will move below the public `LLM.generate()` API and trace a request through the vLLM source code.

The main path to investigate will be:

```text
LLM.generate()
      ↓
request creation
      ↓
EngineCore
      ↓
Scheduler.add_request()
      ↓
waiting / running queues
      ↓
Scheduler.schedule()
      ↓
ModelRunner
```

The objective is to understand exactly how a user request becomes a scheduled GPU workload before modifying any scheduling behavior.
