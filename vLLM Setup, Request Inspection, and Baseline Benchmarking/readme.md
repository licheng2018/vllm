# vLLM Scheduler and KV-Cache Optimization

A source-level study of the vLLM inference engine, focusing on request scheduling, continuous batching, chunked prefill, KV-cache management, prefix caching, and paged KV memory addressing.

The project starts from a working vLLM inference environment and progressively traces the execution path from the public `LLM.generate()` API down to the scheduler, KV-cache allocator, physical block pool, worker-side block table, and Triton slot-mapping kernel.

The longer-term goal is to instrument and modify the scheduler, then evaluate the impact of different scheduling policies on:

* TTFT
* TPOT
* throughput
* tail latency
* KV-cache utilization
* preemption

---

## Project Motivation

High-performance LLM serving is not only a model execution problem.

A serving engine must simultaneously manage:

```text
Incoming requests
        ↓
Request scheduling
        ↓
Token-level compute budget
        ↓
KV-cache memory allocation
        ↓
Dynamic batching
        ↓
GPU execution
```

This project studies how vLLM implements these mechanisms internally rather than treating the framework as a black box.

The main questions are:

1. How does a request move from `LLM.generate()` into the scheduler?
2. How does vLLM dynamically combine prefill and decode requests?
3. How are token budgets allocated across requests?
4. How does chunked prefill prevent long prompts from monopolizing a scheduler step?
5. How are logical KV-cache blocks mapped to non-contiguous physical GPU blocks?
6. How does prefix caching reuse existing KV blocks?
7. What happens when KV-cache capacity becomes insufficient?
8. How can scheduler policies be instrumented and optimized for mixed inference workloads?

---

# Environment

Initial development was performed on Google Colab.

| Component              | Configuration                |
| ---------------------- | ---------------------------- |
| GPU                    | NVIDIA Tesla T4              |
| GPU Memory             | ~14.56 GiB                   |
| Model                  | `Qwen/Qwen2.5-1.5B-Instruct` |
| vLLM                   | 0.29.0                       |
| Python                 | 3.13.15                      |
| PyTorch                | 2.13.0+cu130                 |
| Torch CUDA Build       | 13.0                         |
| Max Model Length       | 4096                         |
| GPU Memory Utilization | 0.70                         |

On the T4, vLLM selected the Triton attention backend because FlashAttention-2 is not supported on compute capability 7.5.

During initialization, vLLM reported approximately:

```text
Available KV cache memory: ~6.13 GiB
GPU KV cache capacity:     229,472 tokens
Estimated concurrency:     ~56x at 4096 tokens/request
```

---

# Repository Structure

```text
vllm-systems-study/
│
├── README.md
│
├── day1/
│   ├── day1_setup_and_baseline.py
│   └── day1_notes.md
│
├── day2/
│   ├── day2_request_lifecycle.py
│   └── day2_request_lifecycle.md
│
├── day3/
│   ├── day3_scheduler_internals.py
│   └── day3_scheduler_internals.md
│
├── day4/
│   ├── day4_kv_cache.py
│   └── day4_kv_cache.md
│
└── ...
```

The exact filenames can be adjusted, but the project is organized around incremental source-code exploration.

---

# Day 1 — Environment, Request Inspection, and Baseline

## Goal

Build a reproducible vLLM environment and establish a working inference baseline before modifying the framework.

The first step was to run offline inference with:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    max_model_len=4096,
    gpu_memory_utilization=0.70,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
)
```

The returned `RequestOutput` objects were inspected to understand request-level and completion-level metadata.

Important fields include:

```text
RequestOutput
├── request_id
├── prompt
├── prompt_token_ids
├── outputs
│   └── CompletionOutput
│       ├── text
│       ├── token_ids
│       └── finish_reason
└── finished
```

## Initial Baseline

A simple short-vs-long prompt experiment was performed while keeping the generated output length fixed at 64 tokens.

| Workload     | Mean End-to-End Latency |
| ------------ | ----------------------: |
| Short prompt |                0.9389 s |
| Long prompt  |                1.1869 s |

Raw measurements:

```text
Short:
0.9361
0.9422
0.9385

Long:
1.4734
1.0434
1.0440
```

The longer prompt was approximately 26% slower despite producing the same number of output tokens, indicating additional input/prefill work.

This experiment measures offline end-to-end latency rather than true TTFT. Streaming inference will be used later to separate TTFT and TPOT.

## Colab Compatibility Note

Calling `LLM(...)` directly inside the Colab notebook kernel caused:

```text
io.UnsupportedOperation: fileno
```

because vLLM internally accesses `sys.stdout.fileno()`, which is not supported by the notebook `OutStream`.

The workaround was to execute the offline engine from a standalone Python process:

```bash
python /content/test_vllm.py
```

This avoids the notebook stdout incompatibility while preserving Colab as the development environment.

---

# Day 2 — Request Lifecycle

## Goal

Trace a request from the public vLLM API to the scheduler.

The source-level path identified was:

```text
User Prompt
    ↓
LLM.generate()
    ↓
_run_completion()
    ↓
_add_completion_requests()
    ↓
_render_and_add_requests()
    ↓
_add_request()
    ↓
LLMEngine.add_request()
    ↓
InputProcessor.process_inputs()
    ↓
EngineCoreRequest
    ↓
EngineCore.add_request()
    ↓
Scheduler.add_request()
    ↓
WAITING
    ↓
Scheduler.schedule()
```

## Public API

`LLM.generate()` is primarily a frontend wrapper.

It forwards requests into:

```python
self._run_completion(...)
```

`_run_completion()` then separates admission from execution:

```text
_add_completion_requests()
        ↓
_run_engine()
```

## Input Processing

`InputProcessor.process_inputs()` converts frontend inputs into an `EngineCoreRequest`.

Important scheduler-relevant fields include:

```text
request_id
prompt_token_ids
sampling_params
arrival_time
priority
```

This creates the boundary between frontend request representation and engine-core execution.

## Engine Admission

`LLMEngine.add_request()` registers the request with two components:

```text
OutputProcessor
    → output/request tracking

EngineCore
    → execution and scheduling
```

`EngineCore.add_request()` then calls:

```python
self.scheduler.add_request(request)
```

A new request enters the waiting queue.

It does **not** execute immediately.

---

# Day 3 — Scheduler Internals

## Goal

Understand how vLLM decides which requests execute in each scheduler iteration and how much work each request receives.

The scheduler maintains:

```python
self.waiting = create_request_queue(self.policy)
self.running: list[Request] = []
```

Two major global constraints are:

```python
self.max_num_running_reqs
self.max_num_scheduled_tokens
```

Conceptually:

```text
max_num_running_reqs
→ maximum number of active requests

max_num_scheduled_tokens
→ maximum total tokens scheduled in one iteration
```

---

## Request Token State

Each request separately tracks logical sequence state and execution progress.

Important fields include:

```text
num_prompt_tokens
num_tokens
num_tokens_with_spec
num_computed_tokens
output_token_ids
spec_token_ids
```

The key relationships are:

```text
num_tokens
=
prompt tokens
+
generated output tokens
```

and:

```text
num_tokens_with_spec
=
num_tokens
+
speculative tokens
```

The scheduler calculates remaining work approximately as:

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

Ignoring asynchronous placeholders and speculative decoding:

```text
remaining work
≈
num_tokens
-
num_computed_tokens
```

This single abstraction allows vLLM to represent both prefill and decode.

### Prefill

```text
prompt tokens = 100
num_computed_tokens = 0

remaining work = 100
```

### Decode

After the prompt is computed and one new output token exists:

```text
num_tokens = 101
num_computed_tokens = 100

remaining work = 1
```

The scheduler therefore does not require completely separate prefill and decode scheduling models.

---

# Token Budget

Each scheduler iteration starts with a global token budget.

Conceptually:

```text
Scheduler iteration

token_budget = 2048

Request A → 1024
Request B → 1
Request C → 800

Total scheduled = 1825
Remaining = 223
```

The scheduler processes existing RUNNING requests first, then uses remaining capacity to admit WAITING requests.

This behavior forms the basis of continuous batching.

---

# Continuous Batching

Continuous batching emerges naturally from repeated scheduler iterations.

Example:

```text
Step 1:
A B C

Step 2:
A C D

Step 3:
A D E
```

Requests can leave as soon as they finish, while new requests can enter when compute and KV-cache capacity become available.

The key WAITING → RUNNING path is:

```text
WAITING request
    ↓
token budget available
    ↓
calculate num_new_tokens
    ↓
allocate KV-cache slots
    ↓
allocation succeeds
    ↓
self.running.append(request)
    ↓
request.status = RUNNING
```

Continuous batching is therefore a scheduler behavior rather than a standalone `continuous_batching()` function.

---

# Chunked Prefill

Long prompts do not have to complete prefill in a single scheduler iteration.

The relevant logic includes:

```python
threshold = self.scheduler_config.long_prefill_token_threshold

if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold
```

and:

```python
if (
    not self.scheduler_config.enable_chunked_prefill
    and num_new_tokens > request_token_budget
):
    break
```

With chunked prefill enabled:

```text
Long prompt
6000 tokens remaining

global token budget = 2048
long prefill threshold = 1024

        ↓

schedule 1024 tokens
```

The remaining capacity can then be used by decode or other prefill requests.

This creates an important latency/throughput tradeoff:

```text
larger prefill chunks
→ better prefill efficiency
→ potentially worse decode latency

smaller prefill chunks
→ better interactivity
→ potentially more scheduling/execution fragmentation
```

This tradeoff will later become one of the optimization targets of the project.

---

# Optimistic Scheduler Progress

After scheduling, vLLM advances request progress before GPU output is processed:

```python
request.num_computed_tokens += num_scheduled_token
request.num_in_flight_tokens += num_scheduled_token
```

This happens in:

```text
_update_after_schedule()
```

The reason is explicitly to allow future scheduling iterations to proceed without waiting for all output processing to finish.

The state machine is:

```text
Scheduler.schedule()
    ↓
num_scheduled_tokens
    ↓
_update_after_schedule()
    ↓
num_computed_tokens += scheduled tokens
num_in_flight_tokens += scheduled tokens
    ↓
ModelRunner executes
    ↓
update_from_output()
    ↓
process sampled tokens
    ↓
rollback rejected speculative tokens if necessary
```

This separates:

```text
scheduled progress
```

from:

```text
completed output processing
```

---

# KV Pressure and Preemption

Token capacity alone does not determine whether a request can execute.

The scheduler must also allocate KV-cache memory.

```text
compute budget available
+
KV capacity available
=
request can execute
```

If:

```python
kv_cache_manager.allocate_slots(...)
```

returns `None`, the scheduler may preempt another running request.

The general path is:

```text
KV allocation fails
    ↓
select preemption victim
    ↓
rollback victim scheduling decision
    ↓
restore token/input budget
    ↓
preempt victim
    ↓
release/reset KV state
    ↓
retry allocation
```

Therefore vLLM scheduling jointly manages:

```text
compute scheduling
+
memory scheduling
```

---

# Day 4 — KV Cache, Block Allocation, and PagedAttention

## Goal

Follow the token-level scheduler decision into block-level KV-cache allocation and finally into physical GPU KV slots.

The complete path identified was:

```text
Scheduler
    ↓
num_new_tokens
    ↓
KVCacheManager.allocate_slots()
    ↓
KVCacheCoordinator
    ↓
SingleTypeKVCacheManager
    ↓
BlockPool
    ↓
physical KV blocks
    ↓
SchedulerOutput
    ↓
ModelRunner block table
    ↓
slot-mapping kernel
```

---

# KVCacheManager.allocate_slots()

The scheduler calls:

```python
self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    ...
)
```

The manager calculates how far the request's KV storage must extend.

In the normal decoder-only case:

```text
required KV positions
≈
num_computed_tokens
+
num_new_tokens
```

The coordinator then determines how many additional blocks are required.

If insufficient blocks are available:

```python
return None
```

which propagates back to the scheduler and can trigger preemption.

---

# Tokens to Blocks

The core conversion happens in `SingleTypeKVCacheManager`:

```python
num_required_blocks = cdiv(
    num_tokens,
    self.block_size
)
```

Conceptually:

```text
num_required_blocks
=
ceil(num_tokens / block_size)
```

For example:

```text
block_size = 16
num_tokens = 50

ceil(50 / 16) = 4 blocks
```

The manager then accounts for blocks already associated with the request:

```python
num_req_blocks = len(
    self.req_to_blocks.get(request_id, ())
)
```

For a normal running request:

```text
new blocks
≈
required blocks
-
already allocated blocks
```

The real implementation additionally handles:

* prefix-cache hits
* sliding-window skipped blocks
* evictable cached blocks
* speculative decoding
* partial cache hits
* multiple KV cache groups

---

# Physical Block Allocation

When additional blocks are needed:

```python
new_blocks = self.block_pool.get_new_blocks(
    num_new_blocks
)
```

The block mapping is then extended:

```python
req_blocks.extend(new_blocks)
```

Conceptually:

```text
Request A logical blocks

Logical 0 → Physical 7
Logical 1 → Physical 21
Logical 2 → Physical 4
```

The physical blocks do not have to be contiguous.

This is the central paged-KV abstraction.

---

# BlockPool

At initialization, vLLM creates descriptors for all GPU KV blocks:

```python
self.blocks = [
    KVCacheBlock(idx)
    for idx in range(num_gpu_blocks)
]
```

Every block contains metadata including:

```text
block_id
ref_cnt
block_hash
free-list links
```

All blocks begin in a free-block queue implemented using a doubly linked list.

Allocation removes blocks from the queue:

```python
self.free_block_queue.popleft_n(num_blocks)
```

and increments:

```python
block.ref_cnt += 1
```

Thus:

```text
FREE
ref_cnt = 0
    ↓ allocate
IN USE
ref_cnt = 1
```

---

# Block Release and Cache-Aware Reuse

When requests release blocks:

```python
block.ref_cnt -= 1
```

A block becomes reusable only when:

```text
ref_cnt == 0
```

vLLM treats uncached and cached free blocks differently.

Uncached blocks are placed at the front of the free queue:

```text
non-cached
→ reuse sooner
→ LIFO-like behavior
```

Cached prefix blocks are placed toward the back:

```text
cached
→ retain longer
→ LRU-like eviction behavior
```

Therefore:

```text
ref_cnt = 0
```

does not necessarily mean the block has no useful information.

A free block can still retain its prefix-cache hash until it must be reused for another allocation.

---

# Prefix Caching

Prefix caching allows multiple requests with the same prefix to reuse previously computed KV blocks.

The lookup path is:

```text
Prompt token IDs
    ↓
chained block hashes
    ↓
FullAttentionManager.find_longest_cache_hit()
    ↓
BlockPool.get_cached_block()
    ↓
cached physical KV blocks
    ↓
hit_length
```

For full attention, vLLM scans hashes from the beginning:

```text
H0 → hit
H1 → hit
H2 → hit
H3 → miss
       ↓
      stop
```

The result is the longest continuously cached prefix.

For full-block lookup:

```text
hit_length
=
number of matched blocks
×
block_size
```

For example:

```text
block_size = 16
75 blocks hit

hit_length = 1200 tokens
```

A 2000-token prompt would then require only approximately:

```text
2000 - 1200
= 800 tokens
```

of new prefill work.

The current implementation also supports finer-grained partial-tail prefix hits.

---

# Chained Block Hashes

Prefix-cache hashes are chained.

Conceptually:

```text
H0 = hash(block0)

H1 = hash(H0, block1)

H2 = hash(H1, block2)
```

Therefore, if one prefix block misses, later hashes cannot represent the same cached prefix chain.

This allows lookup to stop at the first miss.

The block lookup key also incorporates the KV cache group:

```text
(block_hash, group_id)
```

which prevents different cache groups from incorrectly sharing physical blocks.

---

# Scheduler-to-Worker Block Synchronization

For a newly admitted request, the scheduler sends the complete block table through:

```text
NewRequestData.block_ids
```

For requests already cached on the worker, vLLM sends only incremental block updates:

```text
CachedRequestData.new_block_ids
```

The worker then appends new blocks:

```python
block_ids.extend(new_ids)
```

For a request resumed after preemption, the previous physical mapping may be invalid, so the worker replaces the block table instead:

```python
req_state.block_ids = new_block_ids
```

This minimizes scheduler-to-worker communication while preserving correct mappings across preemption.

---

# Worker-Side Block Table

The worker maintains:

```text
CachedRequestState.block_ids
    ↓
InputBatch.block_table
```

The block table maps logical request blocks to physical KV-cache blocks.

Example:

```text
Request A

logical blocks:
0    1    2

physical blocks:
7   21    4
```

The request's physical KV memory therefore does not need to be contiguous.

---

# Logical Token to Physical KV Slot

The final mapping is computed by a Triton kernel:

```text
ComputeSlotMappingKernel
```

For the ordinary single-GPU case, the mapping can be understood as:

```text
logical_block_idx
=
position // block_size

block_offset
=
position % block_size

physical_block
=
block_table[request, logical_block_idx]

physical_slot
=
physical_block * block_size
+
block_offset
```

Example:

```text
block_size = 16

block table:
[7, 21, 4]

token position = 20
```

Then:

```text
logical block = 20 // 16 = 1
offset        = 20 % 16 = 4

physical block = 21

physical slot
= 21 × 16 + 4
= 340
```

This is the core address-translation idea behind paged KV-cache management.

The actual vLLM kernel additionally handles:

* different scheduler and KV-cache block granularities
* context parallelism
* KV-cache interleaving
* local-rank ownership

---

# End-to-End Mental Model

After the first four days, the vLLM execution path can be summarized as:

```text
User Prompt
    ↓
LLM.generate()
    ↓
LLMEngine
    ↓
EngineCore
    ↓
Scheduler.add_request()
    ↓
WAITING
    ↓
Scheduler.schedule()
    ↓
token-budget decision
    ↓
chunked prefill / continuous batching
    ↓
KVCacheManager.allocate_slots()
    ↓
token positions → required KV blocks
    ↓
BlockPool physical allocation
    ↓
request block table
    ↓
SchedulerOutput
    ↓
ModelRunner
    ↓
InputBatch.block_table
    ↓
ComputeSlotMappingKernel
    ↓
physical KV slots
    ↓
attention execution
```

With prefix caching:

```text
Prompt
    ↓
chained block hashes
    ↓
cached physical KV blocks
    ↓
num_computed_tokens > 0
    ↓
skip repeated prefill work
```

With KV pressure:

```text
allocate_slots()
    ↓
insufficient free blocks
    ↓
preemption
    ↓
release KV blocks
    ↓
recompute later
```

---

# Key Takeaways

After the first four days, the main conclusions are:

1. **vLLM unifies prefill and decode through token accounting.**
   Requests are primarily represented by logical token progress rather than separate prefill/decode scheduler states.

2. **Continuous batching emerges from repeated scheduler iterations.**
   Finished requests leave while waiting requests dynamically enter the active batch.

3. **Chunked prefill is a scheduler policy.**
   Large prompts can be split across iterations so that other workloads retain access to compute capacity.

4. **LLM scheduling is both compute scheduling and memory scheduling.**
   Token budget alone is insufficient; KV-cache capacity must also be available.

5. **Paged KV cache separates logical sequence layout from physical GPU layout.**
   Per-request block tables allow physical KV blocks to remain non-contiguous.

6. **Prefix caching is implemented as reusable physical KV blocks indexed by chained prefix hashes.**

7. **Worker-side request state is incremental.**
   Full request state is sent once; later scheduler iterations mostly send state differences.

8. **PagedAttention ultimately relies on address translation.**
   Logical token positions are translated through a block table into physical KV slots before attention execution.

---

# Next Steps

The next phase moves from source-code reading to source-code modification and benchmarking.

## Day 5 — Scheduler Instrumentation

Add scheduler tracing for:

```text
request_id
request status
num_computed_tokens
num_new_tokens
token_budget
free KV blocks
allocated blocks
prefill/decode state
preemption events
```

Export scheduler decisions as JSONL for offline analysis.

## Day 6 — Mixed-Workload Baseline

Build workloads containing:

```text
short prefill
long prefill
decode-heavy requests
mixed concurrency
```

Measure:

```text
TTFT
TPOT
throughput
p50 / p95 latency
KV-cache utilization
preemption count
```

## Day 7 — Scheduling Policy Experiment

Modify the chunked-prefill scheduling policy.

Compare:

```text
baseline
vs.
fixed prefill chunk
vs.
adaptive prefill chunk
```

Study the tradeoff between:

```text
prefill efficiency
decode latency
throughput
fairness
KV-cache pressure
```

---

# Project Direction

The intended final project is:

**vLLM Scheduler and KV-Cache Optimization for Mixed LLM Inference Workloads**

The final deliverable will combine:

```text
source-level framework understanding
+
scheduler instrumentation
+
policy modification
+
controlled benchmarking
+
performance analysis
```

rather than only demonstrating that vLLM can serve a model.

The goal is to understand and experimentally evaluate how inference scheduling and KV-cache management interact under realistic mixed workloads.
