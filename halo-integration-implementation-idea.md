# HALO Integration Implementation Idea

This is an implementation idea document, not a committed integration contract.
It records what HALO appears to do from its docs and source, then maps it to the
model-correction-proxy architecture.

HALO checkout reviewed:

- Path: `/Users/tony/Dev/ThirdParties/halo-ssh`
- Remote: `git@github.com:context-labs/HALO.git`
- Commit: `5fdc06f`
- Note: the repo uses Git LFS. `git-lfs` was installed locally with Nix so the
  SSH checkout could complete.

Primary source files reviewed:

- `README.md`
- `halo_cli/README.md`
- `halo_cli/main.py`
- `engine/main.py`
- `engine/agents/*`
- `engine/tools/*`
- `engine/traces/*`
- `engine/sandbox/README.md`
- `docs/integrations/openai-agents-sdk.md`
- `demo/appworld/README.md`
- `demo/appworld/HALO_PATCH.md`
- `tests/probes/probe_kit.py`

## Short Version

HALO is not a better replacement for this proxy. It works at a different layer.

The proxy is a live compatibility runtime:

- intercept OpenAI-compatible requests
- adapt prompts/tools for a target model
- interpret upstream responses
- repair malformed behavior before the client agent loop breaks
- trace the transaction

HALO is an offline or asynchronous outer improvement loop:

- collect execution traces from an agent harness
- index and query the trace dataset
- let a specialized trace-analysis agent inspect systemic failures
- produce a report with concrete harness fixes
- feed that report to a coding agent or developer
- redeploy, collect more traces, repeat

The two systems fit together well:

- The proxy fixes the current request.
- HALO helps improve the proxy after seeing many requests.

That means HALO can plausibly reduce correction-agent usage over time, but only
if its findings are fed back into proxy code, model profiles, prompts, parser
policies, eval suites, and GEPA/dsrs optimization data. HALO does not directly
make live correction cheaper by itself. It improves the system by finding common
failure modes so they can be prevented or repaired more deterministically.

So the answer to "would something like HALO improve correction rate and reduce
correction-agent triggers?" is yes, if we integrate it as a disciplined feedback
loop rather than another live repair step. It should make the proxy smarter
between runs, not busier during a single user request.

Current related work in this repo: `eval/trace-harness` now provides a smaller
version of the same feedback-loop idea without pulling HALO into the runtime. It
imports third-party agent traces, such as Pi Mono and Hermes agent-reasoning
samples, converts them into neutral OpenAI-compatible scenarios, runs sampled
live structural checks through the proxy, and lets reviewed edge cases flow into
request-adapter GEPA datasets. That is not HALO integration, but it is useful
groundwork for a later HALO-style trace-analysis loop.

## What HALO Is

HALO stands for Hierarchical Agent Loop Optimization. Its README describes it as
an RLM-based automatic agent optimization loop for recursively self-improving
agent harnesses.

The core loop in the HALO README is:

1. Collect execution traces from an agent harness.
2. Feed traces into the HALO-RLM engine.
3. The engine decomposes traces to understand common failure modes across
   executions and produces a report.
4. The report is fed into a coding agent, such as Cursor or Claude Code, to
   generate and apply harness changes.
5. The harness is redeployed, more traces are gathered, and the cycle repeats.

HALO's stated motivation is that general-purpose coding agents are not ideal for
long trace analysis. They can overfit to one or a few examples. HALO gives the
analysis model a specialized trace toolkit so it can inspect a trace dataset
systematically.

## HALO Architecture From The Code

### CLI

The CLI entrypoint is `halo_cli/main.py`.

User shape:

```sh
halo TRACE_PATH --prompt "Diagnose errors you find and suggest fixes"
```

Important options:

- `--model`: model used for root, subagent, synthesis, and compaction
- `--max-depth`: maximum subagent recursion depth
- `--max-turns`: maximum turns per agent
- `--max-parallel`: max concurrent subagents
- `--instructions`: override HALO's default trace-tool instructions
- `--reasoning-effort`: forwarded to supported reasoning models

The CLI requires `OPENAI_API_KEY`. The model/provider layer is
OpenAI-compatible and can also be configured with a base URL.

### Engine Entry Point

The main runtime entrypoint is `engine/main.py`.

`stream_engine_async(...)` does roughly this:

1. Configure the OpenAI Agents SDK client.
2. Get the sandbox if available.
3. Ensure the trace sidecar index exists with `TraceIndexBuilder`.
4. Load a `TraceStore`.
5. Create an `EngineOutputBus`.
6. Build an `EngineRunState` containing config, trace store, output bus,
   sandbox, and optional runner seam.
7. Register a root `AgentExecution`.
8. Build a root `AgentContext`.
9. Build a root SDK agent with trace tools and optional subagent tool.
10. Run the root agent through `OpenAiAgentRunner`.
11. Stream durable output items and text deltas through the output bus.

This is a trace-analysis agent runtime, not a proxy or live request handler.

### Trace Input Shape

HALO consumes OTLP-shaped JSONL where each line is one span.

The OpenAI Agents SDK integration doc describes the span shape:

- `trace_id`
- `span_id`
- `parent_span_id`
- `name`
- `kind`
- `start_time`
- `end_time`
- `status`
- `resource.attributes`
- `scope`
- `attributes`

Important attributes:

- `openinference.span.kind`: `AGENT`, `LLM`, `TOOL`, `CHAIN`, etc.
- `inference.export.schema_version`
- `inference.project_id`
- `inference.observation_kind`
- `inference.llm.provider`
- `inference.llm.model_name`
- `inference.llm.input_tokens`
- `inference.llm.output_tokens`
- `inference.agent_name`
- `tool.name`
- `input.value`
- `output.value`
- `llm.input_messages`
- `llm.output_messages`

The integration guide provides a self-contained `tracing.py` for OpenAI Agents
SDK apps. It registers a tracing processor and appends spans to JSONL.

### Trace Index

`engine/traces/trace_index_builder.py` builds a sidecar index next to the trace
file.

The index stores one row per trace id:

- byte offsets
- byte lengths
- span count
- time bounds
- error flag
- service names
- model names
- agent names
- token totals
- project id

It reuses an index if the trace file size and mtime still match. Large files are
processed with a staged approach:

1. sequentially scan JSONL line offsets
2. split offsets into worker chunks
3. parse spans and accumulate per trace id
4. merge rows
5. write index and metadata atomically

This matters for us because our current traces are readable, but not yet very
queryable at scale.

### Trace Store And Tools

`engine/traces/trace_store.py` exposes a read/query API over the trace file and
index.

The HALO agent gets these trace tools:

- `get_dataset_overview`
- `query_traces`
- `count_traces`
- `view_trace`
- `view_spans`
- `search_trace`
- `synthesize_traces`
- `get_context_item`
- `run_code` when the sandbox is available
- `call_subagent` when depth allows it

The tool design is important:

- The agent must call `get_dataset_overview` first to get real trace ids.
- It can query/count traces by filters.
- It can view small traces directly.
- It must use search plus surgical span reads for large traces.
- Large payloads are truncated, but the tool response tells the agent how to
  get more specific data.

HALO has two attribute caps:

- discovery cap around 4KB per attribute for `view_trace` and `search_trace`
- surgical cap around 16KB per attribute for `view_spans`

`view_trace` also has a total response budget. If a trace is too large, it
returns an oversized summary instead of dumping all spans into context.

This is exactly the kind of ergonomics our trace system currently lacks.

### Agent Runtime

The root agent and subagents are built through
`engine/tools/subagent_tool_factory.py`.

Important details:

- root and subagents are OpenAI Agents SDK agents
- subagent spawning is bounded by maximum depth
- parallelism is bounded by per-depth semaphores
- each agent has an `AgentContext`
- output from root and subagents is interleaved through one `EngineOutputBus`
- output items carry lineage fields:
  - `agent_id`
  - `parent_agent_id`
  - `parent_tool_call_id`
  - `depth`
  - `sequence`

The subagent design is practical for trace analysis. It allows a root agent to
delegate focused trace questions without losing boundedness.

### Prompting Contract

`engine/agents/prompt_templates.py` defines the trace-analysis behavior.

The default system prompt is a tool-usage manual. It tells the model:

- call `get_dataset_overview` first
- do not fabricate trace ids
- use `query_traces` and `count_traces` for dataset-level questions
- use `view_trace` only for small traces
- use `search_trace` and `view_spans` for large traces
- stop and reconsider when a tool errors
- do not retry with guessed ids or arguments

The root agent must emit a final sentinel, `<final/>`, at the end of its final
answer. Subagents must not emit it.

### Context Compaction

`engine/agents/agent_context.py` and `engine/agents/compactor.py` implement
context compaction.

Important details:

- each agent owns its own context
- old text messages can be compacted
- old tool-call turns can be compacted
- assistant tool calls and matching tool results are compacted together
- compacted tool results render as assistant text, avoiding orphan `tool` role
  messages
- the compactor model is separate from the root/subagent/synthesis models
- the compaction prompt preserves tool names, argument shapes, and key facts

This is relevant because our proxy already learned that preserving prior
assistant tool calls and tool results is essential for multi-turn model behavior.

### Synthesis Tool

`engine/tools/synthesis_tool.py` lets the agent summarize selected traces with a
separate model call.

It renders chosen trace ids with a per-trace budget and asks a synthesis model to
produce a short summary with concrete trace ids, error patterns, model names,
and token counts.

### Run Code Tool And Sandbox

`engine/tools/run_code_tool.py` exposes a `run_code` tool backed by
`engine/sandbox`.

The sandbox uses Deno plus Pyodide:

- no network
- no host writes
- no environment access
- no subprocess spawning
- explicit read allowlist for trace/index/runtime files
- fresh subprocess per call
- numpy, pandas, pydantic, and the real `TraceStore` are available

This lets the model write small analysis scripts over the trace dataset without
letting it touch the host freely.

This is useful but optional for a first integration with this proxy. The bigger
near-term win is the indexed trace tool model.

### AppWorld Demo

HALO includes a vendored AppWorld demo wired to emit HALO-shaped traces from an
OpenAI Agents SDK harness.

The AppWorld loop:

1. run benchmark split
2. collect traces and eval results
3. run HALO analysis
4. edit the harness
5. rerun and compare eval reports

The demo calls out likely harness improvement surfaces:

- agent loop
- API predictor
- system prompt template
- few-shot demonstrations
- model configs

That is very similar to our likely proxy improvement surfaces:

- model profiles
- DSRs tool contract prompt
- parser policy
- correction-agent prompt
- retry/continue strategy
- eval/regression cases

## How This Maps To The Proxy

The proxy architecture is:

```text
Client application
  -> OpenAI-compatible proxy endpoint
    -> request normalizer
    -> model profile resolver
    -> prompt/tool adapter
    -> upstream OpenAI-compatible model call
    -> response interpreter
    -> repair / correction pipeline
    -> trace store
  -> OpenAI-compatible response
```

HALO should sit outside that live request path:

```text
Live path:
client -> model-correction-proxy -> upstream model -> repaired response -> client

Outer loop:
proxy traces -> HALO-compatible trace export -> HALO analysis -> report
  -> Codex/developer changes proxy prompts/profiles/parsers/evals
  -> redeploy proxy -> collect better traces
```

This keeps latency and risk out of the live path while giving us a systematic
way to improve the proxy over time.

## Why This Could Reduce Correction-Agent Usage

The user hypothesis is correct with one caveat.

Correct:

- If HALO finds recurring failure patterns, we can prevent them upstream with
  better DSRs prompts, model profiles, and tool rendering.
- If HALO identifies common harmless syntax/shape problems, we can add focused
  deterministic repairs and tests.
- If HALO shows correction-agent prompts are weak, we can improve or optimize
  those prompts with better datasets.
- If HALO shows a profile consistently fails with one tool format, we can change
  that model's profile.
- If recurring failures become regression tests, future changes become safer.

That should reduce correction-agent trigger rate, lower cost, and reduce
semantic drift risk because fewer live requests require model-based repair.

Caveat:

- HALO does not directly correct live responses.
- It only improves correction rate when its findings are turned into proxy
  changes and validated with evals.
- The live correction agent still matters for long-tail failures and newly
  discovered model/provider behavior.

So the right framing is:

```text
Correction agent = runtime fallback for individual failures.
HALO = offline discovery loop that teaches the runtime to need fewer fallbacks.
```

## Integration Strategy

### Phase 0: Keep HALO Out Of The Live Path

Do not call HALO during `POST /v1/chat/completions`.

Reasons:

- HALO is multi-agent and potentially expensive.
- HALO expects datasets, not single transaction repair.
- HALO produces analysis and recommendations, not client-facing responses.
- Calling it live would add latency and another failure mode.

### Phase 1: Finish Proxy Trace Quality

Before HALO can help, our traces need enough structure to analyze.

This lines up with `reliability-refactor-plan.tmp.md`:

- typed response failures
- raw upstream assistant content preserved
- correction attempts recorded
- policy decisions recorded
- repair actions recorded
- model profile and adapter decisions recorded
- final response recorded

Minimum additional fields we should expose in traces:

- `failure_kinds`
- `policy_decision`
- `correction_attempts`
- `correction_model`
- `correction_possible`
- `correction_confidence`
- `correction_error`
- `upstream_content_raw`
- `upstream_reasoning_raw`
- `interpreted_tool_intents`
- `repair_actions`
- `final_tool_calls`

### Phase 2: Add A HALO Exporter Or Converter

HALO expects one OTLP-style span per JSONL line. Our current trace format is one
nested transaction record per line.

The least disruptive integration is a converter:

```sh
cargo run -- export-halo-traces \
  --trace-path traces/model-correction-proxy.jsonl \
  --output-path traces/model-correction-proxy.halo.jsonl
```

The converter can preserve our native trace format while producing HALO-shaped
span JSONL as a derived artifact.

Alternative later:

- emit HALO/OTLP spans directly during request handling
- keep native transaction JSONL for replay/export
- optionally write both

Start with a converter. It avoids committing the live runtime to a tracing
schema before the proxy internals settle.

### Phase 3: Define Proxy Span Model

One proxy request should probably become one HALO trace.

Candidate span tree:

```text
trace_id = proxy trace_id

model_correction_proxy.request                 (AGENT or CHAIN)
|-- gateway.normalize_request                  (SPAN)
|-- profile.resolve                            (SPAN)
|-- prompt_adapter.adapt_request               (SPAN)
|-- upstream.chat_completions                  (LLM)
|-- response_interpreter.interpret             (SPAN)
|-- correction_agent.correct                   (LLM) optional
|-- repair.apply_policy                        (SPAN)
`-- response.build_client_response             (SPAN)
```

Span status should reflect stage success/failure:

- `STATUS_CODE_OK` for normal stages
- `STATUS_CODE_ERROR` for upstream errors, parser panics, correction-agent
  failures, unrecoverable repair, or gateway errors

### Phase 4: Attribute Mapping

Every span should include common attributes:

```text
inference.export.schema_version = 1
inference.project_id = "model-correction-proxy"
service.name = "model-correction-proxy"
mcp.trace_id = <trace id>
mcp.stage = <stage name>
mcp.model = <requested model>
mcp.profile = <profile name>
```

Request/root span attributes:

```text
mcp.request.message_count
mcp.request.tool_count
mcp.request.stream_requested
mcp.request.parallel_tool_calls
mcp.request.tool_choice
mcp.request.latest_user_preview
```

Prompt adapter span attributes:

```text
mcp.adapter.mode
mcp.adapter.tool_format
mcp.adapter.instruction_len
mcp.adapter.upstream_tool_count
mcp.adapter.upstream_stream
```

Upstream LLM span attributes:

```text
openinference.span.kind = "LLM"
inference.observation_kind = "LLM"
inference.llm.provider
inference.llm.model_name
inference.llm.input_tokens
inference.llm.output_tokens
llm.input_messages
llm.output_messages
mcp.upstream.finish_reason
mcp.upstream.content_len
mcp.upstream.reasoning_len
```

Interpreter span attributes:

```text
mcp.interpreted.content_len
mcp.interpreted.reasoning_present
mcp.interpreted.tool_intents
mcp.interpreted.suspicious_stop
mcp.interpreted.parse_events
mcp.interpreted.failure_kinds
```

Correction span attributes:

```text
openinference.span.kind = "LLM"
inference.observation_kind = "LLM"
mcp.correction.enabled
mcp.correction.model
mcp.correction.input_parser_events
mcp.correction.input_failure_kinds
mcp.correction.malformed_response_preview
mcp.correction.possible
mcp.correction.confidence
mcp.correction.explanation
mcp.correction.tool_calls
mcp.correction.content_len
mcp.correction.error
```

Repair span attributes:

```text
mcp.repair.actions
mcp.repair.final_finish_reason
mcp.repair.final_tool_call_count
mcp.repair.final_content_len
mcp.repair.suppressed
mcp.repair.replaced_unusable_content
```

Potentially sensitive raw content should be configurable:

- full raw payload in local trusted mode
- previews only in safer mode
- redacted mode for sharing

### Phase 5: Add HALO Analysis Prompt Pack

We should provide canned prompts for HALO over proxy traces.

Examples:

```text
Find the most common failure kinds by model profile. Which failures are
systemic rather than one-off?
```

```text
Which failures caused correction_agent_tool_recovery? For each recurring
cluster, say whether the fix should be prompt/profile, deterministic parser,
schema repair, retry/continue, or correction prompt improvement.
```

```text
Find traces where the correction agent failed, returned low confidence, or
recovered content when a tool call was likely intended. Group by model and
failure kind.
```

```text
For qwen-dsrs traces, inspect DSRs contract violations. Which malformed
patterns are prompt-template leaks versus tool_calls JSON shape problems?
Suggest profile prompt changes and regression tests.
```

```text
Find cases where the proxy returned a fallback/suppression message. Were these
safe suppressions, false negatives, or missing correction-agent opportunities?
```

```text
Which repair actions are most frequent? Which ones should become model-profile
guidance so the model stops producing those failures?
```

### Phase 6: Feed HALO Findings Back Into The Proxy

HALO reports should become concrete changes in these places:

| Finding | Likely Proxy Surface |
| --- | --- |
| model copies DSRs prompt markers | `src/model_profile.rs`, `src/dsrs_contract.rs` prompt guidance |
| repeated invalid tagged `tool_calls` shape | DSRs parser tests, correction prompt, profile examples |
| repeated untagged JSON tool arrays | prompt guidance, typed failure routing |
| correction agent often recovers same pattern | deterministic repair or parser tolerance |
| correction agent invents arguments | correction prompt, confidence threshold, optional judge |
| specific model overuses parallel tools | model profile parallel guidance |
| specific model stops after "I'll read..." | retry/continue policy and profile guidance |
| schema key typos recur | schema-guided repair tests, profile tool schema rendering |
| hallucinated tool names recur | tool description/rendering, hallucination policy, correction examples |

### Phase 7: Connect To Existing Dataset/Eval/GEPA Work

HALO should not replace our dataset and eval code. It should help select and
cluster better examples.

Integration points:

- `src/dataset.rs`: export raw malformed responses and typed failures
- `src/eval.rs`: add regression cases from HALO-identified clusters
- `src/replay.rs`: replay traces after parser/policy changes
- `src/optimization.rs`: use clustered correction examples for GEPA prompt
  optimization
- `src/model_profile.rs`: update model-specific guidance from recurring failure
  patterns

Suggested flow:

```text
proxy traces
  -> HALO export
  -> HALO analysis report
  -> selected trace ids / failure clusters
  -> dataset export rows
  -> regression cases
  -> prompt/profile/parser changes
  -> replay/eval
  -> deploy
```

## Example HALO Export Span

Sketch only:

```json
{
  "trace_id": "trace_82f983a8c26c45959da7d8699538dbf0",
  "span_id": "span_interpret",
  "parent_span_id": "span_upstream",
  "trace_state": "",
  "name": "response_interpreter.interpret",
  "kind": "SPAN_KIND_INTERNAL",
  "start_time": "2026-05-01T19:40:49.025103000Z",
  "end_time": "2026-05-01T19:40:49.025903000Z",
  "status": {
    "code": "STATUS_CODE_ERROR",
    "message": "DSRs contract violation"
  },
  "resource": {
    "attributes": {
      "service.name": "model-correction-proxy"
    }
  },
  "scope": {
    "name": "model-correction-proxy",
    "version": "dev"
  },
  "attributes": {
    "openinference.span.kind": "CHAIN",
    "inference.export.schema_version": 1,
    "inference.project_id": "model-correction-proxy",
    "inference.observation_kind": "SPAN",
    "inference.llm.model_name": "qwen/qwen3.5-9b",
    "inference.agent_name": "response_interpreter",
    "mcp.profile": "qwen-dsrs",
    "mcp.stage": "response_interpreter",
    "mcp.interpreted.failure_kinds": "[\"DsrsContentOutsideTaggedFields\",\"DsrsInvalidToolCallsJson\",\"TemplateLeak\"]",
    "mcp.interpreted.parse_events": "[\"parsed response through DSRs tool-use contract\",\"DSRs tool_calls field was not valid JSON\"]",
    "mcp.interpreted.tool_intents": "[]",
    "mcp.interpreted.content_preview": "---"
  }
}
```

## What We Should Not Do

Do not put HALO in the live response path.

Do not ask HALO to repair a single malformed model response for the client.
That is the correction agent's job.

Do not let HALO automatically modify the proxy without review. HALO reports
should become PRs, tests, or explicit implementation tasks.

Do not optimize only against one trace. HALO's value is seeing systemic
patterns across a dataset.

Do not assume the HALO trace schema is stable forever. Start with a converter
and keep our native trace format until the proxy trace model settles.

Do not export sensitive traces without a redaction story. Proxy traces can
contain prompts, tool args, file names, command strings, and possibly secrets.

## Open Questions

1. Should one HALO trace equal one proxy request, or should we stitch multiple
   proxy requests into a conversation/session trace?

   Initial answer: one proxy request per HALO trace is simpler. Add
   `conversation_id` later if clients provide stable IDs or if we derive them.

2. Should the live proxy emit OTLP-shaped spans directly?

   Initial answer: no. Build a converter first.

3. Should correction-agent prompts and outputs be included in full?

   Initial answer: yes for local trusted traces, preview/redacted for shareable
   traces.

4. Can HALO run on our current JSONL traces without conversion?

   Likely no. HALO expects span-shaped JSONL with top-level `trace_id`,
   `span_id`, `status`, `resource`, and `attributes`. Our current native trace
   format is one nested transaction object per line.

5. Should we adopt HALO's Python engine as a dependency or just interoperate via
   exported trace files?

   Initial answer: interoperate via files first. Depending on the Python package
   from the Rust proxy would add operational complexity. A file-level boundary is
   cleaner.

6. Can HALO analyze traces from live Pi/proxy sessions?

   Yes, if we export enough raw and structured data. The trace needs to preserve
   prompt adaptation, upstream response, interpreted failures, correction attempts,
   and final response.

## Proposed Roadmap

### Step 1: Finish Reliability Trace Fields

Finish the typed-failure/correction-trace work described in
`reliability-refactor-plan.tmp.md`.

This is the most important prerequisite. HALO cannot find systemic failures if
we collapse raw failures into lossy strings like `"---"`.

### Step 2: Add HALO Export Command

Add a CLI command:

```sh
cargo run -- export-halo-traces \
  --trace-path traces/model-correction-proxy.jsonl \
  --output-path traces/model-correction-proxy.halo.jsonl
```

The command should:

- read native proxy trace records
- emit one HALO trace per proxy request
- emit multiple spans per request
- include parser events and typed failure kinds
- include correction attempts
- include repair actions
- include raw payload previews or full payloads based on config

### Step 3: Validate With HALO

Run:

```sh
halo traces/model-correction-proxy.halo.jsonl \
  -p "Find recurring model-correction-proxy failure modes and suggest concrete proxy changes"
```

Manually validate whether HALO's claims are grounded in trace ids and actual
raw evidence.

### Step 4: Add Proxy-Specific HALO Prompts

Create a small prompt pack in the repo, for example:

```text
halo-prompts/
  correction-rate.md
  dsrs-contract-violations.md
  profile-improvements.md
  correction-agent-failures.md
  regression-candidates.md
```

These are not live prompts. They are operator prompts for offline analysis.

### Step 5: Feed Reports Into Tests And Profiles

For each accepted HALO finding:

- add or update regression tests
- update model profile guidance
- update parser/policy logic only if the fix is general
- add correction-agent examples
- run replay/eval
- compare correction-agent usage and final tool-call recovery rates

### Step 6: Automate The Outer Loop Later

Once the manual loop is trusted:

- nightly or manual command exports HALO traces
- HALO generates a report
- report is saved under `analysis/halo/YYYY-MM-DD.md`
- Codex/developer turns report into patches
- CI runs replay/eval

Do not automate code edits until the report quality is proven.

## Expected Impact

If this works, it should improve the proxy in exactly the way the original
intent describes:

- fewer silent agent-loop stops
- fewer repeated malformed DSRs outputs
- fewer correction-agent calls for known patterns
- lower runtime cost
- lower semantic drift from correction agents
- better model-specific profiles
- better regression tests from real failures
- better prompt optimization datasets
- clearer debugging when a failure happens

The live correction agent remains necessary. The goal is to make it a
high-value fallback for rare or ambiguous failures, not the first line of
defense for common model/profile mistakes.

## Success Metrics

The integration should be judged by measured proxy behavior, not just nicer
reports.

Useful metrics:

- correction-agent invocation rate by model/profile
- correction-agent success rate by failure kind
- deterministic repair rate by failure kind
- suppression/fallback rate
- false correction rate
- valid tool-call recovery rate
- malformed DSRs contract rate
- latency added by repair/correction stages
- cost per successful proxied request
- regression pass rate on HALO-derived trace cases

The main target is not zero correction-agent calls. The target is fewer repeated
correction-agent calls for known patterns, with equal or better final response
quality.

## Practical First Cut

The smallest useful integration is:

1. Finish typed failures and correction trace details.
2. Add a native-trace-to-HALO converter.
3. Run HALO over a few hundred proxy traces.
4. Ask it specifically:

   ```text
   Which recurring failures caused correction-agent usage, and which of them
   should be fixed by prompt/profile changes versus deterministic parser/policy
   changes?
   ```

5. Convert the top two or three findings into tests and patches.

That gives us the HALO feedback loop without making the live proxy more complex.
