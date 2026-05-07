# Model Correction Proxy

Model Correction Proxy is an OpenAI-compatible behavior compatibility layer for agent applications.

It sits between an existing OpenAI-compatible client and an upstream model provider, watches the request/response contract, and repairs model behavior before the application loop breaks. The first target is tool-calling reliability for open-source and OpenAI-compatible models, especially when inference-engine tool parsers, chat templates, or provider quirks turn an intended tool call into a plain assistant message.

## Why this exists

Many open-source models are much more capable than their integration experience suggests. A common failure mode is not that the model is unable to use a tool, but that the surrounding API or inference engine fails to preserve the model's intended tool call.

Typical symptoms:

- the model says "I'll read the file now" and stops without a tool call
- a malformed JSON argument causes native tool parsing to fail
- a model emits a clear tool call in text, but the client expects OpenAI `tool_calls`
- reasoning/thinking fields contain text the client expects in `content`
- a provider reports `finish_reason = stop` even though the response looks like an incomplete or misplaced tool call

This proxy treats tool use as an application-level contract rather than an inference-engine feature to blindly trust. It can render tools into model-friendly text, parse plain-text tool intent, repair JSON/schema issues, record traces, export datasets, replay failures, and optimize correction prompts with DSRs/GEPA.

## Current status

Implemented scope currently covers the first three phases, with the runtime centered on a DSRs-style tool-use contract and a first-class correction-agent path:

- **Phase 1:** OpenAI-compatible chat proxy, request normalization, upstream OpenAI-compatible calls, deterministic repair, DSRs correction agent path, JSONL traces
- **Phase 2:** proxy-owned DSRs tool rendering, model profiles, multi-tool parsing, parallel-tool enforcement, schema-guided repair, inbound auth passthrough, `/v1/models` proxying, buffered client SSE responses
- **Phase 3:** trace-to-dataset export, replay harness, regression evaluation, typed DSRs correction-agent traces, profile-aware config loading, DSRs/GEPA prompt artifacts, explicit artifact promotion into model-profile config

The proxy intentionally buffers upstream chat completions so it can interpret and repair the complete output before returning a client-compatible response. If the client requests `stream: true`, the proxy returns a corrected SSE response after buffering and repair; it does not yet proxy upstream tokens incrementally.

## Architecture

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

### Main components

| Component | Source | Purpose |
| --- | --- | --- |
| Gateway | `src/gateway.rs` | Axum HTTP server exposing `/health`, `/v1/models`, and `/v1/chat/completions` |
| OpenAI types | `src/openai.rs` | OpenAI-compatible request/response structures |
| Normalizer | `src/normalizer.rs` | Converts incoming requests into internal `NormalizedRequest` |
| Model profiles | `src/model_profile.rs` | Chooses model-specific tool rendering and repair defaults |
| Prompt adapter | `src/prompt_adapter.rs` | Converts native tools into a proxy-owned DSRs contract, with XML/tagged renderers still available |
| Upstream client | `src/upstream.rs` | Calls generic OpenAI-compatible `/chat/completions` and `/models` APIs |
| Response interpreter | `src/response_interpreter.rs` | Detects native, DSRs, XML, tagged JSON, markdown JSON, direct JSON, and function-like tool intent |
| Repair engine | `src/repair.rs` | Repairs tool names, JSON arguments, schema key typos, scalar/array shape issues, and parallel-tool violations |
| Correction agents | `src/agents.rs` | Typed DSRs correction-agent path for suspicious malformed responses |
| Traces | `src/trace.rs` | Local JSONL request traces and correction-agent sidecar traces for audit and dataset generation |
| Dataset export | `src/dataset.rs` | Converts traces into correction dataset rows |
| Replay | `src/replay.rs` | Replays recorded traces through the repair pipeline |
| Evaluation | `src/eval.rs` | Runs regression suites and reports correction metrics |
| Optimization | `src/optimization.rs` | DSRs/GEPA optimization for correction-agent and request-adapter prompt artifacts |
| Artifact promotion | `src/promotion.rs` | Validates GEPA artifacts and promotes request-adapter or correction-agent instructions into profile config |

## Tool-calling model

The proxy supports two early modes:

1. **Proxy-owned DSRs tool rendering**
   - The proxy consumes OpenAI-style `tools`.
   - It removes native upstream tool definitions.
   - It renders the conversation, tool definitions, `tool_choice`, and parallel-call setting into a DSRs contract using `dspy-rs`.
   - It parses the model's text back into OpenAI `tool_calls`.
   - It deterministically parses tagged DSRs output and keeps untagged DSRs-like output, contract violations, and empty DSRs completions on the correction path instead of treating them as valid contract output.

2. **Pass-through repair**
   - The proxy can preserve native upstream tool calling behavior.
   - It still interprets and repairs malformed or misplaced outputs after the upstream response.

Proxy-owned DSRs rendering is the primary reliability path because it avoids relying on brittle inference-engine tool parsers. XML and tagged JSON parsing are still supported as compatibility paths, and the older XML/tagged renderers remain available in code for future profiles or experiments.

## Built-in model profiles

Built-in profiles are selected by model-name substring:

| Profile | Matching models | Tool format |
| --- | --- | --- |
| `qwen-dsrs` | `qwen`, `qwq` | DSRs |
| `kimi-dsrs` | `kimi`, `moonshot` | DSRs with compact-output guidance |
| `glm-dsrs` | `glm`, `z-ai` | DSRs |
| `llama-dsrs` | `llama` | DSRs |
| `gemma-dsrs-conservative` | `gemma` | DSRs, conservative parallel-tool defaults |
| `balanced-default` | fallback | DSRs |

Profiles can be provided in TOML, JSON, or JSON5 through `--config` or `MCP_CONFIG_PATH`. Configured profiles are matched before built-ins, so users can tune or replace a built-in profile without recompiling. Unknown models still fall back to `balanced-default`.

Profiles support revision/source metadata plus request-adapter and correction-agent artifact IDs. When `request_adapter_artifact` or `correction_agent_artifact` points at a JSON GEPA report or text prompt file, the proxy loads the instruction at startup and records the artifact IDs in traces. See [`docs/model-configurability.md`](docs/model-configurability.md).

## What the repair engine handles

Current deterministic and schema-guided repairs include:

- tool calls embedded in assistant `content`
- DSRs contract output such as:

  ```text
  [[ ## content ## ]]

  [[ ## tool_calls ## ]]
  [{"name":"read_file","arguments":{"path":"Cargo.toml"}}]
  [[ ## completed ## ]]
  ```

- tagged DSRs variants seen from smaller live models, including placeholder values, repeated field blocks, and malformed JSON inside the tagged `tool_calls` field
- routing untagged DSRs-like contract violations, such as `content: ... tool_calls: []` or adjacent `[]` plus tool-call JSON, to the correction-agent path rather than accepting them as valid DSRs
- XML tool calls such as:

  ```xml
  <tool_call name="read_file">{"path":"Cargo.toml"}</tool_call>
  ```

- nested XML tool calls such as:

  ```xml
  <tool_call>
    <name>read_file</name>
    <arguments>{"path":"Cargo.toml"}</arguments>
  </tool_call>
  ```

- tagged JSON tool calls such as:

  ```xml
  <tool_calls_json>
  {"tool_calls":[{"name":"read_file","arguments":{"path":"Cargo.toml"}}]}
  </tool_calls_json>
  ```

- markdown JSON blocks
- direct JSON tool-call objects
- function-like text for known tool names
- JSON5-like malformed arguments, including unquoted keys, single quotes, and trailing commas
- fuzzy near-match tool-name repair
- simple schema key typo repair, such as `pth` -> `path`
- scalar-to-string coercion when the schema expects a string
- value-to-array wrapping when the schema expects an array
- non-object argument wrapping when there is exactly one required property
- preserving or mapping reasoning/thinking text into `content` when needed for client compatibility
- truncating multiple tool calls when the client disables parallel tool calls

Low-confidence semantic invention is intentionally avoided. If a hallucinated tool is not a close match, the proxy can pass it through rather than silently mapping it to a real tool.

## Correction-agent path

When deterministic parsing finds a suspicious stop, malformed known tool call, DSRs contract violation, empty DSRs output, prompt/template leak, invalid DSRs `tool_calls`, schema violation, or similar failure that needs semantic recovery, the repair pipeline can call the correction agent.

The current live correction agent is itself a DSRs-style internal agent. It is asked to fill typed output fields:

- `possible`
- `confidence`
- `explanation`
- `content`
- `tool_calls`

`tool_calls` is a JSON array of `{ "name": string, "arguments": object }`. The proxy does not ask the correction agent for an OpenAI `tool_calls` envelope; the proxy parses the DSRs fields and then builds the OpenAI-compatible response itself.

The correction model defaults to the same model requested by the client, unless `CorrectionConfig.default_model` or the selected `ModelProfile.correction_model` overrides it. Correction-agent instructions can be profile-specific and loaded from a GEPA artifact. Correction is synchronous in the request path so the client receives the repaired response before its agent loop continues.

## Traceability

Every completed request can produce a JSONL trace containing:

- original request
- normalized request
- selected model profile
- profile revision/source and request/correction artifact IDs
- adapted upstream request
- raw upstream response
- interpreted response
- repair actions and confidence metadata
- correction attempts and policy decisions
- final response
- errors, if any
- a compact `summary` block with the latest user prompt, upstream/final previews, parser events, suspicious-stop state, tool intents, and repair actions

Responses include an `x-model-correction-trace-id` header when a trace is written.

Correction-agent attempts are also written to a sidecar JSONL file, defaulting to `traces/model-correction-proxy-corrections.jsonl`. These records include the parent trace ID, correction input, raw DSRs correction output, accepted/rejected result, confidence, explanation, and recovered content or tool calls.

Trace files can contain prompts, model outputs, and tool arguments. Authorization, cookie, and key-like headers are redacted, but trace storage should still be treated as sensitive application data.

For readable trace triage:

```sh
nix develop --command cargo run -- inspect-traces --limit 20
```

Use `--json` to print only compact summaries as pretty JSON.

## Installation

This project is developed with Nix. Use the flake shell so Rust, Cargo, Clippy, rustfmt, OpenSSL, certificates, `curl`, and `jq` are consistent.

```sh
nix develop
```

Or run commands directly through the flake shell:

```sh
nix develop --command cargo test --all-targets
```

Running Cargo outside Nix may require installing a system C toolchain and TLS dependencies manually.

## Running the proxy

Serve is the default command:

```sh
nix develop --command cargo run --
```

Equivalent explicit form:

```sh
nix develop --command cargo run -- serve
```

With OpenRouter:

```sh
export OPENROUTER_API_KEY="..."

nix develop --command cargo run -- \
  --bind 127.0.0.1:8080 \
  --upstream-base-url https://openrouter.ai/api/v1 \
  --trace-path traces/model-correction-proxy.jsonl \
  serve
```

The server listens on:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Upstream chat completions are always requested with `stream: false` so the proxy can inspect the whole response. Client streaming requests are accepted and returned as corrected OpenAI-style SSE after repair.

Other generation controls, including `max_tokens` and `max_completion_tokens`, are passed through when the client provides them and left unset when the client omits them.

### Runtime options

| Option | Env var | Default |
| --- | --- | --- |
| `--bind` | `MCP_BIND_ADDR` | `127.0.0.1:8080` |
| `--config` | `MCP_CONFIG_PATH` | unset |
| `--upstream-base-url` | `MCP_UPSTREAM_BASE_URL` | `https://openrouter.ai/api/v1` |
| `--upstream-api-key` | `MCP_UPSTREAM_API_KEY` | unset |
| `--trace-path` | `MCP_TRACE_PATH` | `traces/model-correction-proxy.jsonl` |

For `serve`, the upstream API key also falls back to `OPENROUTER_API_KEY`.

If no upstream API key is configured, the proxy forwards the inbound `Authorization` header to the upstream provider.

## Example request

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen/qwen3.5-9b",
    "temperature": 0,
    "messages": [
      {
        "role": "user",
        "content": "Read Cargo.toml"
      }
    ],
    "parallel_tool_calls": false,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "description": "Read a project file",
          "parameters": {
            "type": "object",
            "properties": {
              "path": { "type": "string" }
            },
            "required": ["path"]
          }
        }
      }
    ]
  }'
```

If the upstream model emits:

```xml
<tool_call name="read_file">{pth:"Cargo.toml",}</tool_call>
```

the proxy can return:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "read_file",
              "arguments": "{\"path\":\"Cargo.toml\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

## Working with traces and datasets

### Export a dataset

```sh
nix develop --command cargo run -- \
  export-dataset \
  --trace-path traces/model-correction-proxy.jsonl \
  --output-path datasets/corrections.jsonl
```

Dataset rows are designed for correction-agent and prompt-optimization workflows. They include available tools, recent messages, malformed response text, parser events, expected repair, and repair actions.

The main dataset exporter currently reads the request trace file. Use `--model`, `--profile`, `--failure-kind`, `--repair-action`, and `--correction-result` to filter exports for model/profile-specific GEPA datasets. The correction-agent sidecar trace is useful for copying focused correction failures into datasets and for comparing raw correction-agent output against the accepted final repair.

Request-adapter prompt datasets are kept separate from correction-agent datasets. For example, `datasets/request-adapter/gemma-dsrs-conservative.jsonl` captures a Gemma empty-output trace as a prompt-adapter optimization seed rather than blaming the correction agent.

### Replay traces

```sh
nix develop --command cargo run -- \
  replay \
  --trace-path traces/model-correction-proxy.jsonl
```

Replay re-runs recorded upstream responses through the current interpreter/repair pipeline and reports mismatches. Generated IDs are ignored during semantic comparison.

### Run regression evaluation

```sh
nix develop --command cargo run -- \
  eval \
  --suite-path eval/regressions.jsonl
```

Regression suites are JSONL. Each line is a case:

```json
{
  "name": "xml tool repair",
  "request": {
    "model": "qwen/qwen3.5-9b",
    "messages": [{ "role": "user", "content": "Read Cargo.toml" }],
    "parallel_tool_calls": false,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "parameters": {
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
          }
        }
      }
    ]
  },
  "upstream_response": {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "qwen/qwen3.5-9b",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "<tool_call name=\"read_file\">{pth:\"Cargo.toml\"}</tool_call>"
        },
        "finish_reason": "stop"
      }
    ]
  },
  "expected_tool_calls": [
    {
      "name": "read_file",
      "arguments": { "path": "Cargo.toml" }
    }
  ]
}
```

The evaluator reports:

- total passed/failed
- tool-call recovery rate
- schema-valid final response rate
- deterministic repair count
- correction-agent usage count
- profile-level pass counts

## DSRs and GEPA optimization

The project uses [`dspy-rs`](https://crates.io/crates/dspy-rs), the Rust DSPy-style framework, for internal correction-agent and GEPA optimization work.

After exporting a dataset, run:

```sh
export OPENROUTER_API_KEY="..."

nix develop --command cargo run -- \
  optimize-prompts \
  --dataset-path datasets/corrections.jsonl \
  --output-path datasets/gepa-correction-prompt.json \
  --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen3-8b \
  --iterations 3 \
  --max-examples 12
```

This loads dataset rows as typed DSRs examples, runs GEPA against the correction-prompt program, and writes a report with the best discovered instruction, artifact metadata, and optimization statistics. The command also accepts `--target-model`, `--profile`, and `--artifact-id` so artifacts can be tied back to a runtime profile.

The optimized correction prompt report can be loaded by a profile using `correction_agent_artifact`.

For request-adapter profile guidance, use a separate dataset and optimizer:

```sh
nix develop --command cargo run -- \
  optimize-request-adapter-prompt \
  --dataset-path datasets/request-adapter/gemma-dsrs-conservative.jsonl \
  --output-path datasets/request-adapter/gemma-dsrs-conservative-gepa.json \
  --base-url https://openrouter.ai/api/v1 \
  --model google/gemma-4-26b-a4b-it \
  --target-model google/gemma-4-26b-a4b-it \
  --profile gemma-dsrs-conservative \
  --artifact-id request-adapter/gemma-dsrs-conservative \
  --iterations 3 \
  --max-examples 2
```

This writes a `request_adapter_instruction` artifact that can be loaded through `request_adapter_artifact`. See `configs/gemma-dsrs-conservative.toml` for the Gemma profile wired to the optimized artifact.

After inspecting a GEPA artifact, promote it into a model-profile config explicitly:

```sh
nix develop --command cargo run -- \
  promote-artifact \
  --config-path configs/gemma-dsrs-conservative.toml \
  --artifact-path datasets/request-adapter/gemma-dsrs-conservative-gepa.json
```

The promotion command reads the artifact metadata, validates the artifact layer, updates either `request_adapter_artifact` or `correction_agent_artifact`, preserves existing model patterns unless `--model-pattern` is provided, and bumps the profile revision. Use `--dry-run` to print the promotion report without writing the config.

## Local mock upstream

For manual proxy tests without a real model provider:

```sh
nix develop --command cargo run --example mock_upstream
```

Override the mock response:

```sh
MOCK_UPSTREAM_BIND=127.0.0.1:18081 \
MOCK_UPSTREAM_CONTENT='<tool_call name="read_file">{pth:"Cargo.toml"}</tool_call>' \
nix develop --command cargo run --example mock_upstream
```

Then run the proxy against it:

```sh
nix develop --command cargo run -- \
  --bind 127.0.0.1:18080 \
  --upstream-base-url http://127.0.0.1:18081/v1 \
  --trace-path /tmp/model-correction-proxy-trace.jsonl \
  serve
```

## Development

### Recommended workflow

```sh
nix develop
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
nix flake check
```

One-shot form:

```sh
nix develop --command cargo fmt --check
nix develop --command cargo clippy --all-targets --all-features -- -D warnings
nix develop --command cargo test --all-targets --all-features
nix flake check
```

### Project layout

```text
.
├── Cargo.toml
├── flake.nix
├── brainstorming-guidelines.md
├── intent.md
├── docs/
│   └── model-configurability.md
├── examples/
│   └── mock_upstream.rs
└── src/
    ├── agents.rs
    ├── bin/proxy.rs
    ├── config.rs
    ├── dataset.rs
    ├── eval.rs
    ├── gateway.rs
    ├── lib.rs
    ├── model_profile.rs
    ├── normalizer.rs
    ├── openai.rs
    ├── optimization.rs
    ├── policy.rs
    ├── promotion.rs
    ├── prompt_adapter.rs
    ├── repair.rs
    ├── replay.rs
    ├── response_interpreter.rs
    ├── trace.rs
    └── upstream.rs
```

### Adding a repair

1. Add detection in `src/response_interpreter.rs` if the output shape is new.
2. Add deterministic or schema-guided repair in `src/repair.rs`.
3. Record a clear `RepairAction` with confidence and reason.
4. Add unit tests and, when possible, a regression case.
5. Run replay against existing traces to catch behavior changes.

### Adding a model profile

Prefer adding or overriding profiles through `--config` / `MCP_CONFIG_PATH`. Built-in Rust profiles are still useful for defaults and common families.

For a config profile, decide:

1. `name`, `revision`, and `model_patterns`
2. `tool_mode` and `tool_format`
3. whether to load a `request_adapter_artifact`
4. whether to load a `correction_agent_artifact`
5. any model-specific correction or judge model overrides

For a built-in profile:

1. Add a constructor in `src/model_profile.rs`.
2. Add it to `builtin_profiles()`.
3. Decide:
   - `ToolFormat::Dsrs` for the default path, or `ToolFormat::Xml` / `ToolFormat::TaggedJson` for an explicit experiment
   - whether native tool calling should be avoided
   - whether parallel tool calls are expected to work well
   - any model-specific correction or judge model overrides
4. Add profile resolution tests.

### Adding evaluation coverage

Prefer regression cases based on real traces. A good case should include:

- the original request and tools
- the raw upstream response
- expected final tool calls or content
- the model/profile being evaluated
- the failure mode being protected

## Safety and correctness principles

- Preserve valid upstream output.
- Repair syntax before calling a correction model.
- Repair schema shape only when intent is clear.
- Avoid fabricating missing required values by default.
- Enforce client-provided parallel-tool settings.
- Keep corrections auditable in traces.
- Treat traces as sensitive application data.
- Prefer model-specific profiles over scattered one-off hacks.

## Current limitations

- Upstream token streaming is not implemented. Upstream responses are buffered, then optionally returned to streaming clients as corrected SSE.
- API coverage is intentionally small: `/health`, `/v1/models`, and `/v1/chat/completions`.
- Runtime configuration now supports config files and prompt artifacts, but the parser/repair policy schema is still a first pass and not every future per-model knob is exposed yet.
- Retry/continue policy is represented in configuration but not implemented as an upstream retry loop yet.
- GEPA optimization now covers the correction agent and request-adapter profile guidance, and artifacts can be promoted into profile config with `promote-artifact`; automated dataset curation is still manual.
- Provider-specific adapters beyond generic OpenAI-compatible HTTP are not implemented yet.

## Roadmap

Near-term directions:

- richer config-file schema for parser strictness, repair policy, and provider metadata
- profile validation and migration tooling
- model/profile-specific GEPA artifact promotion in CI or release workflows
- richer correction-agent/judge routing
- replay-driven regression suites from real traces
- richer request-adapter GEPA datasets across more model families
- provider-specific compatibility adapters
- richer streaming fidelity for corrected responses
- debug endpoints or a trace inspection UI

Long-term, this project should become a compatibility runtime that learns how each model prefers to be prompted, how it tends to fail, and how best to recover without requiring client applications to change.
