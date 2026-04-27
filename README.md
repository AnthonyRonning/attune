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

Implemented scope currently covers the first three phases:

- **Phase 1:** non-streaming OpenAI-compatible chat proxy, request normalization, upstream OpenAI-compatible calls, deterministic repair, DSRs correction agent path, JSONL traces
- **Phase 2:** proxy-owned XML/tagged tool rendering, model profiles, multi-tool parsing, parallel-tool enforcement, schema-guided repair
- **Phase 3:** trace-to-dataset export, replay harness, regression evaluation, DSRs/GEPA prompt optimization scaffolding

The proxy is intentionally non-streaming right now. It buffers the upstream response so it can interpret and repair the complete output before returning a client-compatible response.

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
| Gateway | `src/gateway.rs` | Axum HTTP server exposing `/health` and `/v1/chat/completions` |
| OpenAI types | `src/openai.rs` | OpenAI-compatible request/response structures |
| Normalizer | `src/normalizer.rs` | Converts incoming requests into internal `NormalizedRequest` |
| Model profiles | `src/model_profile.rs` | Chooses model-specific tool rendering and repair defaults |
| Prompt adapter | `src/prompt_adapter.rs` | Converts native tools into proxy-owned XML/tagged tool instructions |
| Upstream client | `src/upstream.rs` | Calls generic OpenAI-compatible `/chat/completions` APIs |
| Response interpreter | `src/response_interpreter.rs` | Detects native, XML, tagged JSON, markdown JSON, direct JSON, and function-like tool intent |
| Repair engine | `src/repair.rs` | Repairs tool names, JSON arguments, schema key typos, scalar/array shape issues, and parallel-tool violations |
| Correction agents | `src/agents.rs` | DSRs-powered correction-agent path for suspicious malformed responses |
| Traces | `src/trace.rs` | Local JSONL trace logging for audit and dataset generation |
| Dataset export | `src/dataset.rs` | Converts traces into correction dataset rows |
| Replay | `src/replay.rs` | Replays recorded traces through the repair pipeline |
| Evaluation | `src/eval.rs` | Runs regression suites and reports correction metrics |
| Optimization | `src/optimization.rs` | DSRs/GEPA prompt optimization scaffolding |

## Tool-calling model

The proxy supports two early modes:

1. **Proxy-owned tool rendering**
   - The proxy consumes OpenAI-style `tools`.
   - It removes native upstream tool definitions.
   - It renders available tools into text instructions suited to the selected model profile.
   - It parses the model's text back into OpenAI `tool_calls`.

2. **Pass-through repair**
   - The proxy can preserve native upstream tool calling behavior.
   - It still interprets and repairs malformed or misplaced outputs after the upstream response.

Proxy-owned rendering is the primary reliability path because it avoids relying on brittle inference-engine tool parsers.

## Built-in model profiles

Built-in profiles are selected by model-name substring:

| Profile | Matching models | Tool format |
| --- | --- | --- |
| `qwen-xml` | `qwen`, `qwq` | XML |
| `kimi-xml` | `kimi`, `moonshot` | XML |
| `glm-xml` | `glm`, `z-ai` | XML |
| `llama-tagged-json` | `llama` | tagged JSON |
| `gemma-xml-conservative` | `gemma` | XML, conservative parallel-tool defaults |
| `balanced-default` | fallback | XML |

Profiles are currently code-level configuration. A config-file loader is not implemented yet.

## What the repair engine handles

Current deterministic and schema-guided repairs include:

- tool calls embedded in assistant `content`
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

## Traceability

Every completed request can produce a JSONL trace containing:

- original request
- normalized request
- selected model profile
- adapted upstream request
- raw upstream response
- interpreted response
- repair actions and confidence metadata
- final response
- errors, if any

Responses include an `x-model-correction-trace-id` header when a trace is written.

Trace files can contain prompts, model outputs, and tool arguments. Authorization, cookie, and key-like headers are redacted, but trace storage should still be treated as sensitive application data.

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
- `POST /v1/chat/completions`

### Runtime options

| Option | Env var | Default |
| --- | --- | --- |
| `--bind` | `MCP_BIND_ADDR` | `127.0.0.1:8080` |
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

This loads dataset rows as DSRs examples, runs GEPA against the correction-prompt program, and writes a report with the best discovered instruction and optimization statistics.

The optimized prompt report is not hot-loaded by the live proxy yet. It is currently an optimization artifact that can be used to update the correction signature or future profile/config wiring.

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

1. Add a constructor in `src/model_profile.rs`.
2. Add it to `builtin_profiles()`.
3. Decide:
   - `ToolFormat::Xml` or `ToolFormat::TaggedJson`
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

- Streaming is not implemented; streaming requests are rejected.
- Only `/health` and `/v1/chat/completions` are exposed.
- Runtime configuration is mostly CLI/env plus code-level `ProxyConfig`; no config-file loader yet.
- Retry/continue policy is represented in configuration but not implemented as an upstream retry loop yet.
- Optimized GEPA prompts are written as artifacts but not automatically hot-loaded.
- Provider-specific adapters beyond generic OpenAI-compatible HTTP are not implemented yet.

## Roadmap

Near-term directions:

- config-file loading for profiles and policy modes
- richer correction-agent/judge routing
- replay-driven regression suites from real traces
- loading optimized GEPA prompts into runtime profiles
- provider-specific compatibility adapters
- buffered/corrected streaming
- debug endpoints or a trace inspection UI

Long-term, this project should become a compatibility runtime that learns how each model prefers to be prompted, how it tends to fail, and how best to recover without requiring client applications to change.
