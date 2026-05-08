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

| Profile | Matching models | Tool format | DSRs history |
| --- | --- | --- | --- |
| `qwen-dsrs` | `qwen`, `qwq` | DSRs | Append-only |
| `kimi-dsrs` | `kimi`, `moonshot` | DSRs with compact-output guidance | Append-only |
| `glm-dsrs` | `glm`, `z-ai` | DSRs | Append-only |
| `llama-dsrs` | `llama` | DSRs | Append-only |
| `gemma-dsrs-conservative` | `gemma` | DSRs, conservative parallel-tool defaults | Append-only |
| `balanced-default` | fallback | DSRs | Append-only |

Profiles can be provided in TOML, JSON, or JSON5 through `--config` or `MCP_CONFIG_PATH`. Configured profiles are matched before built-ins, so users can tune or replace a built-in profile without recompiling. Unknown models still fall back to `balanced-default`.

Profiles support revision/source metadata plus request-adapter and correction-agent artifact IDs. When `request_adapter_artifact` or `correction_agent_artifact` points at a JSON GEPA report, text prompt file, or `builtin:<artifact-id>` reference, the proxy loads the instruction and records the artifact IDs in traces. See [`docs/model-configurability.md`](docs/model-configurability.md).

DSRs profiles also support `dsrs_history_format`. The default is `append_only`, which keeps user turns as chat messages and renders prior assistant turns in the same DSRs `content` / `tool_calls` shape expected for the next answer. The previous single regenerated transcript format is still available as `regenerated_context` for models that perform better with one serialized conversation block.

The DSRs response contract intentionally mirrors OpenAI-compatible assistant messages: `content` only is valid, `tool_calls` only is valid, and `content` plus `tool_calls` is valid. Empty content plus empty `tool_calls` remains a failure for a real assistant turn.

Built-in defaults are generated from [`profiles/builtin-defaults.toml`](profiles/builtin-defaults.toml) at build time. Reviewed artifacts listed there are validated by `build.rs` and embedded into the binary, so shipped binaries do not need local dataset files for their default profile prompts. Runtime config profiles still match before built-ins and can override the embedded defaults.

The current Gemma built-in default embeds `datasets/request-adapter/gemma-dsrs-conservative-r4-append-only-gepa.json` as `builtin:request-adapter/gemma-dsrs-conservative/r4-append-only-content-tools-json-meta` at profile revision 5. [`configs/gemma-dsrs-conservative.toml`](configs/gemma-dsrs-conservative.toml) remains a filesystem-artifact example and local override path, but plain `cargo run` now gets the same reviewed Gemma request-adapter instruction through the embedded built-in default.

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
  --config configs/gemma-dsrs-conservative.toml \
  --bind 127.0.0.1:8080 \
  --upstream-base-url https://openrouter.ai/api/v1 \
  --trace-path traces/model-correction-proxy.jsonl \
  serve
```

Drop `--config configs/gemma-dsrs-conservative.toml` when you want only embedded built-in defaults. Keep it when testing config-file overrides or filesystem artifact loading.

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

## Command Reference

Run any command through Nix as `nix develop --command cargo run -- <command> ...`. Use `cargo run -- <command> --help` for the full flag list.

| Command | Purpose |
| --- | --- |
| `serve` | Start the OpenAI-compatible proxy. This is also the default when no subcommand is passed. |
| `inspect-traces` | Print compact summaries from the request trace JSONL, optionally as pretty JSON with `--json`. |
| `export-dataset` | Export correction-agent training rows from request traces. Use this for malformed-response correction datasets. |
| `export-request-adapter-dataset` | Export trace-faithful request-adapter rows from exact trace IDs or filters. This preserves the original OpenAI request and adds explicit expected-output labels. |
| `replay` | Re-run recorded upstream responses through the current interpreter and repair pipeline. |
| `eval` | Run JSONL regression suites and report pass rates by model/profile/revision/artifact. |
| `optimize-prompts` | Run GEPA for the correction-agent DSRs prompt using a correction dataset. |
| `optimize-request-adapter-prompt` | Run GEPA for request-adapter profile guidance using the runtime DSRs formatter and selected `dsrs_history_format`. |
| `promote-artifact` | Deliberately promote a reviewed GEPA artifact into a model-profile config and bump the profile revision. |
| `promote-default-artifact` | Deliberately promote a reviewed GEPA artifact into `profiles/builtin-defaults.toml` so it is embedded into shipped binaries. |
| `trace-harness` | Import third-party harness traces into neutral scenarios, inspect them, and run sampled live structural checks through the proxy. |

The `trace-harness` subcommands currently include `import-pi`, `import-hermes-rows`, `inspect`, and `run`. Use `import-*` commands to build local scenario JSONL from downloaded datasets, `inspect` to review the scenario shape before spending API calls, and `run` to sample live proxy/model behavior.

The normal reliability loop is:

1. Use `inspect-traces` to find a failing trace ID.
2. Use `export-request-adapter-dataset` or `export-dataset` to create the right labeled dataset row.
3. Run the relevant GEPA optimizer.
4. Inspect the generated artifact.
5. Use `promote-artifact --dry-run`, then promote to config only if the artifact is worth adopting.
6. After config-level testing, use `promote-default-artifact --dry-run`, then promote to built-in defaults only if the artifact should ship.

For broader live checks that are not tied to a local failing trace, use the trace harness:

```sh
nix develop --command cargo run -- \
  trace-harness import-pi \
  --input-path eval/trace-harness/raw/pi-mono \
  --output-path eval/trace-harness/scenarios/pi-mono.local.jsonl \
  --max-scenarios 12

nix develop --command cargo run -- \
  --config configs/gemma-dsrs-conservative.toml \
  trace-harness run \
  --scenarios-path eval/trace-harness/scenarios/pi-mono.local.jsonl \
  --output-path eval/trace-harness/results/gemma-pi.local.json \
  --model google/gemma-4-26b-a4b-it \
  --limit 12
```

Trace-harness scenarios preserve the source conversation prefix and tool definitions, then ask the live proxy/model for the next assistant turn. They do not execute source harness tools and they do not grade whether the model made the best engineering choice. They only check whether the final proxy response stays structurally usable for an OpenAI-compatible agent loop. See [`eval/trace-harness/README.md`](eval/trace-harness/README.md).

The harness writes downloaded source traces, converted scenarios, and run reports under `eval/trace-harness/raw/`, `eval/trace-harness/scenarios/`, and `eval/trace-harness/results/`. Those local files are ignored by default. Reviewed edge cases can be copied into request-adapter datasets only after inspection.

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

Request-adapter prompt datasets are kept separate from correction-agent datasets. For example, `datasets/request-adapter/gemma-dsrs-conservative-trace-faithful.jsonl` contains exact labeled Gemma request traces for prompt-adapter optimization rather than blaming the correction agent.

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

Both GEPA commands expose `--lm-max-tokens`, defaulting to `100000`. This is separate from the live proxy request path: client `max_tokens` values are still passed through only when provided. The GEPA runner sets a high optimizer token budget because `dspy-rs` sends an explicit `max_tokens` value for its own optimizer, reflection, and target-model calls, and truncated optimizer instructions are not useful artifacts.

Request-adapter GEPA uses a local JSON adapter for GEPA's own reflection/proposal calls instead of the default DSRs chat adapter. The runtime candidate still renders through the selected proxy `dsrs_history_format`; the JSON adapter only prevents GEPA's outer meta-parser from truncating optimized instructions that legitimately contain literal `[[ ## content ## ]]`, `[[ ## tool_calls ## ]]`, or `[[ ## completed ## ]]` text.

The optimized correction prompt report can be loaded by a profile using `correction_agent_artifact`.

For request-adapter profile guidance, use a separate dataset and optimizer:

```sh
nix develop --command cargo run -- \
  export-request-adapter-dataset \
  --trace-path traces/model-correction-proxy.jsonl \
  --output-path datasets/request-adapter/gemma-dsrs-conservative.jsonl \
  --trace-id trace_bb26a5f316ca486796231bfd184ac723 \
  --expected-output-json '{"content":"","tool_calls":[{"name":"read","arguments":{"path":"/Users/tony/Dev/ThirdParties/pi-mono/packages/coding-agent/docs/packages.md"}}]}' \
  --observed-failure-kind duplicate_immediate_tool_call \
  --observed-problem "model repeated the same deterministic package-listing command" \
  --prompt-goal "read the package docs instead of repeating the listing"
```

This preserves the exact original OpenAI request from the trace in the dataset row, including `messages`, `tools`, `tool_choice`, `parallel_tool_calls`, profile metadata, observed upstream output, and adapted request metadata. The expected output is an explicit label. For successful traces, `--use-final-response` can label from the final response; for triage-only rows, use `--allow-unlabeled` and add labels before running GEPA.

```sh
nix develop --command cargo run -- \
  optimize-request-adapter-prompt \
  --dataset-path datasets/request-adapter/gemma-dsrs-conservative.jsonl \
  --output-path datasets/request-adapter/gemma-dsrs-conservative-draft-gepa.json \
  --base-url https://openrouter.ai/api/v1 \
  --model google/gemma-4-26b-a4b-it \
  --target-model google/gemma-4-26b-a4b-it \
  --profile gemma-dsrs-conservative \
  --profile-revision 3 \
  --dsrs-history-format append_only \
  --artifact-id request-adapter/gemma-dsrs-conservative/append-only \
  --iterations 3 \
  --max-examples 3 \
  --lm-max-tokens 100000
```

This optimizer uses the same runtime DSRs formatter as the proxy. GEPA mutates the profile guidance, the runner installs that candidate guidance into a model profile, renders the request through the selected `dsrs_history_format`, calls the target model, parses the DSRs response, and scores the result against the request-adapter dataset. Exact labeled tool calls or content score highest, but structurally valid different tool calls and real non-placeholder content receive strong partial credit so the optimizer does not overfit to one trace's arbitrary next action. To compare formatter behavior, run the same dataset twice with different `--dsrs-history-format` values and separate artifact IDs.

This writes a `request_adapter_instruction` artifact that records the target profile, profile revision, and history format. It can be loaded through `request_adapter_artifact`; promotion can also carry the artifact's `dsrs_history_format` into the profile config. See `configs/gemma-dsrs-conservative.toml` for a Gemma profile wired to an optimized artifact.

GEPA comparison artifacts are treated as disposable until promoted. Files such as `*-append-only-gepa.json`, `*-regenerated-context-gepa.json`, and `*-trace-faithful-*-gepa.json` are ignored by default; keep or force-add only artifacts that have been reviewed and intentionally promoted.

`datasets/request-adapter/gemma-dsrs-conservative-trace-harness-curated.jsonl` contains a small reviewed set from live trace-harness runs. It is intentionally narrow: exact tool-shape positives plus correction-needed request-adapter failures where the final clean output is a clear label.

The current reviewed Gemma request-adapter GEPA run combines the hand-labeled profile dataset, exact trace-faithful exports, and curated trace-harness examples:

```sh
jq -c . \
  datasets/request-adapter/gemma-dsrs-conservative.jsonl \
  datasets/request-adapter/gemma-dsrs-conservative-trace-faithful.jsonl \
  datasets/request-adapter/gemma-dsrs-conservative-trace-harness-curated.jsonl \
  > /tmp/gemma-request-adapter-all.jsonl
```

The latest full matrix used 14 reviewed rows and tested both DSRs history formats:

| Target model | History format | Score | Result |
| --- | --- | ---: | --- |
| `google/gemma-4-26b-a4b-it` | `append_only` | `0.8714286` | promoted |
| `google/gemma-4-26b-a4b-it` | `regenerated_context` | `0.8000001` | kept as an experiment only |
| `qwen/qwen3.5-9b` | `append_only` | `0.7000001` | not promoted; showed Pi/path-specific overfit |
| `qwen/qwen3.5-9b` | `regenerated_context` | `0.6642858` | not promoted |

The promoted Gemma artifact is `datasets/request-adapter/gemma-dsrs-conservative-r4-append-only-gepa.json`. [`configs/gemma-dsrs-conservative.toml`](configs/gemma-dsrs-conservative.toml) references it at profile revision 5, and [`profiles/builtin-defaults.toml`](profiles/builtin-defaults.toml) embeds it into the shipped Gemma built-in default. The r4 GEPA rerun used the revised content-plus-tools contract and kept the r3 instruction because it remained the best candidate. Higher GEPA scores are not enough by themselves; artifacts still need human review for overfit, brittle one-off rules, and prompt drift before promotion.

After inspecting a GEPA artifact, first promote it into a model-profile config explicitly:

```sh
nix develop --command cargo run -- \
  promote-artifact \
  --config-path configs/gemma-dsrs-conservative.toml \
  --artifact-path datasets/request-adapter/gemma-dsrs-conservative-r4-append-only-gepa.json \
  --profile gemma-dsrs-conservative
```

The promotion command reads the artifact metadata, validates the artifact layer, updates either `request_adapter_artifact` or `correction_agent_artifact`, preserves existing model patterns unless `--model-pattern` is provided, and bumps the profile revision. Use `--dry-run` to print the promotion report without writing the config.

After the config path has been tested, promote the same artifact into built-in defaults if it should ship inside the binary:

```sh
nix develop --command cargo run -- \
  promote-default-artifact \
  --artifact-path datasets/request-adapter/gemma-dsrs-conservative-r4-append-only-gepa.json \
  --profile gemma-dsrs-conservative \
  --model-pattern gemma
```

`promote-default-artifact` updates `profiles/builtin-defaults.toml`. The next build validates the manifest and embeds the artifact content with `include_str!`. See [`profiles/README.md`](profiles/README.md) for the built-in default promotion rules.

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
├── build.rs
├── flake.nix
├── brainstorming-guidelines.md
├── intent.md
├── docs/
│   └── model-configurability.md
├── profiles/
│   ├── README.md
│   └── builtin-defaults.toml
├── configs/
│   └── gemma-dsrs-conservative.toml
├── datasets/
│   └── request-adapter/
├── eval/
│   └── trace-harness/
│       └── README.md
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
3. `dsrs_history_format`, usually `append_only` or `regenerated_context`
4. whether to load a `request_adapter_artifact`
5. whether to load a `correction_agent_artifact`
6. any model-specific correction or judge model overrides

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
- Built-in default artifacts are embedded at build time, but promotion is still a manual review step and currently covers only the reviewed Gemma request-adapter artifact.
- Retry/continue policy is represented in configuration but not implemented as an upstream retry loop yet.
- GEPA optimization now covers the correction agent and request-adapter profile guidance, and artifacts can be promoted into profile config with `promote-artifact` or shipped defaults with `promote-default-artifact`; label quality and dataset curation are still deliberate review steps.
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
