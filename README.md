# Attune

> **MVP under active construction.** Attune is usable for local experiments,
> trace-driven development, and early model-profile work, but the public API,
> configuration surface, datasets, evals, and built-in defaults are still
> moving. Current comparison numbers are directional engineering evidence, not
> final benchmark claims.

Attune is an **Agent Contract Runtime** for reliable tool use across
OpenAI-compatible models.

The name reflects the runtime loop: **A**daptive **T**ool-use
**T**ranslation, **U**nderstanding, **N**ormalization, and **E**valuation.

Attune sits between an existing OpenAI-compatible client and an upstream model
provider. It makes tool use an explicit, inspectable, model-specific contract
instead of assuming the provider's native tool parser, chat template, or
inference stack will preserve the model's intent.

![Attune architecture flow](docs/assets/attune-flow.png)

The first target is agent tool-calling reliability for open-source and
OpenAI-compatible models. When a model clearly intends to call a tool but the
provider returns plain text, malformed JSON, an empty stop, reasoning-only
output, or a non-OpenAI tool shape, Attune can translate, interpret, repair,
trace, and evaluate that turn before the client application loop breaks.

Attune is not a smarter coding agent, and it does not try to judge whether a
model made the best engineering decision. It focuses on the contract boundary:
given the model's agency, keep the assistant turn structurally usable by an
OpenAI-compatible agent harness.

The thesis is simple: many models already know what action they want to take,
but the model/provider/client boundary often fails to carry that action across.
Attune makes that boundary explicit, observable, repairable, and learnable.

## The Problem

Many open-source models are much more capable than their integration
experience suggests. A common failure mode is not that the model cannot use a
tool. It is that the surrounding API, chat template, provider, or inference
engine fails to preserve the model's intended tool call.

Typical symptoms:

- the model says "I'll read the file now" and stops without a tool call
- malformed JSON causes native tool parsing to fail
- a model emits a clear tool call in text, but the client expects OpenAI
  `tool_calls`
- reasoning/thinking fields contain text the client expects in `content`
- a provider reports `finish_reason = stop` even though the response looks like
  an incomplete or misplaced tool call

Most agent harnesses only see one thing: an OpenAI-compatible assistant
message. If that message has neither usable `content` nor valid `tool_calls`,
the loop stops, repeats itself, or loses the model's intended action.

## What Makes Attune Different

Attune is deliberately more than a proxy with a few parser fallbacks:

- **Proxy-owned contracts:** Attune can strip native upstream tools, render a
  DSRs-style contract, and parse model text back into OpenAI `tool_calls`.
- **Model profiles:** Qwen, Gemma, Kimi, GLM, Llama, and unknown fallback
  models can use different history renderers, tool guidance, correction
  settings, provider routing, and optimized prompt artifacts.
- **Conversation-aware formatting:** Prior assistant tool calls and tool
  results are reformatted into the selected contract. Attune does not only
  rewrite the latest prompt; it keeps the full conversation history aligned
  with the format the model is expected to use now.
- **Layered recovery:** The runtime separates deterministic interpretation,
  schema repair, policy decisions, and a typed DSRs correction-agent fallback
  instead of mixing one-off fixes into one parser. Deterministic repair handles
  clear syntax/schema problems; ambiguous or semantic recovery goes through a
  typed correction agent.
- **Trace-to-dataset loop:** Traces are rich enough to inspect failures after
  the fact, export exact request-adapter or correction-agent dataset rows,
  replay regressions, and run GEPA prompt optimization. Every meaningful
  failure should be able to become a regression case or training row.
- **Explicit promotion:** Optimized artifacts are reviewed, promoted into
  profile config, and then promoted into embedded built-in defaults only when
  they should ship with the binary.

The design goal is that any OpenAI-compatible app can point at Attune, choose
any model string, and get the best known contract adapter for that model.
Unknown models receive shared defaults; known model families can accumulate
their own traces, datasets, GEPA artifacts, and profile revisions over time.

## MVP Status

The original planning docs described this work in phases. The public MVP now
covers the core runtime, model-profile, tracing, dataset, evaluation, GEPA, and
artifact-promotion loop. The remaining work is hardening, broadening model
coverage, and improving release ergonomics.

Implemented now:

- OpenAI-compatible `/v1/chat/completions`, `/v1/models`, and `/health`
- request normalization, model profile selection, and generic
  OpenAI-compatible upstream calls
- proxy-owned DSRs tool rendering, including profile-selectable conversation
  history formats
- native, DSRs, XML, tagged JSON, markdown JSON, direct JSON, and
  function-like tool-intent interpretation
- deterministic syntax/schema repair, parallel-tool enforcement, and a typed
  DSRs correction-agent fallback
- JSONL request traces and correction-agent sidecar traces
- trace inspection, replay, regression evaluation, and trace-harness comparison
  against direct baseline
- request-adapter and correction-agent dataset export
- GEPA optimization for request-adapter profile guidance and correction-agent
  prompts
- explicit artifact promotion into config, then embedded built-in defaults for
  shipped binaries
- reviewed built-in request-adapter defaults for Qwen, Gemma, Kimi, and GLM
  profiles

Attune intentionally buffers upstream chat completions so it can interpret and
repair the complete output before returning a client-compatible response. If
the client requests `stream: true`, it returns a corrected SSE response after
buffering and repair; it does not yet proxy upstream tokens incrementally.

## Early Evidence

The strongest fresh comparison result so far is the reviewed 500-scenario
Pi/Hermes trace-harness run against `google/gemma-4-26b-a4b-it` on OpenRouter:

| Path | Structural passes |
| --- | ---: |
| Direct OpenRouter baseline | 475/500 |
| Attune proxy, previous promoted default | 497/500 |
| Attune proxy, Sonnet-5 append-only default | 498/500 |

Attune fixed all 25 direct baseline structural failures in the original run.
The Sonnet-5 append-only follow-up improved the proxy replay by one more case.
Its remaining two final failures were reviewed; representative direct success
and recovered-failure traces were added to the curated GEPA datasets.

Qwen is still the highest-priority in-progress profile, but the latest GEPA
promotions closed a meaningful part of the gap. The original 500-scenario
Pi/Hermes comparison found that direct OpenRouter passed 487/500 structural
checks while the previous Attune proxy profile passed 457/500. After promoting
the Qwen-500 regenerated-context artifact and then the Sonnet-5 regenerated
context artifact, the same 500-scenario file was replayed through the proxy
without rerunning the direct baseline:

| Qwen 3.5 9B 500-scenario run | Structural passes |
| --- | ---: |
| Direct OpenRouter baseline from prior comparison | 487/500 |
| Previous Attune proxy profile | 457/500 |
| Promoted Qwen-500 Attune proxy profile | 471/500 |
| Sonnet-5 Qwen append-only profile, not promoted | 469/500 |
| Promoted Sonnet-5 Qwen regenerated-context profile | 475/500 |

That is an 18-case proxy improvement over the original proxy result on the same
scenario set, and a 4-case improvement over the previous promoted Qwen default.
It is still not better than the prior direct baseline. In the promoted replay,
valid DSR `tool_calls` converted into OpenAI `tool_calls` count as expected
adapter behavior, not correction-agent repair; the remaining failures are
mostly tool intent leaking as prose or malformed tool text instead of clean DSRs
output, plus correction-agent failures where repaired DSRs `tool_calls` JSON
was still invalid.

Representative Qwen traces from this replay were promoted into both the Qwen
request-adapter dataset and the shared correction-agent dataset. These examples
cover inspected direct successes, recovered malformed DSRs, recovered
content-only answers, correction-agent tool recoveries, and correction-agent
failure states where a valid repair was still available. Future GEPA runs can
learn both to avoid the correction path and to repair it more reliably when
needed.

The latest large-model baseline check used the same 500-scenario Pi/Hermes
trace-harness file against `moonshotai/kimi-k2.6` and `z-ai/glm-5.2` on
OpenRouter, ignoring Venice and allowing 420-second request timeouts:

| Run | Direct OpenRouter baseline | Attune proxy | Read |
| --- | ---: | ---: | --- |
| `moonshotai/kimi-k2.6`, Venice ignored | 240/500 | 500/500 | Direct responses often failed structurally; Attune repaired every final response in this run. |
| `moonshotai/kimi-k2.6`, Venice and WandB ignored | 333/500 | not measured | Removing WandB helped, but direct provider tool calling still failed 167/500 cases. |
| `moonshotai/kimi-k2.7-code`, Venice ignored | 489/500 | 497/500 | K2.7 Code routed entirely through Together and was much cleaner natively; unoptimized Attune still fixed all 11 direct failures but introduced 3 raw proxy failures. |
| `z-ai/glm-5.2`, Venice ignored | 500/500 | 473/500 | Native tool calling already passed this dataset; the current Attune profile regressed it. |

The Kimi direct failures were reviewed as real structural failures, dominated
by empty assistant responses, tool-like text without OpenAI `tool_calls`, and
premature tool-action prose without a tool call. Most failed direct Kimi calls
were routed through OpenRouter's WandB provider. A follow-up direct-baseline
rerun with both Venice and WandB ignored improved to 333/500, but did not make
native tool calling reliable: the remaining failures shifted mostly to invalid
or contaminated tool names on other providers, especially DeepInfra. That rerun
used a local mock proxy endpoint only to preserve the harness's authenticated
direct-baseline path without paying for a second proxy call, so only its direct
baseline column is meaningful. The Attune proxy result still shows the existing
DSRs adapter plus correction-agent path can recover these symptoms even before a
Kimi-specific GEPA artifact exists.

The K2.7 Code result is a different shape from K2.6: all direct baseline calls
were routed through Together, and native tool calling was already near-perfect
at 489/500. The current unoptimized Kimi DSRs profile still improved raw final
structural passes to 497/500 by fixing all 11 direct failures, but it needed
147 correction-agent attempts and introduced 3 raw proxy failures. One reviewed
proxy failure appears to be an eval heuristic false positive on a clarifying
answer. Of the other two, the unrecovered raw tool-call marker payload matches
Attune's parser/correction scope; the unavailable `grep` tool case was counted
by the harness but not promoted into GEPA curation because wrong tool selection
is outside the current prompt-format thesis.

The GLM result points in the opposite direction: direct native tool calling was
already structurally perfect on this eval. For GLM, Attune should either
preserve the native behavior or use a model-specific profile optimized for
non-regression. Valid DSR `tool_calls` adapted back into OpenAI `tool_calls`
remain expected adapter behavior in these reports, not correction-agent repair.

A follow-up Sonnet-5/OpenRouter append-only GEPA pass used small curated
trace-harness datasets for K2.6, K2.7 Code, and GLM 5.2. The resulting
request-adapter artifacts were promoted into model-specific built-in defaults.
For this layer, the primary metric is not only final proxy pass count; it is how
often the proxied request passes before invoking the correction agent. Valid DSR
tool-call adaptation remains expected adapter behavior, not correction-agent
repair.

| Model | Direct native | Unoptimized final proxy | Unoptimized no-correction pass | Promoted GEPA final proxy | Promoted GEPA no-correction pass | Correction path | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `moonshotai/kimi-k2.6` | 240/500 | 500/500 | 430/500 | 499/500 | 436/500 | 70 -> 64 | Request layer improved by 6 cases, while one final regression came from the correction path. |
| `moonshotai/kimi-k2.7-code` | 489/500 | 497/500 | 351/500 | 499/500 | 421/500 | 147 -> 79 | Request layer improved by 70 cases and correction load nearly halved. |
| `z-ai/glm-5.2` | 500/500 | 473/500 | 387/500 | 493/500 | 427/500 | 87 -> 72 | Request layer improved by 40 cases, but Attune still trails GLM's perfect native baseline on this eval. |

Artifact inspection found no embedded trace IDs, source IDs, or scenario IDs.
The warnings were mostly long-instruction and generic DSR-format wording. The
GLM artifact's exact-matching warning came from instructions to reproduce DSR
field markers exactly, not from copied benchmark labels. Representative trace
review found the expected split: clean DSR outputs adapted without correction,
native marker leaks recovered through the correction agent, and remaining
failures concentrated in malformed repair-output JSON, lost long-context task
summary for correction, or eval heuristics that needed narrowing.

The follow-up correction-agent GEPA pass used the same Sonnet-5/OpenRouter
reflection and judge setup, filtered the shared correction dataset by
model/profile, and added a correction-agent-only eval so candidates can be
tested without rerunning the full proxy suite. These correction artifacts are
not promoted. Under the stricter correction-only evaluator, the built-in
baseline stayed better or tied: K2.6 was 4/5 strict versus 3/5 for the optimized
artifact, K2.7 Code was 3/3 for both baseline and optimized artifact, and GLM
5.2 was 4/5 baseline versus 2/5 optimized. The failed candidates are retained
as experiment artifacts, not defaults.

## Try It

Start with embedded built-in profile defaults:

```sh
cp .env.example .env
export OPENROUTER_API_KEY="..."
nix develop --command cargo run --
```

Then point any OpenAI-compatible client at:

```text
http://127.0.0.1:8080/v1
```

Known Qwen and Gemma model strings use reviewed built-in request-adapter
defaults. Unknown models fall back to `balanced-default`, which still uses the
shared DSRs contract, trace, repair, and correction pipeline.

For full commands, examples, GEPA runs, artifact promotion, trace-harness
comparisons, and live OpenRouter smoke tests, see
[`docs/command-reference.md`](docs/command-reference.md).

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

Main source areas:

| Component | Source | Purpose |
| --- | --- | --- |
| Gateway | `src/gateway.rs` | Axum HTTP server exposing `/health`, `/v1/models`, and `/v1/chat/completions` |
| OpenAI types | `src/openai.rs` | OpenAI-compatible request/response structures |
| Model profiles | `src/model_profile.rs` | Model-specific tool rendering and repair defaults |
| Prompt adapter | `src/prompt_adapter.rs`, `src/dsrs_contract.rs` | Proxy-owned DSRs request rendering |
| Response interpreter | `src/response_interpreter.rs` | Detects native, DSRs, XML, JSON, and text tool intent |
| Repair engine | `src/repair.rs` | Repairs tool names, JSON arguments, schema shape, and policy violations |
| Correction agents | `src/agents.rs` | Typed DSRs correction-agent path for suspicious malformed responses |
| Traces/datasets/eval | `src/trace.rs`, `src/dataset.rs`, `src/replay.rs`, `src/eval.rs`, `src/trace_harness.rs` | Trace, replay, dataset, and evaluation workflows |
| Optimization/promotion | `src/optimization.rs`, `src/promotion.rs`, `build.rs` | GEPA artifacts and built-in default promotion |

## DSRs Request Transform

In proxy-owned DSRs mode, the client can send a normal OpenAI-compatible
request:

```jsonc
{
  "model": "qwen/qwen3.5-9b",
  "messages": [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Read Cargo.toml"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "read",
            "arguments": "{\"path\":\"Cargo.toml\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": "[package]\nname = \"attune\""
    },
    {"role": "user", "content": "What is this project?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read",
        "description": "Read file contents",
        "parameters": {
          "type": "object",
          "required": ["path"],
          "properties": {"path": {"type": "string"}}
        }
      }
    }
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": false
}
```

Attune selects a model profile, removes native upstream tool definitions for
proxy-owned profiles, and renders the request into a DSRs contract. In the
default `append_only` history format, the upstream model sees a runtime context
first, followed by append-only conversation messages:

```text
system:
  You are an OpenAI-compatible assistant behind Attune.
  Produce exactly these DSRs output fields:
  - content: plain user-facing text
  - tool_calls: JSON array of {"name": string, "arguments": object}
  Use [] when no tool call is needed.
  If parallel_tool_calls is false, emit at most one tool call.

user:
  [[ ## profile_guidance ## ]]
  <model/profile-specific guidance or promoted GEPA artifact>

  [[ ## system_context ## ]]
  role: system
  content:
  You are a coding agent.

  [[ ## conversation ## ]]
  Append-only conversation follows as chat messages.

  [[ ## available_tools ## ]]
  [
    {
      "type": "function",
      "function": {
        "name": "read",
        "description": "Read file contents",
        "parameters": {"type": "object", "...": "..."}
      }
    }
  ]

  [[ ## tool_choice ## ]]
  "auto"

  [[ ## parallel_tool_calls ## ]]
  false

  The remaining chat messages are append-only conversation history. Prior
  assistant messages use the same DSRs content/tool_calls output format you
  must use now. Tool-result messages may use a tool_result field marker and
  are observations, not an output field you should emit.

user:
  Read Cargo.toml

assistant:
  [[ ## content ## ]]

  [[ ## tool_calls ## ]]
  [
    {"name": "read", "arguments": {"path": "Cargo.toml"}}
  ]
  [[ ## completed ## ]]

user:
  [[ ## tool_result ## ]]
  tool_call_id: call_1
  content:
  [package]
  name = "attune"
  [[ ## completed ## ]]

user:
  What is this project?
```

The model must answer with the same output shape:

```text
[[ ## content ## ]]
This is a Rust proxy/runtime for adapting OpenAI-compatible tool use across
models.

[[ ## tool_calls ## ]]
[]

[[ ## completed ## ]]
```

Content-only, tool-calls-only, and content plus tool calls are all valid. Empty
content with empty `tool_calls` is not useful to an agent loop, so Attune treats
that as a structural failure and routes it through recovery.

`append_only` is the default because it preserves normal multi-turn chat shape
while making prior assistant/tool turns teach the current DSRs contract. Attune
also keeps a `regenerated_context` format for profiles that behave better when
the whole non-system conversation is serialized into one compact transcript.

## Core Concepts

**Proxy-owned DSRs contracts:** Attune can consume OpenAI-style `tools`, remove
native upstream tool definitions, render the request into a model-friendly DSRs
contract, then parse the model's text back into OpenAI `tool_calls`. Pass-through
repair remains available for compatibility.

**Model profiles:** Built-in profiles currently cover Qwen, Kimi, GLM, Llama,
Gemma, and a balanced fallback. Profiles can also be supplied through config
files and can load request-adapter or correction-agent artifacts. Built-in
defaults are generated from [`profiles/builtin-defaults.toml`](profiles/builtin-defaults.toml)
at build time and embedded into shipped binaries.

**Repair and correction:** Deterministic repair handles clear syntax/schema
issues such as malformed JSON, near-match tool names, simple schema key typos,
scalar/array shape issues, and misplaced tool-call text. Suspicious stops,
invalid DSRs `tool_calls`, empty outputs, reasoning-only assistant stops, and
semantic recovery cases can route through a typed DSRs correction agent.

**Traceability:** Request traces and correction-agent sidecar traces capture the
original request, selected profile, adapted upstream request, raw upstream
response, parser events, repair actions, correction attempts, and final
response. Traces can contain prompts, model outputs, file paths, and tool
arguments, so treat them as sensitive application data.

**GEPA optimization:** Correction-agent prompts and request-adapter profile
guidance are optimized separately. GEPA artifacts are not applied implicitly:
they are inspected, promoted into profile config, and then promoted into
embedded built-in defaults only when they should ship.

## Documentation

If the README hook is enough and you want the machinery, start here:

- [Command and workflow reference](docs/command-reference.md): how to run the
  proxy, inspect traces, export datasets, run GEPA, promote artifacts, compare
  against direct baseline, and reproduce the live workflows used so far.
- [Configuration reference](docs/config-reference.md): how runtime config,
  model profiles, provider routing, environment variables, and artifact
  references are resolved.
- [Model configurability](docs/model-configurability.md): the deeper
  architecture behind shared vs model-specific behavior, request-adapter GEPA,
  correction-agent GEPA, hidden labels, profile revisions, and promotion.
- [Polar and Attune](docs/polar-attune.md): how NVIDIA's Polar / ProRL Agent
  Server compares to Attune, and how trace-driven RL and DSRs/GEPA runtime
  reliability could combine.
- [Trace harness](eval/trace-harness/README.md): how third-party agent traces
  become neutral OpenAI-compatible scenarios for direct-baseline vs Attune
  comparisons.
- [Built-in defaults](profiles/README.md): how reviewed GEPA artifacts become
  embedded defaults in shipped binaries.
- [Development guide](docs/development.md): contributor workflow, project
  layout, adding repairs, adding profiles, and adding evaluation coverage.

## Current Limitations

- Upstream token streaming is not implemented. Upstream responses are buffered,
  then optionally returned to streaming clients as corrected SSE.
- API coverage is intentionally small: `/health`, `/v1/models`, and
  `/v1/chat/completions`.
- Runtime configuration supports config files and prompt artifacts, but the
  parser/repair policy schema is still a first pass and not every future
  per-model knob is exposed yet.
- Built-in default artifacts are embedded at build time, but promotion is still
  a manual review step.
- Retry/continue policy is represented in configuration but not implemented as
  an upstream retry loop yet.
- Provider-specific adapters beyond generic OpenAI-compatible HTTP are not
  implemented yet.

## License

Attune is published under the MIT License. See [`LICENSE`](LICENSE).

## Roadmap

Near-term directions:

- harden the correction-agent path across Qwen, Kimi, and GLM, especially cases
  where the repair model returns prose or invalid DSRs `tool_calls` JSON instead
  of typed calls, and long traces where the latest user task is not visible in
  the correction-agent input window
- harden config-file validation and parser/repair policy configuration
- add profile validation and migration tooling
- expand trace-harness and GEPA datasets across more model families
- explore per-model tool-surface optimization: treat semantically equivalent
  tools such as alternate edit schemas as profile-selectable artifacts, then
  load the tool shape that evals best for each model instead of assuming one
  universal tool schema. This is adjacent to Armin Ronacher's
  [Better Models: Worse Tools](https://lucumr.pocoo.org/2026/7/4/better-models-worse-tools/)
  observation that newer models can be worse at unfamiliar tool schemas.
- add provider-specific compatibility adapters where generic OpenAI-compatible
  HTTP is not enough
- improve streaming fidelity for corrected responses
- add debug endpoints or a trace inspection UI

Long-term, Attune should become a compatibility runtime that learns how each
model prefers to be prompted, how it tends to fail, and how best to recover
without requiring client applications to change.
