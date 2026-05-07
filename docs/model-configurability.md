# Model Configurability

This document describes how the proxy should evolve from today's code-level model profiles into a durable configuration layer for model-specific behavior.

The goal is not just "load settings from a file." The goal is to make every model-specific behavior explicit, versioned, testable, traceable, and optimizable while keeping shared parsing and repair logic reusable.

## Why this matters

The original project thesis depends on model-specific adaptation. The proxy exists because OpenAI-compatible applications send prompts and tool definitions that may work for one model and fail badly for another. The same tool format, instruction wording, correction-agent prompt, and parser tolerance will not be ideal across Qwen, Kimi, GLM, Llama, Gemma, and future models.

A sustainable architecture needs to answer these questions for each request:

- Which model and provider is this request targeting?
- Which request adapter should be used?
- Which output formats should be considered valid, tolerated, suspicious, or unrecoverable?
- Which deterministic repairs are allowed?
- When should the correction agent be called?
- Which correction model, signature, prompt artifact, confidence threshold, and dataset should be used?
- Which GEPA artifact produced the current instructions?
- Which profile revision should be attached to traces so failures can be reproduced later?

## Current Reality

The current implementation already has the core pipeline needed for this design:

| Area | Current behavior |
| --- | --- |
| API surface | `GET /health`, `GET /v1/models`, and `POST /v1/chat/completions` through `src/gateway.rs` |
| Upstream | Generic OpenAI-compatible `/chat/completions` and `/models` through `src/upstream.rs` |
| Request normalization | `src/normalizer.rs` produces `NormalizedRequest` with messages, tools, tool choice, and parallel-tool setting |
| Model profiles | `src/model_profile.rs` resolves by substring against code-defined profiles |
| Tool modes | `ToolMode::ProxyOwned` strips native upstream tools; `ToolMode::PassThrough` keeps native tools |
| Tool formats | `ToolFormat::Dsrs` is the primary path; XML and tagged JSON renderers still exist |
| DSRs request adapter | `src/dsrs_contract.rs` renders system/developer context, tools, `tool_choice`, `parallel_tool_calls`, and profile-selected conversation history format into DSRs-compatible prompts |
| Streaming behavior | Upstream is forced to `stream: false`; client streaming is synthesized after repair |
| Token controls | Client `max_tokens` and `max_completion_tokens` pass through when present and remain unset when omitted |
| Response interpretation | `src/response_interpreter.rs` detects native tool calls, DSRs, XML, tagged JSON, markdown JSON, direct JSON, and function-like known-tool calls |
| Failure classification | Typed failures include DSRs contract violations, content outside tagged fields, empty DSRs output, invalid DSRs tool JSON, prompt echo, template leak, malformed known tool calls, premature tool stop, schema violation, and unknown tools |
| Repair policy | `src/repair.rs` uses deterministic and schema-guided repair first, then calls the correction agent for failures that need semantic recovery |
| Correction agent | `src/agents.rs` uses a typed DSRs signature with `possible`, `confidence`, `explanation`, `content`, and `tool_calls` fields |
| Correction traces | `src/trace.rs` writes correction-agent sidecar records with input, raw output, accepted state, confidence, explanation, content, and tool calls |
| Dataset flow | `src/dataset.rs` exports main traces into correction datasets |
| Replay and eval | `src/replay.rs` and `src/eval.rs` replay traces and run regression suites |
| Optimization | `src/optimization.rs` runs GEPA against correction-agent and request-adapter prompt programs and writes profile-aware artifacts |
| Artifact promotion | `src/promotion.rs` validates a GEPA artifact and promotes it into the matching profile config field |

The first configuration slice is implemented. `ProxyConfig` can now load TOML, JSON, or JSON5 files from `--config` / `MCP_CONFIG_PATH`; configured profiles override built-ins; profile revision/source/history-format/artifact metadata is written to traces; request-adapter and correction-agent instruction artifacts are loaded at startup; and GEPA reports include profile, revision, artifact, and request-adapter history-format metadata.

The remaining limitation is that not every planned per-model knob is exposed yet. Tool mode, tool format, model matching, profile guidance, correction model, correction passes, and prompt artifacts are configurable. Parser strictness, correction trigger policy, provider identity, and retry strategy still live mostly in shared code and coarse global config.

## Shared vs Model-Specific

The configuration layer should draw a hard line between shared mechanics and model-specific strategy.

Shared logic should include:

- OpenAI-compatible request and response types
- request normalization
- trace schema and redaction rules
- DSRs parser mechanics
- XML/tagged/JSON parser mechanics
- schema validation helpers
- deterministic JSON repair
- OpenAI response construction
- replay and eval harnesses
- correction-agent execution plumbing

Model-specific configuration should include:

- model/provider matching rules
- whether to use native tools or proxy-owned tools
- preferred tool format
- request-adapter instruction text or prompt artifact
- few-shot examples or demonstrations, if added later
- enabled parser families and tolerance levels
- which failure kinds route to correction
- correction model selection
- correction-agent signature version and instruction artifact
- min correction confidence
- max correction passes
- judge model and judge policy, when implemented
- retry/continue policy, when implemented
- GEPA dataset and artifact references

The guiding rule: shared code should know how to parse or repair a format; profiles should decide which formats and policies are appropriate for a model.

## Proposed Config Shape

The exact file format can be TOML, YAML, or JSON. The important part is the structure. A future `model-profiles.yaml` could look like this:

```yaml
version: 1

defaults:
  profile: balanced-default
  upstream:
    stream_mode: buffer_then_sse
    pass_through_token_limits: true
  trace:
    main_path: traces/model-correction-proxy.jsonl
    correction_path: traces/model-correction-proxy-corrections.jsonl

profiles:
  balanced-default:
    revision: 1
    match:
      model_patterns: ["*"]
    request_adapter:
      tool_mode: proxy_owned
      tool_format: dsrs
      dsrs_history_format: append_only
      signature: openai_tool_use_contract/v1
      native_tools: strip
      instruction_artifact: builtins/default-dsrs-tool-instruction.txt
    response_interpreter:
      enabled_sources:
        - native
        - dsrs
        - xml
        - tagged_json
        - markdown_json
        - direct_json
        - function_like_text
      dsrs:
        require_markers: true
        route_content_outside_markers_to_correction: true
        tolerate_inner_field_labels: true
        tolerate_adjacent_tool_arrays: true
      suspicious_stop: true
    repair:
      deterministic_json_repair: true
      schema_guided_repair: true
      unknown_tool_policy: pass_through_unless_close_match
      parallel_tool_policy: enforce_client_setting
      correction_triggers:
        - native_malformed_json_arguments
        - dsrs_contract_violation
        - dsrs_content_outside_tagged_fields
        - dsrs_invalid_tool_calls_json
        - dsrs_invalid_tool_calls_shape
        - dsrs_placeholder_only
        - template_leak
        - prompt_echo
        - premature_tool_stop
        - malformed_known_tool_call
        - untagged_dsrs_like_output
        - schema_violation
    correction_agent:
      enabled: true
      model: same_as_upstream
      signature: correct_malformed_tool_response/v1
      instruction_artifact: builtins/default-dsrs-correction-instruction.txt
      max_context_messages: 12
      min_confidence: 0.65
      max_passes: 1

  qwen-dsrs:
    revision: 1
    extends: balanced-default
    match:
      model_patterns: ["qwen", "qwq"]
    optimization:
      correction_dataset: datasets/qwen/correction-agent.jsonl
      correction_prompt_artifact: profiles/qwen-dsrs/correction-agent/gepa-v1.json

  gemma-dsrs-conservative:
    revision: 1
    extends: balanced-default
    match:
      model_patterns: ["gemma"]
    request_adapter:
      instruction_overrides:
        - Prefer a single item in the tool_calls field.
        - Do not emit multiple tool calls unless explicitly required.
    repair:
      parallel_tool_policy: enforce_single_when_client_disables_parallel
    correction_agent:
      max_passes: 1
```

This shape keeps the current built-in behavior expressible while creating room for profile-specific prompt artifacts and eval datasets. The implemented first pass uses the existing top-level `ProxyConfig` shape plus `model_profiles`; deeper nested `request_adapter`, `response_interpreter`, `repair`, and `correction_agent` sections remain a planned schema refinement.

## Runtime Resolution

Request handling should resolve configuration in this order:

1. Normalize the incoming OpenAI-compatible request.
2. Resolve provider identity, if known.
3. Match the model name against user-configured profiles.
4. Fall back to built-in profiles.
5. Fall back to `balanced-default`.
6. Merge inherited profile settings.
7. Attach the resolved `profile_id`, `profile_revision`, and artifact IDs to the trace.
8. Adapt the request using the resolved request-adapter config.
9. Interpret and repair using shared code plus profile-selected policies.
10. Record policy decisions and correction attempts with the same profile metadata.

Configured profiles should win over built-ins so users can tune a model without recompiling.

## Configuration Layers

### Model Identity

Model matching should support more than substring matching over the raw model string.

Useful match inputs:

- provider name, such as `openrouter`, `vllm`, `ollama`, or `llama.cpp`
- upstream base URL, if provider name is not configured
- exact model name
- model family pattern
- model size or capability tags, if supplied manually
- context length, if known
- native tool support quality, if known

The current substring resolver is a good MVP, but model profile artifacts need stable IDs and revisions.

### Request Adapter

This layer owns how the client request becomes an upstream request.

Configurable fields should include:

- `tool_mode`: `proxy_owned` or `pass_through`
- `tool_format`: `dsrs`, `xml`, `tagged_json`, or future formats
- native tool policy: strip, pass through, or provider-specific
- DSRs signature version
- profile guidance artifact
- examples or demonstrations
- conversation-history rendering policy through `dsrs_history_format`
- tool-result rendering policy tied to the selected history format
- parallel-tool instruction policy
- stream policy
- token-limit policy

The token-limit policy should preserve caller intent: pass through limits that the client set, and leave them unset if the client did not set them.

Implemented DSRs history formats:

- `append_only`: default. Runtime context is sent first, then the original conversation is appended as chat messages. User turns remain normal user messages, prior assistant turns are rendered as DSRs `content` / `tool_calls` outputs, and tool results are rendered as observed tool-result messages.
- `regenerated_context`: legacy format. The whole non-system conversation is regenerated into one serialized `conversation` field with `assistant_tool_calls` and tool-result text. This remains useful for models that respond better to a single compact transcript.

### Response Interpreter

The interpreter should stay shared, but profiles should choose strictness and routing.

Configurable fields should include:

- enabled parser sources
- DSRs marker strictness
- whether content outside DSRs fields is a correction-triggering violation
- whether a syntactically valid DSRs response with empty `content` and empty `tool_calls` is treated as a correction-triggering premature stop
- whether prompt echoes and template artifacts are correction-triggering violations
- whether XML/tagged/direct JSON parses are accepted as valid tool intent or only used as hints
- suspicious-stop detection sensitivity
- reasoning-field mapping policy

The important distinction is "parsed" versus "accepted." A profile can let the shared parser recognize a shape while still routing it to correction if the model violated the selected contract.

### Repair Policy

Policy should describe what the proxy is allowed to change.

Configurable fields should include:

- deterministic JSON repair on/off
- schema-guided repair on/off
- fuzzy tool-name threshold
- unknown-tool policy
- missing-required-argument policy
- scalar-to-string coercion policy
- array wrapping policy
- parallel-tool enforcement policy
- correction-agent trigger failure kinds
- fallback behavior when correction fails

This keeps one-off fixes from spreading through the parser. New failures should become typed failure kinds, policy decisions, regression cases, and eventually dataset examples.

### Correction Agent

The correction agent is the easiest high-value place to start model-specific GEPA optimization because the proxy controls the whole prompt, signature, dataset, and acceptance policy.

Configurable fields should include:

- correction model selector: same as upstream, global override, profile override, or dedicated model
- DSRs signature version
- instruction artifact
- dataset ID
- GEPA artifact ID
- max context messages
- min confidence
- max passes
- accepted output types: tool calls, content, or both
- optional judge model and judge prompt, when implemented

The correction agent should remain a typed DSRs agent. Avoid going back to a custom JSON envelope as the primary contract.

### GEPA and Artifacts

GEPA artifacts should be treated as versioned runtime inputs, not loose reports.

Each artifact should record:

- artifact ID
- target model or model family
- profile ID and revision it was trained against
- request-adapter history format, when the artifact tunes request-adapter guidance
- signature name and version
- dataset path or dataset ID
- optimizer model
- score and metric summary
- created time
- prompt/instruction text
- checksum of the dataset or dataset manifest

Request-adapter prompts and correction-agent prompts should be optimized separately. Their datasets, metrics, and failure modes are different.

For example, a model that emits a valid DSRs envelope with empty `content` and `[]` tool calls has failed at the request-adapter prompt layer, not the correction-agent layer. That trace belongs in a model/profile-specific request-adapter dataset so GEPA can improve the profile guidance, while runtime policy should still route the empty response through correction or a future retry path.

Promotion is now explicit rather than automatic. The optimizer writes JSON artifacts for review; `promote-artifact` then validates `artifact_type`, `profile`, and layer metadata, updates the appropriate profile config field, carries `dsrs_history_format` from request-adapter artifacts into the profile config, and increments the profile revision. That same command works for request-adapter artifacts and correction-agent artifacts, so every profile can follow the same dataset -> GEPA -> inspect -> promote loop. Formatter-comparison GEPA outputs are ignored by default until a specific artifact is reviewed and promoted.

### Traces and Datasets

Every trace should include enough profile metadata to explain why the proxy behaved as it did:

- request model
- provider, if known
- resolved profile ID
- resolved profile revision
- request-adapter artifact ID
- correction-agent artifact ID
- parser sources enabled
- failure kinds
- policy decisions
- repair actions
- correction-agent raw output and accepted state

Dataset export should support filtering by model, profile, failure kind, repair action, and correction-agent result. That makes it practical to build model-specific GEPA datasets instead of mixing unrelated model behavior.

The request-adapter dataset path now has its own trace-faithful exporter. `export-request-adapter-dataset` can select traces by `--trace-id`, model/profile, failure kind, repair action, or correction result. Rows preserve the original OpenAI request JSON from the trace, so GEPA can rehydrate the exact messages, tools, tool choice, parallel-tool setting, profile metadata, adapted request metadata, and observed upstream output. Labels stay explicit through `--expected-output-json`, `--expected-output-path`, or `--use-final-response`; unlabeled rows require `--allow-unlabeled` and should be treated as triage data, not optimization data.

## Implementation Plan

### Step 1: Split profile shape from config loading

Keep the current `ModelProfile` behavior, but introduce config structs that can deserialize the future profile file. Start by supporting the fields that exist today:

- name/profile ID
- model patterns
- tool mode
- tool format
- correction model
- judge model
- max correction passes
- parallel-tool support
- tool instruction

Then add explicit profile revision and source metadata.

Status: implemented for the existing `ModelProfile` shape.

### Step 2: Add a config file loader

Add a CLI option and environment variable:

- `--config`
- `MCP_CONFIG_PATH`

The loader should merge:

1. built-in defaults
2. config-file profiles
3. CLI/env overrides

Config-file profiles should override built-ins by profile ID or by match priority.

Status: implemented. Configured profiles are matched before built-ins.

### Step 3: Add profile metadata to traces

Before behavior gets more dynamic, traces need stable profile metadata. Add:

- `profile_id`
- `profile_revision`
- `profile_source`
- `request_adapter_artifact`
- `correction_agent_artifact`

This makes future regressions reproducible.

Status: implemented for main traces, summaries, correction attempts, correction-agent sidecar traces, datasets, and eval reports.

### Step 4: Hot-load correction-agent instructions

Load the correction-agent instruction artifact into the DSRs correction signature path. This is the safest first hot-loading target because the correction agent is internal and typed.

Do not hot-load arbitrary parser behavior first. Parser behavior should remain typed and code-reviewed.

Status: implemented. `correction_agent_artifact` can load a JSON GEPA report or text instruction file and applies it to the typed DSRs correction signature.

### Step 5: Add model/profile-filtered dataset export

Add CLI flags for:

- `--model`
- `--profile`
- `--failure-kind`
- `--repair-action`
- `--correction-result`

This gives GEPA model-specific datasets instead of one mixed correction file.

Status: implemented for the main trace dataset exporter.

### Step 6: Add profile-aware eval matrices

Regression suites should report by:

- model
- provider
- profile
- profile revision
- adapter artifact
- correction artifact

This is how the project can prove a profile update improved one model without regressing another.

Status: partially implemented. Eval reports now include model/profile/revision/artifact groupings and model/profile filters; richer regression-suite generation from traces is still future work.

### Step 7: Add request-adapter prompt artifacts

After correction-agent artifact loading works, add artifact loading for request-adapter profile guidance. Keep the DSRs signature stable, and let artifacts tune wording, examples, and profile guidance.

Status: artifact loading is implemented through `request_adapter_artifact`. A dedicated request-adapter GEPA optimizer is now available through `optimize-request-adapter-prompt`; it evaluates candidate profile guidance by rendering the real runtime DSRs request format selected by `--dsrs-history-format`. `export-request-adapter-dataset` converts trace IDs into trace-faithful request-adapter rows with explicit labels. `datasets/request-adapter/gemma-dsrs-conservative-trace-faithful.jsonl` is the first exact Gemma-specific prompt-adapter dataset. Separate append-only and regenerated-context GEPA artifacts can be generated for comparison, but trial outputs should stay ignored unless promoted.

### Step 8: Add explicit artifact promotion

GEPA artifacts should not silently change runtime behavior. Promotion should be a deliberate command that can be inspected, reviewed, committed, reverted, and reproduced.

Status: implemented through `promote-artifact`. The command supports request-adapter and correction-agent GEPA artifacts, preserves existing profile model patterns unless `--model-pattern` is supplied, records the previous and new artifact reference in its JSON report, carries request-adapter `dsrs_history_format` into config, and supports `--dry-run`.

## Guardrails

- Do not make every parser behavior a model-specific branch.
- Do not accept malformed output just because one model often emits it.
- Do not hide semantic repairs from traces.
- Do not overwrite caller token limits.
- Do not set token limits when the caller omitted them.
- Do not mix correction-agent datasets across models without tagging them.
- Do not hot-load prompts without recording artifact IDs in traces.
- Do not let profile config silently change OpenAI response shape.

## Near-Term Target

The next practical milestone is deeper profile schema validation and parser/repair policy configuration. The config file, profile revision trace metadata, prompt artifact loading, and explicit artifact promotion path are now in place.

That gets the project closer to the original intent: any OpenAI-compatible application can request any model, and the proxy can select the right behavior profile, collect traces, build datasets, optimize prompts, and improve reliability without requiring the application to change.
