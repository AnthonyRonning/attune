# Configuration Reference

This document describes the runtime configuration that exists today. The
longer-term model-profile architecture is described in
[`docs/model-configurability.md`](model-configurability.md).

Config files may be TOML, JSON, or JSON5. The parser uses the file extension
when present and otherwise tries TOML, JSON, then JSON5.

## Loading And Precedence

Use `--config <path>` or `ATTUNE_CONFIG_PATH=<path>` to load a config file.
Configured `[[model_profiles]]` are matched before built-in profiles, so local
config can override shipped defaults without recompiling.

For `serve`, runtime values resolve in this order:

| Setting | Precedence |
| --- | --- |
| Config path | `--config`, then `ATTUNE_CONFIG_PATH`, then unset |
| Bind address | `--bind`, then `ATTUNE_BIND_ADDR`, then `127.0.0.1:8080` |
| Upstream base URL | `--upstream-base-url`, then `ATTUNE_UPSTREAM_BASE_URL`, then config/default |
| Upstream API key | `--upstream-api-key`, then `ATTUNE_UPSTREAM_API_KEY`, then `OPENROUTER_API_KEY`, then config |
| Main trace path | `--trace-path`, then `ATTUNE_TRACE_PATH`, then config/default |

If no upstream API key is configured, the proxy forwards the inbound client
`Authorization` or `x-api-key`/`api-key` credential to the upstream provider.

Artifact paths inside config files are resolved relative to the config file
directory unless the path is absolute. `builtin:<artifact-id>` references load
an artifact embedded through [`profiles/builtin-defaults.toml`](../profiles/builtin-defaults.toml).

## Minimal Config

```toml
[upstream]
base_url = "https://openrouter.ai/api/v1"
timeout_seconds = 120

[trace]
enabled = true
path = "traces/attune.jsonl"
correction_path = "traces/attune-corrections.jsonl"
```

## Full Shape

```toml
[upstream]
base_url = "https://openrouter.ai/api/v1"
api_key = "optional-provider-key"
timeout_seconds = 120

[correction]
enabled = true
default_model = "optional-correction-model"
max_context_messages = 12
min_confidence = 0.65

[trace]
enabled = true
path = "traces/attune.jsonl"
correction_path = "traces/attune-corrections.jsonl"

[policy]
mode = "balanced"
deterministic_json_repair = true
schema_guided_repair = true
correction_agent = true
retry_or_continue = true
hallucinated_tool_match_threshold = 0.86
max_tool_calls_without_parallel = 1

[[model_profiles]]
name = "gemma-dsrs-conservative"
model_patterns = ["google/gemma-4-26b-a4b-it", "gemma"]
revision = 8
request_adapter_artifact = "builtin:request-adapter/gemma-dsrs-conservative/sonnet-post50-r2-append-only"
correction_agent_artifact = "optional-correction-artifact.json"
tool_mode = "proxy_owned"
tool_format = "dsrs"
dsrs_history_format = "append_only"
correction_model = "google/gemma-4-26b-a4b-it"
max_correction_passes = 1
supports_parallel_tool_calls = false
tool_instruction = "Optional inline request-adapter guidance."

[model_profiles.provider]
ignore = ["venice"]
allow_fallbacks = true
require_parameters = true
```

## Top-Level Sections

| Section | Purpose |
| --- | --- |
| `[upstream]` | OpenAI-compatible upstream endpoint and request timeout. |
| `[correction]` | Global correction-agent settings. |
| `[trace]` | Main request trace and correction-agent sidecar trace paths. |
| `[policy]` | Shared repair policy toggles and thresholds. |
| `[[model_profiles]]` | Model-specific request formatting, correction, artifacts, and provider routing. |

## Upstream

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `base_url` | string | `https://openrouter.ai/api/v1` | OpenAI-compatible API root. |
| `api_key` | string | unset | Used when no inbound client auth or CLI/env override exists. |
| `timeout_seconds` | integer | `120` | Applies to upstream HTTP calls. |

## Correction

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Global switch for correction-agent calls. |
| `default_model` | string | unset | Used when a profile does not set `correction_model`; otherwise the upstream request model is used. |
| `max_context_messages` | integer | `12` | Recent messages sent to the correction agent. |
| `min_confidence` | float | `0.65` | Accepted correction outputs must be `possible=true` and meet this confidence. |

Correction agents use the same upstream base URL as the main proxy unless future
provider-specific correction routing is added.

## Trace

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enables main request JSONL traces. |
| `path` | path | `traces/attune.jsonl` | Main request/response trace file. |
| `correction_path` | path | `traces/attune-corrections.jsonl` | Correction-agent attempt sidecar trace file. |

Trace files contain full request/response data. Treat them as sensitive
debugging artifacts, especially when user prompts, tool outputs, or API
responses may contain secrets.

## Policy

| Field | Type | Default | Current behavior |
| --- | --- | --- | --- |
| `mode` | `conservative`, `balanced`, `aggressive_recovery` | `balanced` | Recorded/logged as policy posture; detailed mode branching is still limited. |
| `deterministic_json_repair` | bool | `true` | Enables deterministic malformed JSON argument repair. |
| `schema_guided_repair` | bool | `true` | Enables current schema-guided key/shape repair. This is lightweight validation, not full JSON Schema enforcement. |
| `correction_agent` | bool | `true` | Allows correction-agent fallback when `correction.enabled` and profile passes permit it. |
| `retry_or_continue` | bool | `true` | Reserved for a future upstream retry/continue loop; not a full retry policy yet. |
| `hallucinated_tool_match_threshold` | float | `0.86` | Similarity threshold for mapping close tool-name mistakes to known tools. |
| `max_tool_calls_without_parallel` | integer | `1` | Maximum final tool calls when the client disables parallel tool calls. |

## Model Profiles

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | string | required | Stable profile ID recorded in traces. |
| `model_patterns` | string array | `[]` | Case-insensitive substring matching. `*` matches all models. Configured profiles are checked before built-ins. |
| `revision` | integer | `1` | Stored in traces and incremented by promotion commands. |
| `source` | string | `builtin` | Config-loaded profiles are marked as `config` after loading. |
| `request_adapter_artifact` | string | unset | JSON GEPA report, text prompt file, or `builtin:<artifact-id>`. Overrides `tool_instruction` when loaded. |
| `correction_agent_artifact` | string | unset | JSON GEPA report, text prompt file, or `builtin:<artifact-id>`. Overrides `correction_instruction` when loaded. |
| `correction_instruction` | string | unset | Inline correction-agent instruction override. |
| `tool_mode` | `proxy_owned`, `pass_through` | `proxy_owned` | Proxy-owned mode strips native upstream tools and renders a prompt contract. |
| `tool_format` | `dsrs`, `xml`, `tagged_json` | `dsrs` | Request-adapter format. DSRs is the primary path. |
| `dsrs_history_format` | `append_only`, `regenerated_context` | `append_only` | Controls how prior user/assistant/tool messages are rendered into the DSRs request. |
| `correction_model` | string | unset | Per-profile correction-agent model override. |
| `judge_model` | string | unset | Reserved/profile metadata today; GEPA judge model is configured by GEPA CLI flags/env. |
| `provider` | table | unset | OpenRouter-compatible provider routing object injected only when the request did not already provide one. |
| `max_correction_passes` | integer | `1` | `0` disables correction-agent fallback for the profile. |
| `supports_parallel_tool_calls` | bool | `true` | Profile capability metadata and prompt guidance. Final truncation is driven by the client `parallel_tool_calls` request setting and policy limit. |
| `tool_instruction` | string | built-in DSRs instruction | Inline request-adapter/profile guidance. Usually generated from a promoted request-adapter artifact. |

## Provider Routing

Provider routing currently targets OpenRouter-style request fields.

```toml
[model_profiles.provider]
order = ["deepinfra", "together"]
only = []
ignore = ["venice"]
allow_fallbacks = true
require_parameters = true
```

| Field | Type | Notes |
| --- | --- | --- |
| `order` | string array | Preferred providers, in order. |
| `only` | string array | Restrict routing to these providers. |
| `ignore` | string array | Exclude providers, commonly `["venice"]` for Qwen tests. |
| `allow_fallbacks` | bool | Maps to OpenRouter provider fallback behavior. |
| `require_parameters` | bool | Requires provider support for requested parameters. |

The trace harness also has provider flags for one-off live comparisons:
`--provider-order`, `--provider-only`, `--provider-ignore`,
`--disable-provider-fallbacks`, and `--require-provider-parameters`.

Live harness commands also accept transport controls:
`--request-timeout-seconds` (default `180`), `--retries` (default `3`),
`--retry-backoff-ms` (default `1000`), and `--parallel` (default `1`). Retries
cover request errors, response-body read errors, HTTP 429, HTTP 408, HTTP 425,
and 5xx responses. The harness records `Retry-After` and `x-ratelimit-*`
headers in endpoint reports and honors `Retry-After` / `x-ratelimit-reset`
when deciding retry delay.

## Artifact References

Artifacts may be:

- relative paths, resolved from the config file directory
- absolute paths
- embedded references, such as
  `builtin:request-adapter/gemma-dsrs-conservative/sonnet-post50-r2-append-only`

JSON GEPA artifacts are expected to contain an instruction field such as
`best_instruction`. Plain text files are loaded as the instruction body.

Current active built-in request-adapter defaults:

| Profile | Artifact | History format |
| --- | --- | --- |
| `qwen-dsrs` | `builtin:request-adapter/qwen-dsrs/sonnet-post50-r2-regenerated-context` | `regenerated_context` |
| `gemma-dsrs-conservative` | `builtin:request-adapter/gemma-dsrs-conservative/sonnet-post50-r2-append-only` | `append_only` |

## Examples

### Use Embedded Defaults Only

```sh
export OPENROUTER_API_KEY="..."

nix develop --command cargo run -- \
  --bind 127.0.0.1:8080 \
  serve
```

### Use The Local Gemma Profile Override

```sh
export OPENROUTER_API_KEY="..."

nix develop --command cargo run -- \
  --config configs/gemma-dsrs-conservative.toml \
  --bind 127.0.0.1:8080 \
  serve
```

### Override Qwen Provider Routing

```toml
[[model_profiles]]
name = "qwen-dsrs"
model_patterns = ["qwen", "qwq"]
revision = 4
request_adapter_artifact = "builtin:request-adapter/qwen-dsrs/sonnet-post50-r2-regenerated-context"
tool_mode = "proxy_owned"
tool_format = "dsrs"
dsrs_history_format = "regenerated_context"

[model_profiles.provider]
ignore = ["venice"]
```
