# Command And Workflow Reference

This is the reproducible command playbook for the workflows we have been
running locally: serving the proxy, inspecting traces, exporting datasets,
running GEPA, promoting artifacts, and sampling third-party harness traces.

Most commands can be run either directly with `cargo run -- ...` or through the
Nix shell:

```sh
nix develop --command cargo run -- <command> ...
```

The examples below use `nix develop --command` where reproducibility matters.

## Environment

```sh
export OPENROUTER_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

`OPENROUTER_API_KEY` is used for OpenRouter target-model calls and normal
serving. `ANTHROPIC_API_KEY` is used by GEPA reflection/proposal and judge calls
when the model is `anthropic:claude-sonnet-4-6`.

Useful logging:

```sh
RUST_LOG=model_correction_proxy=debug,tower_http=debug \
nix develop --command cargo run -- serve
```

Use `model_correction_proxy=trace` only when you need request-shape detail; it
can be noisy and traces already contain the full postmortem data.

## Serve The Proxy

Embedded built-in defaults only:

```sh
export OPENROUTER_API_KEY="..."

nix develop --command cargo run -- \
  --bind 127.0.0.1:8080 \
  --upstream-base-url https://openrouter.ai/api/v1 \
  --trace-path traces/model-correction-proxy.jsonl \
  serve
```

With the local Gemma config-file override:

```sh
export OPENROUTER_API_KEY="..."

nix develop --command cargo run -- \
  --config configs/gemma-dsrs-conservative.toml \
  --bind 127.0.0.1:8080 \
  --upstream-base-url https://openrouter.ai/api/v1 \
  --trace-path traces/model-correction-proxy.jsonl \
  serve
```

`serve` is the default command, so `cargo run --` also starts the proxy.

## Inspect And Manage Traces

Show compact summaries:

```sh
nix develop --command cargo run -- \
  inspect-traces \
  --trace-path traces/model-correction-proxy.jsonl \
  --limit 20
```

Write pretty JSON summaries for review:

```sh
nix develop --command cargo run -- \
  inspect-traces \
  --trace-path traces/model-correction-proxy.jsonl \
  --limit 50 \
  --json
```

Replay recorded upstream responses through the current interpreter/repair path:

```sh
nix develop --command cargo run -- \
  replay \
  --trace-path traces/model-correction-proxy.jsonl
```

When traces are stale, archive or clear them only after exporting any examples
you want to preserve:

```sh
mkdir -p /tmp/model-correction-proxy-traces
cp traces/model-correction-proxy*.jsonl /tmp/model-correction-proxy-traces/
: > traces/model-correction-proxy.jsonl
: > traces/model-correction-proxy-corrections.jsonl
```

## Export Datasets From Traces

Correction-agent dataset rows:

```sh
nix develop --command cargo run -- \
  export-dataset \
  --trace-path traces/model-correction-proxy.jsonl \
  --output-path datasets/corrections.jsonl
```

Filtered correction rows:

```sh
nix develop --command cargo run -- \
  export-dataset \
  --trace-path traces/model-correction-proxy.jsonl \
  --output-path datasets/corrections-qwen.jsonl \
  --model qwen \
  --profile qwen-dsrs \
  --failure-kind dsrs_contract_violation
```

Trace-faithful request-adapter row with an explicit label:

```sh
nix develop --command cargo run -- \
  export-request-adapter-dataset \
  --trace-path traces/model-correction-proxy.jsonl \
  --output-path datasets/request-adapter/gemma-dsrs-conservative-trace-faithful.jsonl \
  --trace-id trace_bb26a5f316ca486796231bfd184ac723 \
  --expected-output-json '{"content":"","tool_calls":[{"name":"read","arguments":{"path":"/Users/tony/Dev/ThirdParties/pi-mono/packages/coding-agent/docs/packages.md"}}]}' \
  --observed-failure-kind duplicate_immediate_tool_call \
  --observed-problem "model repeated the same deterministic package-listing command" \
  --prompt-goal "read the package docs instead of repeating the listing" \
  --append
```

Use the final proxy response as the label for a known-good successful trace:

```sh
nix develop --command cargo run -- \
  export-request-adapter-dataset \
  --trace-path traces/model-correction-proxy.jsonl \
  --output-path datasets/request-adapter/gemma-dsrs-conservative-trace-faithful.jsonl \
  --trace-id trace_45804185fe3a4a7fa2e9c1af5e55c494 \
  --use-final-response \
  --append
```

Use `--allow-unlabeled` only for triage exports that you will review and label
before GEPA.

## Correction-Agent GEPA

Correction-agent GEPA optimizes the internal DSRs correction agent. It is not
the same as request-adapter GEPA.

```sh
export OPENROUTER_API_KEY="..."
export ANTHROPIC_API_KEY="..."

nix develop --command cargo run -- \
  optimize-prompts \
  --dataset-path datasets/corrections.jsonl \
  --output-path datasets/gepa-correction-prompt.json \
  --base-url https://openrouter.ai/api/v1 \
  --model anthropic:claude-sonnet-4-6 \
  --judge-model anthropic:claude-sonnet-4-6 \
  --target-model qwen/qwen3.5-9b \
  --profile qwen-dsrs \
  --profile-revision 3 \
  --artifact-id correction-agent/qwen-dsrs/local-r1 \
  --iterations 3 \
  --max-examples 12 \
  --lm-max-tokens 100000
```

Rules:

- `--target-model` is the model being improved.
- `--model` is the GEPA reflection/proposal model.
- `--judge-model` is the scoring model.
- Reflection and judge must not be the same model as the target.
- Anthropic roles use `ANTHROPIC_API_KEY`; OpenRouter target calls use
  `OPENROUTER_API_KEY`.

## Request-Adapter GEPA

Request-adapter GEPA optimizes the model/profile-specific guidance used before
the upstream model is called. This is the path we have used most for Qwen and
Gemma.

Build the Gemma dataset used by the latest reviewed run:

```sh
jq -c . \
  datasets/request-adapter/gemma-dsrs-conservative.jsonl \
  datasets/request-adapter/gemma-dsrs-conservative-trace-faithful.jsonl \
  datasets/request-adapter/gemma-dsrs-conservative-trace-harness-curated.jsonl \
  > /tmp/gemma-request-adapter-all.jsonl
```

Build the current Qwen curated dataset:

```sh
jq -c . \
  datasets/request-adapter/qwen-dsrs-trace-harness-curated.jsonl \
  > /tmp/qwen-request-adapter-all.jsonl
```

Run Gemma append-only from scratch:

```sh
export OPENROUTER_API_KEY="..."
export ANTHROPIC_API_KEY="..."

nix develop --command cargo run -- \
  optimize-request-adapter-prompt \
  --dataset-path /tmp/gemma-request-adapter-all.jsonl \
  --output-path datasets/request-adapter/gemma-dsrs-conservative-sonnet-fresh-r1-append-only-gepa.json \
  --base-url https://openrouter.ai/api/v1 \
  --model anthropic:claude-sonnet-4-6 \
  --judge-model anthropic:claude-sonnet-4-6 \
  --target-model google/gemma-4-26b-a4b-it \
  --profile gemma-dsrs-conservative \
  --profile-revision 7 \
  --dsrs-history-format append_only \
  --artifact-id request-adapter/gemma-dsrs-conservative/sonnet-fresh-r1-append-only \
  --iterations 5 \
  --max-examples 6 \
  --lm-max-tokens 100000
```

Run Qwen regenerated-context from scratch:

```sh
export OPENROUTER_API_KEY="..."
export ANTHROPIC_API_KEY="..."

nix develop --command cargo run -- \
  optimize-request-adapter-prompt \
  --dataset-path /tmp/qwen-request-adapter-all.jsonl \
  --output-path datasets/request-adapter/qwen-dsrs-sonnet-fresh-r1-regenerated-context-gepa.json \
  --base-url https://openrouter.ai/api/v1 \
  --model anthropic:claude-sonnet-4-6 \
  --judge-model anthropic:claude-sonnet-4-6 \
  --target-model qwen/qwen3.5-9b \
  --profile qwen-dsrs \
  --profile-revision 3 \
  --dsrs-history-format regenerated_context \
  --artifact-id request-adapter/qwen-dsrs/sonnet-fresh-r1-regenerated-context \
  --iterations 5 \
  --max-examples 8 \
  --lm-max-tokens 100000
```

Add `MCP_GEPA_DEBUG=1` when diagnosing optimizer behavior. It prints one line
per scored rollout with the layer, case ID, score, generalized judge feedback,
and parsed prediction. GEPA infrastructure failures are intentionally fatal:
target-model HTTP failures, judge HTTP failures, non-JSON judge responses, and
invalid judge JSON abort the command and do not write artifacts. Model behavior
failures, such as empty content with empty tool calls, are still scoreable data.

To compare history format behavior, run the same dataset again with the other
history format and a different output/artifact ID:

| Profile | History format | Output path | Artifact ID |
| --- | --- | --- | --- |
| `gemma-dsrs-conservative` | `append_only` | `datasets/request-adapter/gemma-dsrs-conservative-sonnet-fresh-r1-append-only-gepa.json` | `request-adapter/gemma-dsrs-conservative/sonnet-fresh-r1-append-only` |
| `gemma-dsrs-conservative` | `regenerated_context` | `datasets/request-adapter/gemma-dsrs-conservative-sonnet-fresh-r1-regenerated-context-gepa.json` | `request-adapter/gemma-dsrs-conservative/sonnet-fresh-r1-regenerated-context` |
| `qwen-dsrs` | `append_only` | `datasets/request-adapter/qwen-dsrs-sonnet-fresh-r1-append-only-gepa.json` | `request-adapter/qwen-dsrs/sonnet-fresh-r1-append-only` |
| `qwen-dsrs` | `regenerated_context` | `datasets/request-adapter/qwen-dsrs-sonnet-fresh-r1-regenerated-context-gepa.json` | `request-adapter/qwen-dsrs/sonnet-fresh-r1-regenerated-context` |

When continuing an existing line of experimentation, seed from the current best
artifact:

```sh
  --seed-artifact datasets/request-adapter/gemma-dsrs-conservative-sonnet-fresh-r1-append-only-gepa.json
```

## Promote GEPA Artifacts

First promote to a runtime config for local testing:

```sh
nix develop --command cargo run -- \
  promote-artifact \
  --config-path configs/gemma-dsrs-conservative.toml \
  --artifact-path datasets/request-adapter/gemma-dsrs-conservative-sonnet-fresh-r1-append-only-gepa.json \
  --profile gemma-dsrs-conservative \
  --dry-run
```

Then run without `--dry-run` only after reviewing the report and artifact:

```sh
nix develop --command cargo run -- \
  promote-artifact \
  --config-path configs/gemma-dsrs-conservative.toml \
  --artifact-path datasets/request-adapter/gemma-dsrs-conservative-sonnet-fresh-r1-append-only-gepa.json \
  --profile gemma-dsrs-conservative
```

After config-level testing, promote the same reviewed artifact into embedded
built-in defaults:

```sh
nix develop --command cargo run -- \
  promote-default-artifact \
  --artifact-path datasets/request-adapter/gemma-dsrs-conservative-sonnet-fresh-r1-append-only-gepa.json \
  --profile gemma-dsrs-conservative \
  --model-pattern gemma \
  --dry-run
```

Then run without `--dry-run`:

```sh
nix develop --command cargo run -- \
  promote-default-artifact \
  --artifact-path datasets/request-adapter/gemma-dsrs-conservative-sonnet-fresh-r1-append-only-gepa.json \
  --profile gemma-dsrs-conservative \
  --model-pattern gemma
```

Built-in promotion updates [`profiles/builtin-defaults.toml`](../profiles/builtin-defaults.toml).
The next build validates and embeds the referenced artifact.

## Trace Harness With Hugging Face Samples

The trace harness imports third-party harness traces into neutral
OpenAI-compatible scenarios, then runs sampled live structural checks through
the proxy. It does not execute source tools and does not grade whether the
model made the smartest engineering choice.

Download Pi and Hermes samples:

```sh
mkdir -p eval/trace-harness/raw/pi-mono
curl -L 'https://huggingface.co/datasets/badlogicgames/pi-mono/resolve/main/2026-01-16T03-18-40-694Z_89afd3da-1fa3-45f9-87ad-a023f92372ee.jsonl' \
  -o eval/trace-harness/raw/pi-mono/demo.jsonl

mkdir -p eval/trace-harness/raw/hermes
curl -L 'https://datasets-server.huggingface.co/rows?dataset=lambda/hermes-agent-reasoning-traces&config=glm-5.1&split=train&offset=0&length=50' \
  -o eval/trace-harness/raw/hermes/glm-rows.json
curl -L 'https://datasets-server.huggingface.co/rows?dataset=lambda/hermes-agent-reasoning-traces&config=kimi&split=train&offset=0&length=50' \
  -o eval/trace-harness/raw/hermes/kimi-rows.json
```

Import a combined 100-scenario Pi/Hermes set:

```sh
nix develop --command cargo run -- \
  trace-harness import-pi \
  --input-path eval/trace-harness/raw/pi-mono \
  --output-path eval/trace-harness/scenarios/pi-hermes-100.local.jsonl \
  --max-scenarios 50 \
  --seed 13

nix develop --command cargo run -- \
  trace-harness import-hermes-rows \
  --input-path eval/trace-harness/raw/hermes/glm-rows.json \
  --output-path eval/trace-harness/scenarios/pi-hermes-100.local.jsonl \
  --max-scenarios 25 \
  --seed 13 \
  --append

nix develop --command cargo run -- \
  trace-harness import-hermes-rows \
  --input-path eval/trace-harness/raw/hermes/kimi-rows.json \
  --output-path eval/trace-harness/scenarios/pi-hermes-100.local.jsonl \
  --max-scenarios 25 \
  --seed 13 \
  --append
```

Inspect before spending live API calls:

```sh
nix develop --command cargo run -- \
  trace-harness inspect \
  --scenarios-path eval/trace-harness/scenarios/pi-hermes-100.local.jsonl \
  --limit 12
```

Run Gemma against the sample:

```sh
export OPENROUTER_API_KEY="..."

nix develop --command cargo run -- \
  trace-harness run \
  --scenarios-path eval/trace-harness/scenarios/pi-hermes-100.local.jsonl \
  --output-path eval/trace-harness/results/gemma-pi-hermes-100.local.json \
  --model google/gemma-4-26b-a4b-it \
  --limit 100 \
  --parallel 4 \
  --request-timeout-seconds 180 \
  --retries 3
```

Run Qwen while ignoring the problematic Venice provider:

```sh
export OPENROUTER_API_KEY="..."

nix develop --command cargo run -- \
  trace-harness run \
  --scenarios-path eval/trace-harness/scenarios/pi-hermes-100.local.jsonl \
  --output-path eval/trace-harness/results/qwen-ignore-venice-pi-hermes-100.local.json \
  --model qwen/qwen3.5-9b \
  --provider-ignore venice \
  --limit 100 \
  --parallel 4 \
  --request-timeout-seconds 180 \
  --retries 3
```

Compare direct upstream baseline against the proxy on a smaller smoke set:

```sh
export OPENROUTER_API_KEY="..."

nix develop --command cargo run -- \
  trace-harness compare \
  --scenarios-path eval/trace-harness/scenarios/pi-hermes-100.local.jsonl \
  --output-path eval/trace-harness/results/qwen-ignore-venice-pi-hermes-25.compare.local.json \
  --model qwen/qwen3.5-9b \
  --provider-ignore venice \
  --limit 25 \
  --parallel 4 \
  --request-timeout-seconds 180 \
  --retries 3
```

The compare report records:

- direct baseline structural pass/fail
- proxy structural pass/fail
- `proxy_fixed_baseline_failure`
- `proxy_regressed_baseline_success`
- `both_passed` / `both_failed`
- categorized failures, such as tool-like text without OpenAI `tool_calls`
- proxy repair actions and correction-agent attempts when the proxy is started
  in-process
- latency for each endpoint call
- retry attempts and any `Retry-After` / `x-ratelimit-*` response headers

`trace-harness run` and `trace-harness compare` default to
`--request-timeout-seconds 180`, `--retries 3`, `--retry-backoff-ms 1000`, and
`--parallel 1`. Increase `--parallel` for exploratory live runs once the model
and provider are stable. Retries are used for request/response read failures,
HTTP 429, HTTP 408, HTTP 425, and 5xx responses; `Retry-After` and
`x-ratelimit-reset` are honored when present, otherwise exponential backoff is
used.

Use `--baseline-base-url` to compare against another OpenAI-compatible target
endpoint. Use `--proxy-url` when you already have a proxy process running and do
not want the harness to start one in-process.

If you already have a proxy running externally, pass `--proxy-url` to avoid the
in-process proxy:

```sh
nix develop --command cargo run -- \
  trace-harness run \
  --scenarios-path eval/trace-harness/scenarios/pi-hermes-100.local.jsonl \
  --output-path eval/trace-harness/results/qwen-external.local.json \
  --proxy-url http://127.0.0.1:8080 \
  --model qwen/qwen3.5-9b \
  --provider-ignore venice \
  --limit 24
```

## Live OpenRouter E2E Smoke

The live e2e test is ignored by default because it makes paid API calls.

```sh
set -a
source .env
set +a

OPENROUTER_LIVE_MODELS=qwen/qwen3.5-9b,google/gemma-4-26b-a4b-it \
nix develop --command cargo test --test proxy_e2e live_openrouter_pi_prompt_matrix -- --ignored --nocapture
```

Run one model at a time when debugging provider or profile behavior:

```sh
OPENROUTER_LIVE_MODELS=qwen/qwen3.5-9b \
nix develop --command cargo test --test proxy_e2e live_openrouter_pi_prompt_matrix -- --ignored --nocapture
```

## Local Mock Upstream

Start a mock upstream:

```sh
nix develop --command cargo run --example mock_upstream
```

Override its response:

```sh
MOCK_UPSTREAM_BIND=127.0.0.1:18081 \
MOCK_UPSTREAM_CONTENT='<tool_call name="read_file">{pth:"Cargo.toml"}</tool_call>' \
nix develop --command cargo run --example mock_upstream
```

Run the proxy against it:

```sh
nix develop --command cargo run -- \
  --bind 127.0.0.1:18080 \
  --upstream-base-url http://127.0.0.1:18081/v1 \
  serve
```

## Standard Verification

```sh
nix develop --command cargo fmt --check
nix develop --command cargo test --all-targets --all-features -- --test-threads=1
nix develop --command cargo test --all-targets --all-features
```

Use the serial test command when validating a branch before a commit. The normal
parallel run is still useful for catching order-dependent test flakes.
