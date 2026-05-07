# Trace Harness Integration Evals

This directory is for dataset-backed live proxy checks. The goal is not to judge whether a model made the best engineering decision. The goal is to verify that the proxy keeps model outputs structurally usable for an OpenAI-compatible agent harness.

The flow is:

1. Download or sample a third-party trace dataset into `eval/trace-harness/raw/`.
2. Convert it into neutral scenario JSONL with `trace-harness import-*`.
3. Inspect the scenarios before spending API calls.
4. Run a small sampled set through the live proxy with `trace-harness run`.
5. Review the result report and proxy traces.
6. Promote only reviewed failures into GEPA datasets.

Raw downloads, local scenario files, and run reports are ignored by git.

## Scenario Format

Each scenario is a single OpenAI-compatible chat completion request:

- `request.messages`: the source trace conversation prefix before one assistant turn
- `request.tools`: tools inferred from the source harness trace or provided by the dataset
- `metadata.observed_*`: what the source model did next, for review only

The runner does not execute tools. Each scenario is an independent structural check of what our target model emits through the proxy for that harness-shaped context.

## Download Samples

Pi traces are JSONL sessions:

```bash
mkdir -p eval/trace-harness/raw/pi-mono
curl -L 'https://huggingface.co/datasets/badlogicgames/pi-mono/resolve/main/2026-01-16T03-18-40-694Z_89afd3da-1fa3-45f9-87ad-a023f92372ee.jsonl' \
  -o eval/trace-harness/raw/pi-mono/demo.jsonl
```

Hermes traces are large Parquet shards, so use the Hugging Face rows API for small reviewed samples:

```bash
mkdir -p eval/trace-harness/raw/hermes
curl -L 'https://datasets-server.huggingface.co/rows?dataset=lambda/hermes-agent-reasoning-traces&config=glm-5.1&split=train&offset=0&length=20' \
  -o eval/trace-harness/raw/hermes/glm-rows.json
curl -L 'https://datasets-server.huggingface.co/rows?dataset=lambda/hermes-agent-reasoning-traces&config=kimi&split=train&offset=0&length=20' \
  -o eval/trace-harness/raw/hermes/kimi-rows.json
```

## Import

```bash
cargo run -- trace-harness import-pi \
  --input-path eval/trace-harness/raw/pi-mono \
  --output-path eval/trace-harness/scenarios/pi-mono.local.jsonl \
  --max-scenarios 12

cargo run -- trace-harness import-hermes-rows \
  --input-path eval/trace-harness/raw/hermes/glm-rows.json \
  --output-path eval/trace-harness/scenarios/hermes.local.jsonl \
  --max-scenarios 6

cargo run -- trace-harness import-hermes-rows \
  --input-path eval/trace-harness/raw/hermes/kimi-rows.json \
  --output-path eval/trace-harness/scenarios/hermes.local.jsonl \
  --max-scenarios 6 \
  --append
```

Inspect before running:

```bash
cargo run -- trace-harness inspect \
  --scenarios-path eval/trace-harness/scenarios/pi-mono.local.jsonl \
  --limit 12
```

## Run

This starts the proxy in-process unless `--proxy-url` is provided. It reads `OPENROUTER_API_KEY` or `MCP_UPSTREAM_API_KEY` from the environment or `.env`. Pass `--config` before the `trace-harness` subcommand when you want a promoted profile config loaded for the in-process proxy.

```bash
cargo run -- \
  --config configs/gemma-dsrs-conservative.toml \
  trace-harness run \
  --scenarios-path eval/trace-harness/scenarios/pi-mono.local.jsonl \
  --output-path eval/trace-harness/results/gemma-pi.local.json \
  --model google/gemma-4-26b-a4b-it \
  --limit 12
```

The report checks structural invariants:

- final response parses as OpenAI-compatible JSON
- no DSRs markers or field labels leak into user-facing content
- no empty content plus empty tool calls
- no content and tool calls in the same final message
- tool call names are known for the scenario
- tool call arguments are JSON objects
- duplicate identical final tool calls are warnings

The report is a review aid, not an automatic quality grade. A model can choose a different tool than the source trace and still be structurally valid. A model can also make a poor engineering choice that is outside this proxy's scope. The proxy is responsible for clean OpenAI-compatible structure, not for making every target model as smart as the source model.

## Curating GEPA Examples

Do not bulk-add every successful harness run to GEPA. Curate examples where the expected structural behavior is clear:

- a positive case that should keep producing a valid tool call shape
- a failure where the proxy saw an empty response, malformed DSRs, leaked field labels, or a duplicate deterministic tool call
- a correction-needed case where the final clean output is a good explicit label

Reviewed request-adapter examples live under `datasets/request-adapter/`. The current trace-harness contribution is:

```text
datasets/request-adapter/gemma-dsrs-conservative-trace-harness-curated.jsonl
```

For the latest Gemma request-adapter GEPA run, combine the curated harness rows with the existing hand-labeled and trace-faithful datasets:

```bash
jq -c . \
  datasets/request-adapter/gemma-dsrs-conservative.jsonl \
  datasets/request-adapter/gemma-dsrs-conservative-trace-faithful.jsonl \
  datasets/request-adapter/gemma-dsrs-conservative-trace-harness-curated.jsonl \
  > /tmp/gemma-request-adapter-all.jsonl
```

Then run the request-adapter optimizer for the history format being tested:

```bash
cargo run -- optimize-request-adapter-prompt \
  --dataset-path /tmp/gemma-request-adapter-all.jsonl \
  --output-path datasets/request-adapter/gemma-dsrs-conservative-r3-append-only-gepa.json \
  --base-url https://openrouter.ai/api/v1 \
  --model google/gemma-4-26b-a4b-it \
  --target-model google/gemma-4-26b-a4b-it \
  --profile gemma-dsrs-conservative \
  --profile-revision 3 \
  --dsrs-history-format append_only \
  --artifact-id request-adapter/gemma-dsrs-conservative/append-only \
  --iterations 3 \
  --max-examples 14 \
  --lm-max-tokens 100000
```

Run the same dataset with `--dsrs-history-format regenerated_context` and a separate output path when comparing history renderers. Promote only after reading the artifact and confirming it improves the profile without overfitting to source-harness paths or one-off commands.
