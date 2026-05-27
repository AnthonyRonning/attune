# Development Guide

This document captures contributor-facing workflow notes that should not crowd
the public README.

## Recommended Workflow

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

Use the command reference for live OpenRouter tests, trace-harness runs, GEPA,
promotion, and mock upstream workflows:

- [`docs/command-reference.md`](command-reference.md)

## Project Layout

```text
.
├── Cargo.toml
├── build.rs
├── flake.nix
├── brainstorming-guidelines.md
├── intent.md
├── docs/
│   ├── command-reference.md
│   ├── config-reference.md
│   ├── development.md
│   ├── model-configurability.md
│   └── polar-attune.md
├── profiles/
│   ├── README.md
│   └── builtin-defaults.toml
├── configs/
├── datasets/
├── eval/
│   └── trace-harness/
├── examples/
│   └── mock_upstream.rs
├── src/
└── tests/
```

Important source areas:

| Area | Files |
| --- | --- |
| HTTP/OpenAI surface | `src/gateway.rs`, `src/openai.rs`, `src/upstream.rs` |
| Request adaptation | `src/normalizer.rs`, `src/prompt_adapter.rs`, `src/dsrs_contract.rs` |
| Profiles/config | `src/model_profile.rs`, `src/config.rs`, `profiles/`, `configs/` |
| Interpretation/repair | `src/response_interpreter.rs`, `src/repair.rs`, `src/policy.rs` |
| Correction agents | `src/agents.rs` |
| Traces/datasets/eval | `src/trace.rs`, `src/dataset.rs`, `src/replay.rs`, `src/eval.rs`, `src/trace_harness.rs` |
| GEPA/artifacts | `src/optimization.rs`, `src/promotion.rs`, `build.rs` |

## Adding A Repair

1. Add detection in `src/response_interpreter.rs` if the output shape is new.
2. Add deterministic or schema-guided repair in `src/repair.rs`.
3. Record a clear `RepairAction` with confidence and reason.
4. Add unit tests and, when possible, a regression case.
5. Run replay against existing traces to catch behavior changes.

Prefer typed failure kinds and traceable repair actions over ad hoc string
handling. If a case requires semantic judgment, route it through the
correction-agent path instead of making the deterministic parser guess.

## Adding A Model Profile

Prefer adding or overriding profiles through `--config` /
`ATTUNE_CONFIG_PATH`. Built-in Rust profiles are useful for shipped defaults
and common model families.

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
3. Decide whether the profile should use DSRs, XML, tagged JSON, native
   pass-through, conservative parallel-tool settings, or model-specific
   correction/judge overrides.
4. Add profile resolution tests.

See also:

- [`docs/config-reference.md`](config-reference.md)
- [`docs/model-configurability.md`](model-configurability.md)
- [`profiles/README.md`](../profiles/README.md)

## Adding Evaluation Coverage

Prefer regression cases based on real traces. A good case should include:

- the original request and tools
- the raw upstream response
- expected final tool calls or content
- the model/profile being evaluated
- the failure mode being protected

For live third-party harness traces, use the trace harness first, review the
result, then curate specific examples into the request-adapter or
correction-agent datasets. Do not bulk-add passing traces just because a run
was successful.

See:

- [`eval/trace-harness/README.md`](../eval/trace-harness/README.md)

## Safety And Correctness Principles

- Preserve valid upstream output.
- Repair syntax before calling a correction model.
- Repair schema shape only when intent is clear.
- Avoid fabricating missing required values by default.
- Enforce client-provided parallel-tool settings.
- Keep corrections auditable in traces.
- Treat traces as sensitive application data.
- Prefer model-specific profiles over scattered one-off hacks.
