# Built-in Defaults

This directory controls which reviewed GEPA artifacts are compiled into the
shipped binary as default model-profile behavior.

Runtime config files are still the normal customization path. The built-in
defaults layer is the final promotion step for artifacts that should work even
when someone installs or ships `model-correction-proxy` as a standalone binary
with no local `datasets/` directory.

For the runtime config schema that can override these built-ins, see
[`../docs/config-reference.md`](../docs/config-reference.md).

## Promotion Ladder

1. Run GEPA and write an experiment artifact under `datasets/`.
2. Inspect the artifact for score, behavior, overfit, and prompt drift.
3. Promote it into a config with `promote-artifact`.
4. Test the config against unit, integration, trace-harness, and live checks.
5. Promote it into `profiles/builtin-defaults.toml` with
   `promote-default-artifact` only when it should become a shipped default.

The config promotion and built-in-default promotion are deliberately separate.
A profile can be good enough for local testing before it is good enough to ship
as the default for everyone.

## Manifest Shape

`builtin-defaults.toml` has two arrays:

- `[[artifacts]]` lists checked-in GEPA artifact JSON files.
- `[[profiles]]` says which artifact IDs a built-in model profile should use.

Artifact paths are repository-root relative in normal committed manifests. The
build script validates that each artifact exists, is valid JSON, has matching
`artifact_id` and `artifact_type`, and contains an instruction field such as
`best_instruction`.

Profile entries can set:

- `name`
- `revision`
- `model_patterns`
- `dsrs_history_format`
- `request_adapter_artifact`
- `correction_agent_artifact`

The build script embeds the referenced artifact contents with `include_str!`.
At runtime, built-in profiles expose artifact references as `builtin:<artifact-id>`
in traces and profile metadata.

## Current Shipped Defaults

| Profile | Active history format | Active artifact | Revision |
| --- | --- | --- | ---: |
| `qwen-dsrs` | `regenerated_context` | `request-adapter/qwen-dsrs/sonnet-fresh-r1-regenerated-context` | 3 |
| `gemma-dsrs-conservative` | `append_only` | `request-adapter/gemma-dsrs-conservative/sonnet-fresh-r1-append-only` | 7 |

The fresh Sonnet-judged append-only Qwen artifact and regenerated-context Gemma
artifact are also embedded as reviewed alternates, but they are not active
profile defaults. The older Gemma r4 artifact remains embedded for provenance.

Runtime config files still match before these built-ins. For example,
`configs/gemma-dsrs-conservative.toml` is a filesystem-artifact override that
points at the same reviewed Gemma instruction as the embedded default.

## Commands

Promote a reviewed artifact into a runtime config:

```sh
cargo run -- promote-artifact \
  --config-path configs/gemma-dsrs-conservative.toml \
  --artifact-path datasets/request-adapter/gemma-dsrs-conservative-sonnet-fresh-r1-append-only-gepa.json \
  --profile gemma-dsrs-conservative
```

Promote the same reviewed artifact into the shipped built-in defaults:

```sh
cargo run -- promote-default-artifact \
  --artifact-path datasets/request-adapter/gemma-dsrs-conservative-sonnet-fresh-r1-append-only-gepa.json \
  --profile gemma-dsrs-conservative \
  --model-pattern gemma
```

Use `--dry-run` first for both commands. After changing this manifest, run the
normal test suite so `build.rs` validates the embedded artifacts and the runtime
tests verify profile resolution.

## Override Rules

Built-in defaults are not a lock-in mechanism. Runtime config profiles still
match before built-ins and can override any profile field.

Config artifacts can point at normal files:

```toml
request_adapter_artifact = "../datasets/request-adapter/custom-gepa.json"
```

They can also reuse an embedded built-in artifact:

```toml
request_adapter_artifact = "builtin:request-adapter/gemma-dsrs-conservative/sonnet-fresh-r1-append-only"
```

That lets a user start from shipped defaults, override model patterns or policy,
and still avoid depending on local artifact files.
