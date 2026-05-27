# Polar And Attune

This note compares Attune with NVIDIA's Polar / ProRL Agent Server project and
captures how the two systems could fit together.

Reviewed sources:

- Paper: <https://arxiv.org/pdf/2605.24220>
- Repository: <https://github.com/NVIDIA-NeMo/ProRL-Agent-Server>

## Short Version

Polar and Attune share the same important insight: the model API boundary is a
powerful place to work on agent behavior without rewriting the agent harness.

They use that boundary for different purposes.

| System | Main goal | Improvement mechanism |
| --- | --- | --- |
| Polar | Make existing agent harnesses trainable | Capture token-faithful rollouts, score them, reconstruct trajectories, then train the model with GRPO or other trainer loops |
| Attune | Make tool-use contracts reliable at inference time | Translate requests into DSRs contracts, interpret model output, repair malformed responses, call typed correction agents, trace failures, and GEPA-optimize model profiles |

A fair framing is:

> Polar uses the proxy boundary to make harnesses trainable. Attune uses the
> proxy boundary to make model tool-use contracts reliable without requiring a
> fine-tune.

The caveat is important: Attune does not replace fine-tuning for capability
alignment. It changes the runtime contract around the model. Polar changes the
model weights.

## What Polar Does

Polar is an RL rollout framework for real agent harnesses. Instead of asking
the harness to implement a trainer-specific environment interface, Polar points
the harness at a compatible model API gateway.

The gateway:

1. Detects the incoming provider API shape, including OpenAI Chat, OpenAI
   Responses, Anthropic Messages, and Google generateContent-style requests.
2. Translates the provider request into the OpenAI Chat shape expected by the
   local inference backend.
3. Adds training fields such as `logprobs`.
4. Calls a served model, currently through an SGLang-oriented path.
5. Stores the original request, transformed request, response, token ids,
   logprobs, finish reason, and metadata as completion records.
6. Converts the response back into the provider shape expected by the harness.
7. Builds trainer-facing trajectories after the harness run finishes.
8. Runs evaluators and sends rewards/results back to a trainer bridge.

The "improvement" in Polar's reported results is model training. Their SWE-Gym
experiment starts from `Qwen/Qwen3.5-4B`, runs real coding harness rollouts,
scores the results with SWE-Bench/SWE-Gym style evaluators, reconstructs
token-faithful trajectories with `prefix_merging`, and feeds those trajectories
to Slime asynchronous GRPO.

Reported SWE-Bench Verified results from the paper:

| Harness | Base | Polar RL | Gain |
| --- | ---: | ---: | ---: |
| Codex | 3.8% | 26.4% | +22.6 |
| Claude Code | 29.8% | 34.6% | +4.8 |
| Qwen Code | 34.6% | 35.2% | +0.6 |
| Pi | 34.2% | 40.4% | +6.2 |

So Polar is not primarily fixing malformed tool calls at runtime. It is making
real harness execution produce usable RL/SFT data.

## What Attune Does

Attune sits at a similar API boundary, but it is an inference-time contract
runtime.

Attune's runtime path:

1. Accepts a normal OpenAI-compatible request.
2. Normalizes messages, tools, tool choice, and profile metadata.
3. Selects a model profile.
4. Renders a proxy-owned DSRs contract for the selected model.
5. Rewrites conversation history so prior assistant tool calls and tool results
   teach the same contract the model is expected to follow now.
6. Calls the upstream model.
7. Interprets native OpenAI tool calls, DSRs, XML, tagged JSON, markdown JSON,
   direct JSON, and function-like text tool intent.
8. Repairs clear syntax/schema failures deterministically.
9. Sends ambiguous or semantic failures to a typed DSRs correction agent.
10. Returns a valid OpenAI-compatible assistant message when recovery succeeds.
11. Stores traces that can become regression cases, GEPA datasets, or promotion
    evidence.

Attune's improvement loop is not weight training. It is:

1. Run the proxy against real client/harness traffic.
2. Inspect successful and failed traces.
3. Export exact traces into request-adapter or correction-agent datasets.
4. Run GEPA with a stronger optimizer/judge model.
5. Review the artifact.
6. Promote the artifact into profile config.
7. Promote reviewed defaults into the binary when they should ship.

This is closer to DSPy/DSRs-style behavioral adaptation than RL fine-tuning:
make the contract explicit, optimize the instructions around that contract, and
repair the boundary when the model/provider/client stack loses intent.

## Where They Overlap

The overlap is meaningful:

- Both systems treat the agent harness as mostly opaque.
- Both intervene at the LLM API boundary.
- Both normalize provider/harness request and response formats.
- Both care about multi-turn tool-use traces.
- Both keep the client/harness contract stable while changing what happens
  behind the model endpoint.
- Both can turn real harness behavior into reusable data.
- Both are motivated by the gap between model capability and harness-specific
  execution protocols.

This is good validation for Attune. Polar independently arrives at the same
architectural pressure point: the model endpoint is the lowest common interface
shared by many agent systems.

## Where They Differ

Polar optimizes for training fidelity.

- It wants the exact sampled token ids and logprobs.
- It tries to avoid retokenization drift.
- It reconstructs trajectories with loss masks.
- It keeps generated model tokens separate from harness-inserted interstitial
  tokens.
- It is trainer and harness infrastructure.

Attune optimizes for runtime structural reliability.

- It wants the client to receive usable `content` and/or `tool_calls`.
- It can transform the prompt contract before the model call.
- It can repair or correct malformed outputs after the model call.
- It cares about whether an assistant turn is structurally usable, not whether
  the model made the smartest task decision.
- It is an inference contract runtime and evaluation loop.

The biggest technical difference is token fidelity. Polar cannot blindly train
on repaired output, because repaired tokens were not sampled by the behavior
policy. Attune is allowed to repair output for the user-facing response because
its first job is keeping the live harness loop alive.

## How To Combine Them

The systems can be combined, but they should not be naively chained without
preserving raw model behavior.

### 1. Polar As A Trace Factory For Attune

Polar can run real harnesses at scale and capture much richer sessions than
manual Attune testing.

Useful flow:

```text
Polar harness rollout
  -> completion/session records
  -> Attune trace-harness scenarios
  -> direct baseline vs Attune proxy eval
  -> curated GEPA dataset rows
  -> promoted Attune model profiles
```

This is the lowest-risk integration. It does not change Polar's training path
and it gives Attune far more real-world traces.

### 2. Attune As A Structural Evaluator For Polar

Attune's interpreter can label structural quality:

- clean content
- clean tool call
- content plus tool call
- empty assistant output
- reasoning-only output
- malformed DSRs
- malformed native JSON arguments
- tool intent leaked as prose
- deterministic repair needed
- correction-agent repair needed
- correction-agent failed

Polar could use those labels as auxiliary rewards, filters, or diagnostics.
This would train models away from structural failures that break harness loops.

Important constraint: structural rewards should not dominate task rewards. A
model can produce perfectly formatted but useless actions. The structural signal
is best used as an auxiliary reliability reward or as a filter for broken
rollouts.

### 3. Train Models On Attune's DSRs Contract

This is the most interesting long-term path.

Instead of only using Attune as a runtime wrapper, Polar could run harness
rollouts where the model is trained to speak Attune's DSRs contract natively:

```text
Harness request
  -> Attune request transform
  -> trainable model endpoint
  -> raw DSRs model output captured with token ids/logprobs
  -> Attune interpreter labels structural quality
  -> harness-compatible OpenAI response
  -> evaluator reward
  -> Polar trajectory builder
  -> GRPO/SFT training
```

If done correctly, this trains the model to emit the same contract Attune uses
at inference time. Over time, fewer deterministic repairs and correction-agent
calls should be needed.

### 4. Store Dual Records

For any combined path, store both raw and Attune-final state:

| Record | Purpose |
| --- | --- |
| Raw model output | Token-faithful RL/SFT training, logprob alignment, behavior-policy evidence |
| Interpreted output | Structural labels, parser diagnostics, eval/debugging |
| Repaired/corrected output | User-facing response, correction-agent datasets, possible SFT targets, GEPA examples |

The raw output is the only safe source for behavior-policy token training.
The repaired output is still extremely valuable, but for different uses.

### 5. Attune For Deployed Polar-Trained Models

After Polar trains a model for a harness, Attune can still sit in front of that
model in production:

```text
Client/harness -> Attune profile for trained model -> trained model -> Attune repair/correction -> client/harness
```

The trained model should require less help, but Attune remains useful for
provider drift, model regressions, unknown prompts, and trace-driven monitoring.

## Risks And Design Constraints

**Do not train on repaired tokens as if they were sampled.**  
This would violate Polar's token-faithful training premise. Repaired output can
be SFT/correction data, but the RL trajectory should know what was actually
sampled.

**Avoid recursive Attune correction during model-under-test training.**  
If the correction agent uses the same model being trained, the signal can become
confusing. Correction/evaluator models should be explicitly configured.

**Keep runtime reliability and training fidelity separate.**  
Attune may choose to save the user-facing turn. Polar may need to preserve a
failed raw turn because failure is the training signal.

**Do not overfit to coding harnesses only.**  
Both systems are currently most validated on coding agents. The integration
should preserve generic trace schemas so browser, OS, workflow, and custom tool
agents can be added later.

**Provider defaults may conflict.**  
Polar's transformers are training-backend oriented and may impose fields like
`logprobs` or backend-compatible generation settings. Attune should keep its
policy of passing through caller token limits and avoiding unsolicited
`max_tokens`.

**Synthetic streaming needs explicit semantics.**  
Both systems may buffer upstream responses and synthesize provider-shaped
streams. That is acceptable, but trace metadata should make it explicit.

## Suggested Roadmap

### Phase 1: Reference And Import

- Keep Polar as an external reference project.
- Add a Polar/ProRL import path for completion sessions or saved trajectories.
- Convert selected Polar records into Attune trace-harness scenarios.
- Run direct baseline vs Attune proxy comparisons from those scenarios.

### Phase 2: Structural Labels

- Reuse Attune's response interpreter as a batch labeler.
- Emit structural labels and failure kinds for Polar-style completion records.
- Export labels into Attune GEPA datasets and optional Polar reward metadata.

### Phase 3: DSRs Contract Training Experiment

- Pick one small model and one harness, likely Qwen or Gemma with Pi.
- Route rollouts through an Attune DSRs request transform.
- Capture raw DSRs outputs token-faithfully.
- Compare:
  - direct harness baseline
  - Attune-only runtime improvement
  - Polar-trained model without Attune
  - Polar-trained model with Attune

### Phase 4: Fine-Tuned Profile Defaults

- Add profile metadata for trained checkpoints.
- Track whether a profile is prompt-adapted, fine-tuned, or both.
- Keep GEPA artifact promotion and fine-tune artifact/version promotion
  separate.

## Current Recommendation

Treat Polar as complementary, not competitive.

For Attune's near-term MVP, the highest-value use is learning from Polar's
architecture and possibly importing its traces. Attune should keep focusing on
the no-fine-tune runtime path: DSRs contracts, model profiles, response
interpretation, repair/correction, GEPA, and baseline comparisons.

Longer term, the combined system is compelling:

```text
Attune makes the contract explicit and reliable.
Polar turns real harness execution into trainable trajectories.
Together they can produce models that natively follow reliable tool-use
contracts, while still keeping a runtime safety net in front of them.
```
