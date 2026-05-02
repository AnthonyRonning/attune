# Reliability Refactor Notes

Temporary working note captured before pausing implementation.

## Context

The project intent is still valid: this proxy should be a model behavior
compatibility layer for OpenAI-compatible agent applications, not a pile of
case-by-case JSON and string fixers.

The important thesis from the original docs is:

- Tool calling is an application-level contract.
- The proxy should own that contract instead of trusting inference-engine tool
  parsing.
- Failures should be parsed, repaired, retried, traced, evaluated, and optimized
  per model.
- DSRs/dspy-style signatures and correction agents are central, not incidental.

The current implementation has the right major pieces:

- OpenAI-compatible gateway
- proxy-owned DSRs rendering
- response interpreter
- deterministic/schema-guided repair
- DSRs correction agent
- JSONL traces
- dataset export, replay, eval, GEPA scaffolding

The main drift is that parser failures are still mostly untyped strings plus a
single `suspicious_stop` boolean. That makes policy hard to reason about and
encourages one-off fixes.

## Refactor Goal

Make the live pipeline more reliable and adaptable by separating:

1. raw parsing
2. typed failure classification
3. policy decision
4. repair/correction execution
5. trace/dataset recording

The desired behavior is not "deterministically salvage everything." The desired
behavior is:

- accept valid DSRs contract output
- deterministically repair simple syntax/schema problems when meaning is clear
- route malformed DSRs contract violations to the correction agent
- suppress unsafe leftovers when correction fails
- record enough structured data to debug and optimize later

## First Implementation Pass

### 1. Add typed interpreter failures

Extend `InterpretedResponse` with something like:

```rust
pub failures: Vec<ResponseFailure>,
```

Candidate failure kinds:

```rust
pub enum ResponseFailureKind {
    NoChoices,
    NativeMalformedJsonArguments,
    DsrsContractViolation,
    DsrsContentOutsideTaggedFields,
    DsrsInvalidToolCallsJson,
    DsrsInvalidToolCallsShape,
    DsrsPlaceholderOnly,
    TemplateLeak,
    PromptEcho,
    PrematureToolStop,
    MalformedKnownToolCall,
    UntaggedDsrsLikeOutput,
    SchemaViolation,
    UnknownTool,
}
```

Keep `parse_events` for human-readable logs, but stop making policy depend on
string matching.

### 2. Make DSRs parser report contract violations

In `src/dsrs_contract.rs`, detect:

- non-whitespace text before the first DSRs output marker
- non-whitespace text after `[[ ## completed ## ]]`
- invalid `tool_calls` JSON
- invalid `tool_calls` shape
- placeholder-only content/tool fields
- copied prompt/template artifacts

Tagged DSRs can still be parsed when valid. But if tagged output violates the
contract, the interpreter should carry a typed DSRs contract failure.

Important policy decision:

Malformed tagged DSRs should not be accepted as "close enough" just because a
fallback parser extracts `content = "---"` or similar. That should route to
correction.

### 3. Route contract violations through correction

In `src/repair.rs`, update `should_try_correction_agent` so correction runs for:

- suspicious premature tool stops
- malformed known tool calls
- DSRs contract violations
- invalid DSRs `tool_calls`
- schema-missing or malformed arguments where intent exists

If correction fails, suppress contract-violation leftovers instead of returning
raw artifacts such as `---`, copied prompt text, invalid arrays, or DSRs markers.

### 4. Record correction attempts in traces

Add trace fields, likely:

```rust
pub correction_attempts: Vec<CorrectionTrace>,
pub policy_decisions: Vec<PolicyDecisionTrace>,
```

Correction trace should include:

- input model/profile
- parser events
- typed failures
- malformed raw assistant response
- correction model
- correction output preview or parsed envelope
- accepted/rejected
- confidence
- error if any
- latency

This is how we answer: "where did the problem actually happen?"

### 5. Fix dataset export to preserve raw failures

`src/dataset.rs` currently uses `interpreted.content` as `malformed_response`.
That can lose the actual failure. For example, a long malformed upstream output
can collapse to `"---"`.

Dataset rows should include:

- raw upstream assistant content
- raw upstream reasoning
- interpreted content
- typed failures
- parser events
- correction attempts
- final response
- repair actions

This makes traces useful for correction-agent optimization.

### 6. Add focused regression/e2e coverage

Add tests for the exact recent failure class:

- upstream emits prose and JSON tool calls outside DSRs fields
- then leaks copied conversation/prompt template
- then emits empty DSRs markers
- parser classifies it as a DSRs contract violation
- repair routes it to correction agent
- correction recovers tool calls
- if correction is unavailable or fails, final response suppresses malformed
  artifacts instead of returning `---`

Also keep tests proving:

- valid tagged DSRs with tool calls still works
- valid tagged DSRs no-tool content still works
- adjacent JSON arrays inside the tagged `tool_calls` field still work
- untagged DSRs-like output goes to correction, not deterministic DSRs parsing

## Longer-Term Follow-Ups

### Policy engine

Move from scattered conditionals to a policy plan:

```text
Interpret raw response
  -> artifacts + typed failures
  -> policy plan
  -> execute deterministic repair / correction / retry / suppress / pass-through
```

### Configurable model profiles

Move profiles out of hardcoded Rust defaults and into versioned config files.
Profiles should control:

- tool format
- contract strictness
- correction model
- judge model
- retry/continue behavior
- known failure modes
- prompt examples
- dataset tags

### Retry/continue

`retry_or_continue` exists in policy config but is not a real stage yet. Add one
bounded retry/continue path for clear "I will use a tool now" premature stops.

### Metrics

Trace and eval should expose:

- tool-call recovery rate
- correction-agent usage rate
- deterministic repair rate
- suppression rate
- false correction rate
- latency by stage
- failure kinds by model/profile

## Resume Point

Start with typed failures and DSRs contract violation routing. That is the
smallest change that gets the architecture back toward the original intent while
directly addressing the latest trace failure.
