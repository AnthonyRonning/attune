# Brainstorming Guidelines

## Working framing

This project is best understood as a **model behavior compatibility layer for OpenAI-compatible agent applications**.

The proxy should sit between existing applications and model providers or inference engines, observe the request and response contract, and repair or adapt behavior so that open-source models can participate reliably in ecosystems that were mostly designed around a small number of highly capable proprietary models.

The core thesis:

> Tool calling should not be treated as an inference-engine feature that applications blindly trust. Tool calling should be treated as an application-level contract that can be prompted, parsed, repaired, retried, evaluated, and optimized per model.

That framing matters because the project is not merely a “JSON fixer.” It is a compatibility and reliability layer for model behavior.

## Why this matters

Open-source models are often much more capable than their integration experience suggests. Many failures that appear to be model intelligence failures are actually failures in:

- chat templates
- inference-engine tool parsers
- stop condition handling
- brittle structured-output assumptions
- model-agnostic prompts
- application prompts tuned for a small set of frontier models
- provider-specific quirks leaking into supposedly standard APIs

The result is a frustrating mismatch:

- The model understands that it should use a tool.
- The model begins expressing that intent.
- The inference engine or API layer fails to parse the intended structure.
- The client receives a “finished” assistant message with no tool call.
- The application stops prematurely.

In an autonomous-agent setting, silent stopping is especially harmful. A bad tool parse should not collapse the whole agent loop.

## Guiding belief

The proxy should assume that models are text generators first.

Tool use, structured output, reasoning fields, and function-call messages are contracts imposed on top of generated text. Those contracts can be model-specific, optimized, and repaired. They should not be entrusted entirely to vLLM, llama.cpp, provider APIs, or any single chat template implementation.

## Project goals

### Primary goal

Make open-source models work more reliably with existing OpenAI-compatible agent applications without requiring those applications to be rewritten.

### Secondary goals

- Preserve the OpenAI-compatible API surface expected by clients.
- Improve tool-calling reliability for models whose native tool support is brittle.
- Detect and recover from malformed, partial, or misplaced model outputs.
- Support model-specific prompt and repair strategies.
- Generate high-quality datasets from real failures.
- Enable prompt optimization using DSPy-style methods, dsrs, GEPA, or similar systems.
- Make every correction auditable and replayable.

### Long-term goal

Become a general-purpose compatibility runtime that learns how each model prefers to be prompted, how it tends to fail, and how best to recover from those failures.

## Non-goals for the early project

The first version should avoid trying to solve everything.

Early non-goals:

- Do not build a full agent framework.
- Do not build an IDE-specific coding agent.
- Do not require client applications to change their prompting style.
- Do not require all users to adopt XML tools on day one.
- Do not optimize for perfect streaming behavior in the MVP.
- Do not attempt to solve every model family immediately.
- Do not hide correction decisions from developers.
- Do not make aggressive semantic changes unless there is a clear policy and confidence score.

The proxy should be infrastructure, not an agent product.

## Current implementation snapshot

The current codebase has implemented most of the early architecture described in this document.

What exists now:

- OpenAI-compatible `/v1/chat/completions`, `/v1/models`, and `/health` endpoints.
- Generic OpenAI-compatible upstream calls.
- Request normalization into an internal request shape.
- Code-level built-in model profiles selected by model-name substring, with config-file profile overrides.
- Proxy-owned DSRs tool rendering as the primary path, with pass-through repair still available.
- Translation of system/developer context, conversation history, prior assistant tool calls, and tool results into DSRs request fields.
- Buffered upstream responses, with corrected SSE returned to streaming clients after repair.
- Response interpretation for native tool calls, DSRs, XML, tagged JSON, markdown JSON, direct JSON, and function-like known-tool calls.
- Typed response failure kinds for contract violations, template leaks, prompt echoes, malformed known tool calls, schema violations, and suspicious stops.
- Deterministic and schema-guided repair before model-based correction.
- A typed DSRs correction agent with `possible`, `confidence`, `explanation`, `content`, and `tool_calls` outputs.
- Main JSONL request traces plus correction-agent sidecar traces.
- Trace summaries, correction-agent dataset export, request-adapter dataset export, replay, regression evaluation, and metadata-rich GEPA artifacts.
- TOML/JSON/JSON5 config loading through `--config` / `MCP_CONFIG_PATH`.
- Runtime loading of request-adapter and correction-agent instruction artifacts from profiles.
- Profile revision/source/artifact metadata recorded in main and correction-agent traces.
- Dataset export filters by model, profile, failure kind, repair action, and correction result.
- Append-only and regenerated-context DSRs history renderers as profile-selectable request-adapter formats.
- Request-adapter GEPA optimization using the same runtime DSRs formatter as the proxy.
- A local JSON GEPA meta-adapter so optimizer reflection/proposal calls do not collide with literal DSRs bracket examples in generated instructions.
- Explicit `promote-artifact` flow for reviewed request-adapter and correction-agent GEPA artifacts.
- Explicit `promote-default-artifact` flow plus `profiles/builtin-defaults.toml` and `build.rs` validation for artifacts that should ship inside the binary.
- `eval/trace-harness` for importing third-party agent traces, sampling live proxy runs, and curating structural edge cases into request-adapter datasets.
- A promoted Gemma append-only request-adapter artifact in both `configs/gemma-dsrs-conservative.toml` and the embedded built-in defaults manifest.

What is still missing:

- Full parser/repair policy configurability per model profile.
- Broader promoted default coverage for more model families.
- Profile validation and migration tooling.
- Provider-specific adapters beyond generic OpenAI-compatible HTTP.
- An implemented retry/continue upstream loop.
- True upstream token streaming.

The new model configurability design lives in `docs/model-configurability.md`.

## Core architecture

At a high level:

```text
Client application
  -> OpenAI-compatible proxy endpoint
    -> request normalizer
    -> model profile resolver
    -> prompt and tool adapter
    -> upstream model call
    -> response interpreter
    -> repair / judge / retry pipeline
    -> OpenAI-compatible response builder
  -> client application
```

The client should believe it is speaking to a normal OpenAI-compatible endpoint. Internally, the proxy can use a much richer process.

## Major components

### 1. Gateway

The gateway owns the public API contract.

Responsibilities:

- expose OpenAI-compatible endpoints
- accept chat completion requests
- handle authentication or pass-through authentication
- route requests to model providers
- support provider-specific configuration
- normalize request metadata
- return responses in the shape the caller expects

The gateway should be intentionally boring. Its job is to preserve compatibility.

### 2. Request normalizer

The normalizer converts messy client requests into an internal representation.

It should understand:

- messages
- system prompts
- developer prompts, if applicable
- user messages
- assistant messages
- tool result messages
- available tools
- tool choice settings
- response format hints
- temperature and sampling settings
- model name aliases
- provider-specific options

The internal representation should make it easy for later stages to reason about the request without constantly dealing with provider-specific request shapes.

### 3. Model profile resolver

The model profile resolver decides which behavior profile applies to the request.

A model profile may include:

- preferred tool rendering format
- prompt preamble style
- known parser weaknesses
- retry strategy
- correction strategy
- maximum correction passes
- whether to use native tool calling
- whether to avoid native tool calling
- how to handle reasoning fields
- how strict to be about JSON schemas
- whether multiple tool calls are allowed
- preferred correction model
- preferred judge model
- dataset tags

This is one of the most important design ideas in the project. A Kimi profile, Gemma profile, GLM profile, Qwen profile, and Llama profile may all need different prompting and repair behavior.

### 4. Prompt and tool adapter

This layer transforms the client request into a form suitable for the upstream model.

In the simplest early mode, it may mostly pass through the prompt and tools.

In a more opinionated mode, it may:

- consume OpenAI-style tools
- render them as XML-style tools
- render them as tagged JSON
- add model-specific instructions
- rewrite overly frontier-model-specific tool instructions
- include examples
- include schema summaries
- include explicit recovery instructions
- prevent the upstream engine from using brittle native tool parsing

This is where the project can eventually apply DSPy/dsrs/GEPA optimizations.

### 5. Upstream client

The upstream client talks to the actual model backend.

The first version should treat the upstream as a normal **OpenAI-compatible chat/completions API**. That gives the proxy immediate coverage across most practical backends without building special integrations first.

Initial upstream target:

- generic OpenAI-compatible `/v1/chat/completions`

Backends commonly covered by that shape:

- vLLM
- OpenAI-compatible model servers
- Ollama
- LM Studio
- llama.cpp servers
- OpenRouter
- direct provider APIs
- local custom inference endpoints

The important principle is that upstreams should be swappable. The proxy should not become coupled to vLLM, even if vLLM is the most common place where these issues appear.

Later versions can add tighter provider-specific adapters for vLLM, llama.cpp, custom endpoints, or direct provider APIs, but the first candidate should be the standard OpenAI-compatible client path.

### 6. Response interpreter

The response interpreter takes the raw upstream output and asks:

- Is this a normal assistant message?
- Is this a valid tool call?
- Is there an attempted tool call hidden in content?
- Is there malformed JSON?
- Is content present in a reasoning/thinking field that needs client-contract-aware handling while preserving the data?
- Did the model indicate intent to use a tool but fail to emit one?
- Did the model emit a tool name that does not exist?
- Did the model produce multiple tool calls in a nonstandard format?
- Did the upstream finish reason look suspicious?

This layer should separate parsing from correction. First interpret what happened. Then decide what to do.

### 7. Repair engine

The repair engine attempts to turn an imperfect response into a valid client-facing response.

Repair should proceed from cheapest and most deterministic to most semantic:

1. exact parse
2. tolerant parse
3. deterministic cleanup
4. schema-guided repair
5. correction-agent repair
6. retry or continue
7. pass through or fail according to policy

The repair engine should avoid unnecessary model calls when simple deterministic fixes are enough.

### 8. Policy engine

The policy engine decides when to repair, retry, continue, pass through, or fail.

Examples:

- If JSON is invalid only because of a trailing comma, fix it deterministically.
- If the tool name is slightly misspelled and there is one obvious match, correct it.
- If the tool name is hallucinated and no close match exists, either retry or return the hallucinated call depending on policy.
- If the model says “I will read the file now” and stops with no tool call, try a continue or correction path.
- If the response is ambiguous and correction confidence is low, avoid fabricating behavior.

The policy layer is what prevents the proxy from becoming too magical or unsafe.

### 9. Telemetry and dataset store

Every interesting request should be available for analysis and optimization.

Useful trace data:

- normalized request
- model profile used
- upstream provider
- raw upstream response
- parsed response
- parser errors
- repair actions
- correction-agent prompt
- correction-agent output
- final client response
- latency
- token usage
- confidence scores
- user-configured policy decisions
- whether the client later accepted or rejected the tool call

This trace store is not just logging. It is the foundation for future datasets and optimization.

Initial trace decision:

- write local JSONL traces in the first version
- make traces easy to replay into parser, repair, correction-agent, and GEPA/dsrs optimization workflows
- keep richer stores such as SQLite or external databases as later options
- make retention configurable
- default retention should be “do not delete”; users can rotate or delete traces explicitly by configuration

## Phase 1 product shape

The first useful version should be intentionally narrow and reliable.

Recommended phase 1:

- OpenAI-compatible `/v1/chat/completions`
- non-streaming responses only
- support standard OpenAI-style `tools`
- normalize requests into an internal format
- call one or more upstream OpenAI-compatible model endpoints
- hold the full upstream response before returning anything to the client
- detect common tool-call and structured-output failures
- repair simple malformed JSON deterministically
- include correction-agent fallback as a first-class configurable capability
- allow optional correction features to be disabled by configuration
- return a valid OpenAI-compatible response
- record local JSONL traces for debugging, replay, and dataset generation

The MVP should prove that the proxy can make existing applications more reliable without requiring those applications to change.

## Important phase 1 decision: pass-through tools vs proxy-owned tools

There are two plausible early modes.

### Mode A: pass-through with repair

The proxy sends tools to the upstream API mostly as the client provided them, then repairs bad responses afterward.

Benefits:

- easier to integrate
- closer to existing OpenAI-compatible behavior
- less invasive
- simpler first demonstration

Risks:

- still depends on vLLM or provider tool parsing
- may not fix the deepest failure mode
- may receive already-truncated or “safely failed” responses

### Mode B: proxy-owned tool rendering

The proxy consumes OpenAI-style tools, does not rely on upstream native tool calling, renders tools into text using a model-specific format, and parses the model’s plain-text response back into OpenAI-compatible tool calls.

Benefits:

- avoids brittle inference-engine tool parsers
- allows XML-style tools
- enables model-specific prompting
- gives the proxy full control over parsing and repair
- aligns with the DSPy/dsrs/GEPA philosophy

Risks:

- more design work
- more parsing responsibility
- may need per-model prompt profiles sooner
- harder to be fully transparent to every client

### Recommendation

Support both modes, but make **proxy-owned tool rendering** the primary reliability path.

Pass-through repair is useful for drop-in compatibility and comparison testing, but it should not be the main bet. If the core complaint is that inference-engine tool parsing is unreliable, the distinctive value of this project comes from owning the tool contract outside the inference engine.

## Tool-calling philosophy

Tool calling should be represented internally as intent plus arguments, not merely as whatever shape the upstream returned.

The proxy should be able to recognize tool intent from several forms:

- native API tool calls
- JSON blobs
- XML-style tags
- markdown code fences
- plain text with obvious tool intent
- multiple structured calls in one response
- partially malformed structured calls

The final client response can still be OpenAI-compatible. Internally, the proxy should be more flexible.

## Correction scenarios

### 1. “Let me do X” but no tool call

The model says something like:

> I’ll read the file now.

Then it stops with no tool call.

Possible causes:

- upstream tool parser swallowed the tool attempt
- stop sequence fired too early
- model emitted invalid tool syntax
- model was prompted in a way that encouraged narration instead of action
- the application prompt assumes a stronger model

Possible responses:

- continue the same upstream conversation with an instruction to emit the intended tool call
- ask a correction agent to infer the intended tool call from context
- retry with a stronger tool-format reminder
- return the natural language response unchanged if confidence is too low

This case is central because it reflects the “silent stop” failure mode.

### 2. Malformed JSON tool call

Examples:

- trailing comma
- single quotes
- comments inside JSON
- unquoted keys
- markdown fenced JSON
- escaped string problems
- missing closing brace
- duplicated fields
- schema field in the wrong place

Preferred handling:

- deterministic repair first
- schema validation second
- correction agent only if deterministic repair fails

The project should be careful not to “invent” missing required values unless policy allows it.

### 3. Tool call inside normal content

The model may emit a tool call as text instead of an API tool call.

Example patterns:

- XML-style call in content
- JSON object in content
- function-call-looking text
- markdown block containing arguments

This should be considered a recoverable format issue if the tool name and arguments are valid.

### 4. Content inside reasoning or thinking

Some backends or model wrappers may place user-visible text in a reasoning field.

The proxy should parse reasoning/thinking fields as part of the response payload and preserve them by default. This project is not a content filter and should not strip reasoning, content, or other generated text simply because it appeared in a particular field.

If a client contract expects visible text in `content` but a backend placed it in a reasoning/thinking field, the proxy may map or duplicate that text into the appropriate client-facing field. That should be treated as compatibility normalization, not content removal.

If summarized reasoning is added later, that is an explicit output-mode feature that replaces or augments the reasoning stream according to configuration. Until then, reasoning fields should be parsed, preserved, and passed through as faithfully as the client API allows.

### 5. Tool hallucination

The model calls a tool that does not exist.

Possible policies:

- return the hallucinated tool and let the application fail it
- map to the nearest valid tool if confidence is very high
- retry with the actual available tool list
- ask a correction agent to map the intended action to an available tool
- convert to a normal assistant message explaining the tool is unavailable

Current decision:

- use the correction agent as a judge for hallucinated tool names
- if the tool name is a slight mistake or near-obvious match, correct it as a normal repair
- if the tool name is not close to any available tool, pass the hallucinated tool call back to the client and let the application handle the failure

The proxy cannot and should not catch everything. A tool call whose intent maps to no available tool is a client/application-level error case, and many clients can naturally handle it by returning a tool-not-found error back into the conversation.

### 6. Wrong schema shape

The model calls the right tool but arguments do not match the schema.

Examples:

- string instead of array
- array instead of object
- wrong field name
- missing nested object
- field placed at top level instead of under `arguments`
- enum value close but not exact

Possible handling:

- deterministic coercion for safe cases
- schema-guided repair
- correction agent
- retry

The proxy should preserve the original malformed value in traces.

Current decision for missing required or incorrect arguments:

- default behavior should be lenient rather than overly strict
- do not aggressively invent missing required values by default
- allow malformed or incomplete arguments to pass through to the client when repair confidence is low
- make stricter enforcement optional by policy, model profile, or tool risk
- allow correction-agent inference only when configured and when the missing value is clear from context

### 7. Multiple tool calls in one response

Some models can produce consecutive tool calls naturally, especially with XML-style prompting.

The proxy should eventually support this well.

The client request should guide this behavior. If the incoming API request includes a setting such as `parallel_tool_calls` or an equivalent compatibility flag, the proxy should parse and preserve that setting during request normalization and enforce it during response building.

Possible handling:

- if multiple tool calls are allowed, preserve multiple valid calls in the final response
- if multiple tool calls are disabled, return at most one tool call or trigger a retry/correction policy
- if the client omits the setting, use the model profile and OpenAI-compatible default behavior
- if the model emits multiple calls in a malformed block, correction may split them only when the client request allows multiple calls

Current decision:

- strictly enforce the client-provided multiple/parallel tool-call setting

This is an area where open models may actually be more capable than the native API layer allows.

### 8. Mixed reasoning, content, and tools

Models may produce:

- reasoning text
- user-facing content
- tool calls
- summaries
- scratchpad artifacts

The proxy needs rules for parsing these fields and returning them in the client-compatible shape without silently discarding generated data.

Early versions should be conservative:

- preserve reasoning/thinking fields when present and supported by the client-facing response shape
- preserve user-facing content
- extract clear tool calls
- avoid stripping or filtering content as an implicit correction feature
- only replace reasoning with summarized reasoning if that explicit mode is enabled later

### 9. False refusal or tool denial

The model may say it cannot access tools even though tools are available.

Possible handling:

- retry with a reminder of available tools
- correction agent identifies whether a tool should have been used
- pass through if the response is otherwise acceptable

This may be a prompt-template issue rather than a model capability issue.

### 10. Suspicious finish reason

The upstream may report a finish reason that does not match the apparent response.

Examples:

- `stop` despite an incomplete structured output
- `length` after a partial tool call
- provider-specific “done” with no tool call
- no error despite parser failure upstream

The proxy should not blindly trust finish reasons. It should compare finish reason against content shape.

## Correction confidence

Every repair should have a confidence level.

Example correction metadata:

- action taken
- confidence
- reason
- original parse error
- repair method
- model profile
- whether a model-based correction was used
- whether the output was schema-valid after repair

The client does not necessarily need to see this in the normal response, but developers should be able to inspect it through logs, headers, or a trace UI.

## Repair policy hierarchy

A useful default hierarchy:

1. Preserve valid upstream output.
2. Repair syntax without changing meaning.
3. Repair schema shape when intent is clear.
4. Ask a correction agent when deterministic repair fails.
5. Retry or continue when the model clearly intended an action but failed to produce it.
6. Avoid fabricating tool calls when intent is ambiguous.
7. Expose enough trace data for developers to understand what happened.

The proxy should be helpful, not reckless.

## Retry and continue strategy

Retries are powerful but can become expensive or unstable.

Potential retry types:

- same prompt, same settings
- same prompt with lower temperature
- same prompt with stronger tool instruction
- continue the assistant response
- correction-agent-assisted reconstruction
- retry using a different upstream model

Recommended early policy:

- allow one lightweight retry or continue for obvious premature stops
- prefer deterministic repair for syntax issues
- avoid repeated retry loops
- log all retries
- make retry behavior model-profile-specific

## Streaming strategy

For the MVP, do not stream upstream tokens directly to the client.

The proxy needs to see the complete response before it can:

- parse tool calls
- detect malformed output
- repair JSON
- decide whether to retry
- decide whether to call a correction agent
- return a clean OpenAI-compatible message

Later streaming options:

- buffer upstream response, then stream corrected response quickly
- stream artificial progress events
- stream reasoning summaries while correction is happening
- stream normal content only when no tools are present
- support “strict reliability mode” vs “low-latency mode”

The advanced version could mimic ChatGPT or Claude-style summarized thinking, but that should come after the core correction pipeline works.

## Prompt optimization and datasets

The project should treat real failures as valuable training data.

Dataset examples:

- original client request
- available tools
- upstream model output
- parser failure
- corrected output
- final accepted tool call
- model profile
- correction strategy
- whether retry was required

This enables:

- prompt optimization per model
- repair prompt optimization
- judge prompt optimization
- correction-agent optimization
- regression tests for known failure modes
- model comparison
- tool-format comparison

This is where DSPy-style workflows, dsrs, and GEPA become central. The proxy can collect the exact examples needed to teach each model how to behave better.

Internal agents should be built as DSPy/dsrs-style agents where possible, not ad hoc prompts scattered through the codebase. Correction agents, judges, and retry/continue decision agents should have their own datasets, evaluation loops, and GEPA optimization paths. These are controlled agents inside the system, so they are the right place to apply the full DSPy/dsrs stack aggressively.

Current implementation note: prompt optimization is now split by layer. Correction-agent GEPA works on malformed-response repair datasets. Request-adapter GEPA works on exact OpenAI request traces, selected profile metadata, and explicit expected output labels. GEPA runs require a separate target model under test, use native Anthropic Claude Sonnet 4.6 by default for reflection/proposal and judging, and reject configurations where the reflection or judge model is the same as the target model. Labels are kept outside the reflected example payload and are available only to the judge, which returns generalized feedback for reflection. The trace harness can import Pi Mono and Hermes-style third-party traces into neutral scenarios, run small live structural checks through the proxy, and contribute only reviewed edge cases to request-adapter datasets. Fresh Sonnet-judged runs promoted Gemma append-only and Qwen regenerated-context as the active embedded request-adapter defaults, while the lower-scoring alternate history-format artifacts remain checked in for provenance and future comparisons.

## Model profiles as first-class product surface

Model-specific behavior should not be treated as a hack. It should be a core feature.

A model profile answers:

- How should tools be rendered?
- How strict should parsing be?
- What mistakes does this model commonly make?
- Does it handle XML well?
- Does it prefer JSON blocks?
- Does it over-narrate before tool use?
- Does it support multiple tool calls?
- Does it need examples?
- Does it need short instructions?
- Does it need verbose instructions?
- Should native tool calling be disabled?
- What repair prompt works best?
- What judge model should validate corrections?

This is the “stop forcing a square through a circular hole” principle turned into architecture.

## Application compatibility

The proxy should accept that client applications vary widely.

Some apps:

- send very long system prompts
- rely on OpenAI-specific behavior
- assume tool calls are native
- expect strict response shapes
- stream everything
- use special tool-choice settings
- use structured outputs
- send reasoning parameters
- rely on provider-specific quirks

The proxy should not assume the client prompt is well-designed for the target model. It should adapt around it where possible.

## Client transparency

There are two possible philosophies:

### Transparent by default

The client receives only a normal OpenAI-compatible response. Corrections are invisible unless debug mode is enabled.

This is best for drop-in compatibility.

### Observable by default

The client or developer can easily see that a correction happened.

This is best for debugging and trust.

Recommended approach:

- normal API responses remain compatible
- return a trace ID header by default
- keep correction metadata in local traces rather than normal client responses
- reserve richer debug headers or debug endpoints for explicit later debug modes

## Observability

The proxy should make failure modes visible.

Useful observability views:

- requests per model
- tool-call success rate
- repair rate
- retry rate
- correction-agent usage rate
- most common malformed JSON patterns
- most hallucinated tools
- average correction latency
- accepted vs rejected repairs
- model profile comparison
- app compatibility issues

This helps the project become empirical instead of anecdotal.

## Evaluation

The project needs regression tests based on real model failures.

Useful evaluation categories:

- valid native tool call passes through unchanged
- malformed JSON is repaired
- XML tool call is parsed correctly
- content tool call is converted to API tool call
- hallucinated tool is handled according to policy
- premature “let me do X” response triggers continue or retry
- content in reasoning is mapped or preserved without dropping generated data
- multiple tool calls are preserved
- invalid repair is rejected by schema validation
- ambiguous intent is not over-corrected

Key metrics:

- tool-call recovery rate
- false correction rate
- schema-valid final response rate
- latency overhead
- cost overhead
- retry frequency
- correction-agent frequency
- client-visible failure rate

The most important metric may be: “Did the agent loop continue when it otherwise would have stopped incorrectly?”

## Safety and correctness

The proxy must not become a source of hidden unsafe behavior.

In this document, safety is about preserving application intent and avoiding bad behavioral corrections. It is not a proposal to strip reasoning or filter generated content.

Risks:

- fabricating tool calls
- changing user intent
- dropping generated data during normalization
- overcorrecting ambiguous outputs
- making destructive tool calls more likely
- hiding model unreliability from developers
- silently mapping a hallucinated tool to a real tool

Mitigations:

- confidence scores
- conservative defaults
- schema validation
- correction traces
- strict policies for destructive tools
- configurable repair modes
- no silent semantic rewrites for low-confidence cases
- dataset-driven evaluation

## Policy modes

The proxy may eventually support different operating modes.

Correction opt-out decision:

- opt-out should be global configuration only in the first version
- do not add per-request correction opt-out yet
- do not make per-model profiles the primary opt-out surface, though profiles can still tune correction behavior
- deployments that need pure pass-through, deterministic-only repair, or full correction should choose that through global configuration

### Conservative mode

- syntax repair only
- no inferred tool calls
- no hallucinated tool mapping
- minimal retries

Best for high-risk tools.

### Balanced mode

- deterministic repair
- correction-agent fallback
- limited retries
- inferred tool calls only when confidence is high

Best default mode.

### Aggressive recovery mode

- infer intended tools more often
- retry or continue more often
- use correction agents liberally
- optimize for keeping the agent loop alive

Best for experimentation and low-risk environments.

## Role of correction agents

Correction agents should not be treated as magic. They should be specialized, constrained components.

Correction is a central reason this proxy can work, so it should not be deferred or treated as a minor add-on. The first version should include the full correction pathway, while making each expensive or opinionated step configurable so users can turn pieces off.

The early correction stack should support:

- deterministic repair
- schema-guided repair
- correction-agent repair
- retry or continue policies
- optional judge validation where configured

Default correction-agent model policy:

- use the same upstream model by default
- allow a global default correction model override
- allow per-model-profile correction model overrides
- implement internal correction/judge agents through the DSPy/dsrs stack where possible
- collect datasets and optimize those internal agents with GEPA over time

For the first version, live correction should happen synchronously in the request path. The proxy needs to return the corrected response before the client continues the agent loop. Asynchronous work can come later for evaluation, optimization, exports, and background analysis, but not as the primary live correction behavior.

This lets the proxy start with the “whole shebang” while still supporting conservative deployments that disable model-based correction, retries, judges, or other optional behaviors.

Inputs:

- available tools
- tool schemas
- bounded recent conversation window
- malformed model response
- parser error
- model profile
- desired output contract

Outputs:

- corrected tool call or content
- explanation for correction
- confidence
- whether correction was possible
- whether retry is recommended

Correction agents should be evaluated just like any other model behavior.

## Role of judges

Judges can validate correction quality.

Possible judge questions:

- Does the corrected response preserve the original model intent?
- Is the tool name valid?
- Do the arguments satisfy the schema?
- Was any required argument invented?
- Is the correction safe?
- Should the proxy retry instead of returning this?

Judges should be optional in the MVP because they add latency, but the architecture should leave room for them.

## Handling reasoning

Reasoning should be treated as part of the model response contract, not as a special content-filtering problem for this project to solve.

Default policy:

- parse reasoning/thinking fields alongside content and tool calls
- preserve and pass through reasoning data when the client-facing API shape supports it
- do not strip reasoning or content merely because it appeared in a reasoning field
- map or duplicate user-visible text into `content` only when needed for client compatibility
- allow provider-specific adapters

Artificial reasoning summaries can be a later feature, especially for streaming UX. If enabled, they should be an explicit configured output mode that still processes the original reasoning data rather than ignoring it.

Current decision:

- preserve and pass through native reasoning/thinking fields when supported in the first version

## How this differs from OpenRouter

OpenRouter primarily routes requests across providers and models.

This project would route and also actively adapt behavior.

The differentiator is not just “one API for many models.” It is:

- model-specific prompting
- tool-format adaptation
- response correction
- structured-output recovery
- replayable traces
- optimization loops
- empirical model profiles

It is closer to a behavioral compatibility runtime than a marketplace router.

## How this differs from normal structured-output libraries

Structured-output libraries usually live inside an application.

This project lives between arbitrary applications and arbitrary models.

That means it must:

- preserve external API compatibility
- work with prompts it did not author
- handle tools it did not design
- repair outputs for applications it does not control
- support many model families

That makes the proxy harder, but also much more generally useful.

## Suggested roadmap

### Phase 0: design validation

- write intent and architecture docs
- define target client API compatibility as OpenAI-compatible chat/completions
- define first upstream interface as generic OpenAI-compatible API
- define initial failure cases
- use proxy-owned tool rendering as the primary reliability path while keeping pass-through repair available

### Phase 1: non-streaming correction proxy

- OpenAI-compatible chat completions
- non-streaming only
- request normalization
- model profiles
- generic OpenAI-compatible upstream model calls
- response parsing
- minimal proxy-owned tool rendering
- pass-through repair mode for compatibility and comparison
- deterministic JSON repair
- basic tool extraction from content
- correction-agent fallback as a first-class configurable capability
- synchronous live correction only in the first version
- local JSONL trace logging

### Phase 2: expanded tool rendering and optimization

- richer OpenAI tool schema to XML/tool-text rendering
- model-specific tool prompt profiles
- parsing XML or tagged output back into OpenAI tool calls
- multiple tool-call support
- GEPA-optimized prompts per model
- profile-selectable DSRs history renderers
- explicit GEPA artifact promotion
- embedded built-in defaults generated from reviewed artifacts

### Phase 3: datasets and evaluation

- trace-to-dataset export
- trace-harness import of third-party agent traces
- replay harness
- regression suites
- correction quality metrics
- model profile comparison
- prompt optimization loops

### Phase 4: streaming and UX

- buffered streaming
- corrected response streaming
- progress events
- reasoning summaries
- low-latency vs reliability modes

### Phase 5: advanced compatibility

- app-specific adapters
- provider-specific quirks
- structured output support
- policy modes by tool risk
- admin/debug UI
- team-shared model profiles

## Architectural principles

### 1. Preserve the client contract

The client should receive a response it already understands.

### 2. Own the model contract

The proxy should not blindly trust upstream native tool behavior.

### 3. Prefer deterministic repair first

Do not call a correction model when a safe parser can fix the issue.

### 4. Make model differences explicit

Model-specific behavior should live in profiles, not scattered hacks.

### 5. Treat failures as data

Every repair-worthy response is a future training or optimization example.

### 6. Keep correction auditable

Developers should be able to understand exactly what the proxy changed.

### 7. Avoid reckless semantic invention

Repair format aggressively; repair meaning conservatively.

### 8. Design for iteration

The project will discover new edge cases over time. The architecture should make it easy to add detectors, parsers, policies, and profiles.

## Open questions to revisit before implementation

Current decisions from the first pass:

- MVP supports both pass-through repair and proxy-owned tool rendering, with proxy-owned rendering as the primary reliability path.
- The upstream interface starts as a generic OpenAI-compatible chat/completions API.
- The first trace format is local JSONL.
- The correction pipeline should be included early and fully, with configuration flags to disable optional pieces.
- Correction agents should see a bounded recent conversation window.
- Live correction should be synchronous only in the first version.
- Client-visible debug should default to a trace ID header only.
- Hallucinated tools should be judged by the correction agent: correct slight/near-obvious mistakes, but pass through tool names that are not close to any available tool.
- The default correction-agent model should be the same upstream model, with global and per-model-profile overrides.
- Internal agents should use the DSPy/dsrs stack where possible and have their own datasets and GEPA optimization loops.
- Missing required or incorrect tool arguments should be lenient by default and pass through when repair confidence is low, with stricter enforcement configurable.
- Multiple/parallel tool-call settings from the client should be strictly enforced.
- Local JSONL trace retention should be configurable, with the default being “do not delete.”
- Native reasoning/thinking fields should be preserved and passed through when supported.
- Correction opt-out should be global configuration only in the first version.

Remaining questions:

- No major open questions captured so far; additional questions should be added as implementation design exposes new trade-offs.

## Early success criteria

The project is working if:

- an existing OpenAI-compatible agent app can point at the proxy without code changes
- open-source models complete tool loops that previously stopped prematurely
- malformed tool calls are repaired into valid client-facing tool calls
- developers can inspect what was repaired and why
- real failures become replayable test cases
- model-specific profiles measurably improve reliability
- the proxy can prove that model capability was being hidden by brittle tool parsing

## Short version

This project should give open-source models a better behavioral interface to the existing agent ecosystem.

The proxy should not merely route requests. It should understand the contract the client expects, understand the model-specific ways that contract can fail, and repair the interaction before the application loop breaks.

The winning architecture is one where prompts, tools, parsers, correction agents, judges, traces, and optimization loops all reinforce each other over time.
