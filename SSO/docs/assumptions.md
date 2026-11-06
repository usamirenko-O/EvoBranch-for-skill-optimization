# Reproduction assumptions (Phases 1–3)

This file records implementation choices not pinned down by the paper
(arXiv:2607.28777v1). Phase 1 covers deterministic evidence aggregation;
Phase 2 adds the five LLM modules; Phase 3 adds generic execution and
orchestration. Selection policies and concrete benchmark harnesses remain
deferred. Each new REPRODUCTION_CHOICE is recorded below.

Nothing in this file was silently folded into "how SSO works" -- every item
is also flagged as `REPRODUCTION_CHOICE` at its point of use in the code.

## 1. Final ranking tie-break

**Where:** `sso/core/evidence.py::rank_clusters`

**What the paper says:** "The remaining clusters are ranked by descending
`|Q_c|`, with larger `E_c` and evidence from more distinct tasks used as
successive tie-breakers. Any remaining ties are resolved deterministically."

**Gap:** no mechanism is given for the final tie-break once `|Q_c|`, `E_c`,
and distinct-task-count are all equal.

**Choice made:** break remaining ties by ascending `cluster_id`
(lexicographic order on the string id). This is arbitrary but stable and
reproducible given the same cluster ids, which is all "deterministic"
requires.

**How to replace it:** `rank_clusters` isolates this in a single `sort` call
with `cluster_id` as the last key; swapping the tie-break rule (e.g. to
"first cluster encountered" or an explicit priority list) means changing
only that key, not the scoring logic in `comparison_evidence` /
`aggregate_evidence`.

## 2. Definition of "distinct contributing tasks"

**Where:** `sso/core/types.py::ClusterScore.contributing_tasks`,
`sso/core/evidence.py::aggregate_evidence`

**What the paper says:** the ranking rule names "evidence from more distinct
tasks" as a tie-breaker, implying some notion of how many different tasks
"vouched for" a cluster, but never formally defines it as an equation the
way `Q_c`/`E_c` are defined.

**Choice made:** a task `x_i` is counted as contributing to cluster `c` if
at least one of its decisions `y_i^k` assigned `c` nonzero evidence
(`e_ic^k != 0`). Multiple probes `k` for the same task count once.

**How to replace it:** the `contributing_tasks` set is built in one place
inside `aggregate_evidence`; an alternative definition (e.g. counting
probe-task pairs instead of tasks, or weighting by evidence magnitude) would
only require changing that accumulation step.

## 3. Missing activation entries default to UNCLEAR

**Where:** `sso/core/evidence.py::resolve_cluster_activation`

**What the paper says:** eq. 7 states the behavior extractor "assigns an
activation state to each input execution", implying every `BehaviorRecord`
should have an entry for every execution it was shown (the anchor and all
`K` probe executions for that task).

**Gap:** the paper doesn't say what a downstream consumer should do if a
record is missing an entry for some execution (e.g. malformed/partial
extractor output).

**Choice made:** treat a missing `ExecutionRef` key in
`BehaviorRecord.activations` the same as an explicit `Activation.UNCLEAR`
vote, rather than raising an error. This is a defensive engineering default
for malformed input, not an interpretation of the algorithm's intended
behavior on well-formed data -- on well-formed data (every record covering
every execution it saw) this branch is never exercised.

## 4. Representational choices with no numeric effect

**Where:** `sso/core/types.py`

`ExecutionRef` as a frozen dataclass keyed by `(task_id, is_anchor,
probe_index)`, `BehaviorRecord.activations` as a `dict[ExecutionRef,
Activation]`, and similar structural decisions are plain data-modeling
choices to make eq. 9-14 directly executable. They carry no algorithmic
assumption -- any structurally-equivalent representation would produce
identical `Q_c`/`E_c`/ranking outputs -- so they are noted here for
completeness but are not treated as REPRODUCTION_CHOICEs in the same sense
as items 1-3 above.

## 5. `BehaviorClusterer`'s id-only interface and `assemble_clusters` split

**Where:** `sso/llm/clusterer.py::BehaviorClusterer.cluster`, `ClusterAssignment`,
`assemble_clusters`

**What the paper says:** "The clusterer receives only the behavior
descriptions, without activation states or pairwise decisions" (Methodology,
"Behavior Extraction and Clustering").

**Choice made:** `cluster()` is typed to take only `Sequence[Tuple[str,
str]]` (`(record_id, description)` pairs) and to return `List[ClusterAssignment]`
(record ids only -- a representative id plus a member-id set). No
`BehaviorRecord`, `Activation`, or `PairwiseDecision` is reachable through
this signature, so an implementation structurally cannot condition its
clustering decision on activations or judge outcomes -- this is enforced by
the interface, not left to a docstring convention. A separate, deterministic
(non-LLM) function, `assemble_clusters`, then re-attaches the original
`BehaviorRecord` objects (which the caller already holds) to produce the
`BehaviorCluster` objects `sso/core/evidence.py` expects.

**REPRODUCTION_CHOICE specifics:**
- `record_id` as the bookkeeping key for the partition -- matches
  `BehaviorRecord.record_id`, carries no algorithmic meaning of its own.
- `BehaviorCluster.cluster_id` is set to `representative_record_id` in
  `assemble_clusters` -- stable and unique given unique `record_id`s, and
  needs no separate id-generation scheme.
- Splitting "decide the partition" (`cluster`) from "attach full records"
  (`assemble_clusters`) into two functions rather than one method that both
  clusters and assembles -- a pure engineering decomposition motivated by
  the information-barrier requirement above, with no numeric effect on the
  eventual `BehaviorCluster` contents.

**How to replace it:** if a future revision needs the clusterer to see more
than descriptions (the paper does not currently ask for this), widen
`cluster`'s parameter type and update this entry; `assemble_clusters` would
likely become unnecessary at that point since the clusterer could assemble
`BehaviorCluster` directly.

## 6. Paper requirements deferred by Phase 2a interfaces

**Where:** `sso/llm/judge.py`, `sso/llm/renderer.py`, `sso/llm/probe_generator.py`

The paper states several behavioral *requirements* that Phase 2a's abstract
interfaces could not enforce. At that stage no LLM calls existed. Each is recorded in the relevant
module's docstring as a deferred contract, listed here for visibility:

- `PairwiseJudge`: presentation-order randomization + identity remapping to
  mitigate position bias (eq. 5 discussion).
- `Renderer`: may add/strengthen an `ENCOURAGE`d behavior and remove/
  weaken/restrict a `SUPPRESS`ed one, but must never introduce an
  unsupported opposite (eq. 15 discussion).
- `ProbeGenerator`: each probe must be a complete skill distinct from `S_t`
  and from its peer probes (eq. 3 discussion).

None of these is a REPRODUCTION_CHOICE in the sense of items 1-5 above --
each is PAPER_SPECIFIED. Phase 2b now implements randomization/remapping,
textual probe distinctness checks, and prompts for all three contracts.
Semantic completeness and the prohibition on unsupported opposites remain
LLM instruction constraints, not properties a JSON parser can prove.

## Explicitly NOT assumed in Phase 1

Per review feedback, the following are called out so they are not mistaken
for settled choices:

- **Training-batch selection (`X_t ⊂ D_tr`)**: the paper only says "Select
  `X_t ⊂ D_tr`" (Algorithm 1, line 3) without specifying a policy. Phase 1
  contains no sampling code at all, so no policy (random or otherwise) has
  been chosen yet. This will be an explicit, separately documented
  REPRODUCTION_CHOICE in `sso/sampling.py` in a later phase.
- **History subset (`H_t`) and validation batch (`V_t`) selection**: same
  status as `X_t` -- open, deferred to the sampling phase.

## 7. Phase 2b client and provider configuration — REPRODUCTION_CHOICE

**Where:** `sso/llm/client.py`

The paper defines LLM functions, not a provider SDK or transport protocol.
We use a structural `LLMClient` protocol: one stateless synchronous request
with system and user strings returns response text. The optional
`ChatCompletionsClient` uses `urllib.request` and UTF-8 JSON with an explicit
full HTTP(S) endpoint and model; optional authentication is a Bearer token.
The installed environment has Python 3.14.4 and pytest; requests/httpx are
available but undeclared, and the OpenAI SDK is absent. The standard library
avoids introducing any dependency or relying on undeclared packages.

The adapter sends two messages, `stream=false`, and
`response_format={"type":"json_object"}`. No provider-side JSON Schema is
required. The endpoint must support this format. No default model or endpoint
is selected. The paper's reported optimizer model is GPT-5.5; choosing a
different model would be an explicit experimental deviation. Keep the judge
model and configuration fixed across comparisons. The library neither trains
models nor guarantees that a remote model alias remains pinned.

Temperature, top-p, token limits, reasoning settings, and provider seeds are
omitted, using endpoint defaults. This is a reproduction choice, not evidence
that defaults reproduce the paper's decoding settings. There is no automatic
context truncation, batching, tool calling, or conversation memory. Injected
clients must likewise avoid cross-call conversation state to preserve the
extractor/clusterer information barriers. Sharing a stateless transport is
safe; sharing an accumulated conversation is not.

## 8. Failure and retry policy — REPRODUCTION_CHOICE

**Where:** `client.py`, `_structured.py`, concrete subclasses

**Updated 2026-09-10** (phase1_fix_prompt.md §1.8/§2.2.3): a valid
invocation now makes up to `max_retries + 1` completion call attempts
(`max_retries` defaults to 2, caller-configurable at `ChatCompletionsClient`
construction) before raising -- a bounded, transport-level retry on a
transient failure (a truncated `finish_reason`, or an HTTP/network error),
not a repair prompt, fallback decision, or partial result. `max_retries=0`
reproduces the exact previous single-attempt behavior byte-for-byte. Every
attempt (successful or not) still goes through the same well-formedness
checks below; a failed attempt increments `usage_snapshot()["failed_attempts"]`
but never `["calls"]`, and never changes which response the caller
ultimately receives -- only whether a transient hiccup gets one more chance
before the caller sees `LLMClientError`. Timeout defaults to 120 seconds and
is caller-configurable, finite, and positive, independently per attempt.
HTTP/transport failures and invalid/refused/incomplete provider envelopes
raise `LLMClientError`; provider bodies and credentials are not included in
its message. The adapter accepts exactly one choice with
`finish_reason="stop"`, nonempty text, no refusal, and no tool/function call.
Provider metadata outside the consumed envelope fields is ignored.

Still true, unchanged by the above: no repair prompts, no fallback
decisions (e.g. "keep the current skill on failure" is NOT implemented --
a run that exhausts its retries still raises and the caller/driver decides
what to do), and no partial results.

Model-output schema failures raise `LLMResponseError` (a `ValueError`). Input
contract failures raise `ValueError` (wrong Python argument shapes may raise
`TypeError`). Errors propagate to the caller; no optimization decision follows
from a failure. No credentials are read or network requests made on import.

## 9. Prompts and strict JSON — REPRODUCTION_CHOICE

**Where:** each module's `_..._PROMPT` and `_structured.py`

All exact English prompt wording, JSON keys, document tags, and delimiters are
ours. Each prompt's adjacent comment distinguishes its PAPER_SPECIFIED
requirements from this implementation. Inputs use explicit field selection
and JSON serialization, with an instruction to treat embedded instructions
as data. No prompts use dataclass `__dict__` or implicit object serialization.
All requests are independent. This reduces accidental information leakage;
it is not a proof against prompt injection or misleading input content.

Responses must be single JSON documents. Module schemas reject extra/missing
keys, wrong containers/types, duplicate object keys at any depth, non-finite
JSON constants, Markdown fences, and trailing prose. There is no case folding,
numeric/string coercion, fence stripping, or substring extraction. Nonempty
strings are checked using `.strip()` but returned with original whitespace.
Nested schemas are checked before returning any SSO state.

| Module | Exact response shape (illustrative values) |
|---|---|
| Generator | `{"probes":[{"document_type":"complete_skill","content":"..."}]}` |
| Judge | `{"decision":"A"}`; value is exactly `A`, `B`, or `tie` |
| Extractor | `{"behaviors":[{"description":"...","activations":{"e0":"absent","e1":"present"}}]}`; keys cover all supplied execution aliases |
| Clusterer | `{"clusters":[{"representative_record_id":"r1","member_record_ids":["r1","r2"]}]}` |
| Renderer | `{"document_type":"complete_skill","content":"..."}` |

Generator and renderer explicitly request all retained instructions and reject
schema-level edit/diff responses or empty documents. The tag is an LLM claim,
not a semantic certificate: a prose edit mislabeled as a complete skill can
still pass. We add no language-specific heuristic, extra LLM verifier, or
arbitrary minimum length. The same limitation applies to behavioral accuracy,
semantic equivalence, and renderer direction fidelity.

## 10. Judge randomness and anonymization — REPRODUCTION_CHOICE

**Where:** `judge.py::LLMPairwiseJudge`

Randomized presentation and semantic remapping are PAPER_SPECIFIED (eq. 5).
The mechanism is ours: one independent `random.Random` instance per judge,
one uniform `randrange(2)` draw per call, optional RNG injection, no fixed
default seed. Draw 0 puts the probe at A; draw 1 puts it at B. A matching
probe win maps to +1, an anchor win to -1, and tie to 0. No repeated voting.
An injected seed reproduces ordering, not remote LLM sampling.

The payload contains only task content and A/B execution contents. It omits
task ids, probe indices, execution references, and current/experimental role
labels. The paper does not specify exact anonymization labels or a detailed
judging rubric in v1; our minimal rubric asks which execution better fulfills
the task. Original execution text is preserved even if it contains identity
hints. No external reference answer, score, or reward is added.

## 11. Generator bookkeeping and validation — REPRODUCTION_CHOICE

**Where:** `probe_generator.py::LLMProbeGenerator`

K must be a positive integer (booleans rejected). History entries must have
matching task/anchor references and unique task ids. Empty history and empty
current skill are permitted; no batch membership or selection policy is
invented. All selected pairs enter one request. The paper reports a no-skill
initial condition, so nonempty validation is only applied to generated skills.

Exactly K output documents are required. Probe indices are assigned 1..K in
response order; skill ids remain `None`. Duplicate detection compares content
after stripping outer whitespace, against the current skill and all peers.
No deduplication, replacement generation, or semantic similarity threshold is
introduced. Semantic distinctness beyond identical text remains a prompt
constraint. Contents themselves are returned unchanged.

## 12. Extractor aliases and record validation — REPRODUCTION_CHOICE

**Where:** `extractor.py::LLMBehaviorExtractor`

The anchor is `e0`; aliases `e1..` follow input probe order, then map back to
the actual `ExecutionRef`s. Nonconsecutive positive probe indices are preserved.
Task/role references must agree and probe references must be unique; at least
one probe is required. This method sees all probes supplied to it but cannot
detect probes omitted by its caller, since its reviewed interface has no K.

One request returns zero or more records. Every description is nonempty;
every activation map must exactly cover the shown executions. Values are
exactly the existing enum values `present`, `absent`, `unclear`. Missing states
are rejected here, rather than invoking Phase 1's defensive UNCLEAR fallback.
Record ids are generated as `task_id:j`, with j starting at 1 in output order;
they are unique within a batch with unique task ids, not across repeated calls
for the same task. Repeated descriptions are not silently merged or dropped.
The caller retains Algorithm 1's decisive-task filter; no outcomes enter the
extractor signature, prompt, or parser.

## 13. Cluster parsing and assembly — REPRODUCTION_CHOICE

**Where:** `clusterer.py::LLMBehaviorClusterer`

Only nonempty string ids and descriptions are serialized. Duplicate input ids
fail before a call. Returned clusters must cover each input id exactly once,
contain no unknown/duplicate members, and choose a member representative.
The partition and original-member requirement are PAPER_SPECIFIED; rejecting
malformed partitions rather than repairing them is our parsing policy.
No response summary field is accepted. Cluster order follows response order,
members become the existing `frozenset`, and existing assembly reattaches
original records without altering descriptions or activations. Member-list
iteration order in that unchanged helper is not guaranteed across processes;
membership and evidence aggregation do not depend on it.

An empty description input still makes one request and requires empty clusters.
No singleton fallback or semantic checker is added. The information barrier
applies to explicitly selected request fields; caller-supplied descriptions
and ids should not themselves embed hidden outcomes.

## 14. Renderer validation and evidence representation — REPRODUCTION_CHOICE

**Where:** `renderer.py::LLMRenderer`

An empty U_t raises `ValueError` before any call. This signals that Algorithm
1's caller must skip rendering; it does not keep/accept/reject a skill itself.
Duplicate cluster ids, neutral or sign-inconsistent intents, non-finite Q/E,
zero Q, and E less than |Q| are rejected. Ranked order is retained; no budget
or ranking is recomputed. The payload carries all existing BehaviorUnit fields:
cluster id, description, intent, Q, E, contributing-task count. Evidence for
and against is recoverable as (E+Q)/2 and (E-Q)/2; no new evidence is invented.

The returned Skill has the validated content and no generated skill id.
No lexical-change requirement relative to the current skill is imposed. The
renderer synthesizes from S_t and U_t, never selects a probe, and cannot check
against unavailable probe text. The prohibition on unsupported opposites is
in the prompt, not falsely claimed as a structural guarantee.

## 15. Executor and contextual references — REPRODUCTION_CHOICE

**Where:** `sso/executors/base.py`, `sso/core/_execution.py`,
`sso/core/validation.py`

The paper specifies f_exec(skill, task) -> observable execution (eq. 1), not
a Python interface or identity scheme. We add an ABC with exactly those two
inputs and the existing `Execution` output. No harness implementation is
provided. The executor returns a task-matching reference; an anchor reference
is a convenient placeholder because the executor cannot know the optimization
role. Orchestration creates a new Execution with the appropriate contextual
reference and exactly the original content. Wrong task ids, non-Execution
results, and non-text content fail visibly. Role/index metadata from the
executor is not used to infer optimization state.

Validation adapts the existing probe-shaped judge signature with a local
positive slot 1 for the candidate, while retaining the same executor and judge
instances used for training. This slot has no algorithmic meaning. A candidate
is never wrapped in a Probe or fed into training evidence. Validation stores
named current/candidate executions and y in `ValidationComparison`; training
decisions and validation results remain separate. Do not pool reference keys
across phases or steps without adding external step/phase context. No Phase 1
types or Phase 2 interfaces were redesigned for this bookkeeping bridge.

## 16. Explicit selections and empty inputs — REPRODUCTION_CHOICE

**Where:** `algorithm.py::StepInputs`, `SSOOptimizer`, `validation.py`

`StepInputs` requires X_t tasks, selected history task ids, and V_t tasks,
snapshotting each sequence as a tuple. Membership and order are supplied by
the caller, including empty history. H_t pairs are assembled only after this
step's anchor executions exist, so no cached/stale execution history is used.
This implements precomputed history membership; a policy depending on observed
content is deferred rather than invented. All selections are outside the
optimizer; no RNG, default sample size, shuffle, or fallback selection exists.

Task ids are nonempty strings, unique within each batch; history ids are a
unique subset of training ids. Dataset disjointness is PAPER_SPECIFIED in the
Problem Setup. Its enforcement by task-id intersection is our representation
choice: a step rejects X_t/V_t overlap, and the loop checks all supplied
training ids against all supplied validation ids before executing anything.
The caller must assign consistent identities and ensure disjointness for
dataset instances beyond the provided schedule. Repeated tasks within one
split across different steps are allowed. Separate manual calls to `step`
have no global split registry.

K must be a positive integer and U a nonnegative integer (booleans rejected).
They have no defaults in the optimizer. Empty batches are not replaced or
resampled: empty X_t still goes through generation and empty clustering; a
well-formed empty partition yields empty U_t. U=0 likewise yields no update
through the existing evidence budget. Empty V_t yields W=L=0 and rejects
under the existing strict W>L rule. No additional acceptance threshold is
introduced. Empty U_t skips validation regardless of the supplied V_t.

## 17. Scheduling, contracts, and inspection — REPRODUCTION_CHOICE

**Where:** `algorithm.py::SSOOptimizer.step`, `_execution.py`, `validation.py`

Operations are sequential in input task order and generated probe order. Each
probe is executed on the whole training batch before its comparisons, matching
Algorithm 1 lines 8–9. Anchors are reused for all training comparisons and
history within a step. Validation executes current then candidate per task;
the judge still independently randomizes presentation as in Phase 2. No
execution cache is shared across steps or between training and validation.

For injected components, reject K/count/index mismatches (indices must be
exactly 1..K, each once), misidentified or non-integer judge decisions,
cross-task/duplicate behavior record ids, and invalid/overlapping/incomplete
cluster partitions. These guards prevent silent misrouting; they do not
renumber probes, repair records, or add evidence. Skill semantics and behavior
semantics remain component responsibilities. Activation resolution remains
unchanged in evidence.py, including its defensive missing-state behavior.
Clustering is still invoked on an empty description sequence, as in Algorithm
1 and the Phase 2 interface; no extra early-exit policy is inserted.

`StepResult` retains anchors, probes, probe executions, training decisions,
records, assembled clusters, units, candidate, validation, and selected skill.
`candidate_skill` and `validation` are None only on the empty-unit branch.
`ValidationResult` retains per-task executions/outcomes, W, L, ties, acceptance,
and selected skill. These dataclass/container shapes, retention of traces, and
explicit tie counts are engineering choices. Existing mutable records/maps
are not deep-copied: results are inspection objects, not immutable audit logs
or checkpoints. Identical skill text does not bypass rendering/validation or
override the judge's decision.

Exceptions propagate without retries or an invented retain-current fallback.
A failed call returns no step result; side effects of a caller's harness are
not rolled back. Nothing is persisted, charged, benchmark-scored, or evaluated
against test labels by orchestration.

## 18. Minimal outer loop — REPRODUCTION_CHOICE

**Where:** `algorithm.py::SSOOptimizer.optimize`

The loop consumes an explicit finite sequence of StepInputs, snapshots the
schedule, and uses T=len(schedule). Each step receives the previous selected
skill. It returns S_T plus the per-step results; no best-of-history selection,
early stopping, adaptive sampling, or additional callbacks are introduced.
An empty schedule returns the initial skill and an empty trace. The same
caller-owned component instances and K/U are reused throughout. Dynamic
selection can be managed externally with repeated calls to `step`, with the
caller responsible for global dataset separation. OfficeQA and every other
benchmark-specific selection/execution policy remain deferred.
