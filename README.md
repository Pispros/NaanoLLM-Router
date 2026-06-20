# Naanollm-Router

A tiny, single-binary Go proxy that exposes **one OpenAI-compatible endpoint** and
internally routes each turn to a **planner** or a **coder** model. It keeps each
role's conversation state in SQLite, hands the structured plan from planner to
coder, and folds the coder's self-emitted summary back into the planner — so a
single chat thread "switches models" with nothing to configure on the IDE side.

The router is **engine-agnostic**: it speaks plain OpenAI HTTP to whatever serves
each role. Planner, coder and the tiny turn-router can each run on a different
backend — **llama.cpp, vLLM, SGLang, Text Generation Inference, or Ollama** — and
you can launch and supervise them straight from the control panel, or point the
router at servers you already run.

Point any IDE (Cline, Roo, Zed, Continue, Cursor, Aider, opencode, JetBrains AI
Assistant…) at `http://localhost:4000/v1` with model `auto`. The IDE thinks it's
talking to one model; it's talking to the router.

## The one rule that makes it work

**SQLite stores TEXT + METADATA. It never stores the KV cache.**

- Context (messages, the plan, summaries) → SQLite. Small, queryable, durable, the
  source of truth, and the same regardless of which engine serves a role.
- KV cache → the engine's concern, never the router's. The router only ever moves
  *text* between roles, never KV.

A KV cache is a disposable accelerator bound to *one model + one quant*; it is never
shared between models. If a cache is lost, it's rebuilt by re-prefilling from SQLite.
How the cache is kept warm depends on the engine (see *Warm context across turns*
below), but the router's contract is identical everywhere: persist text, forward a
request, stream the reply.

## How a conversation is identified

This is the part that has to be right, because every IDE injects volatile context
differently. The router maps each incoming request back to its conversation by
**matching the assistant transcript**, not by hashing the prompt.

Why not hash the prompt: the system prompt and the first user message are full of
content that changes every single turn — working directory, timestamps, open
tabs, `<environment_details>` blocks, project structure. Hashing any of that mints
a brand-new "discussion" on almost every turn, which fragments one chat into many
and resets all per-conversation state (so the planner→coder handoff dies). The fix
removes prompt content from identity entirely.

What is stable across turns, on every client, is the **assistant transcript**: the
IDE resends the full message list each turn (clients are stateless) and echoes
back each prior assistant message verbatim. So identity is anchored there:

- Each assistant turn gets a fingerprint = `hash(normalized text + canonical tool_calls)`.
  Reasoning blocks (`<think>…`) are stripped on both sides, since some clients drop
  them from re-sent history. Tool-call ids (server-generated) are excluded; the
  function name + canonicalized arguments are kept.
- A request continues an existing discussion when that discussion's recorded
  fingerprint sequence is a **prefix** of the request's — i.e. same conversation,
  one more turn. The longest such match wins.
- The first turn (no assistant messages yet) is de-duplicated on a normalized
  first-user-message anchor (so a retried first turn reuses the same discussion),
  otherwise a new discussion is created.
- When the client sends a stable `prompt_cache_key` (Zed mints a fixed UUID per
  thread), it is used as the primary discussion key — the most reliable signal of
  all, since it is content-free and collision-free.

Because tool_calls are part of the fingerprint, this works for agentic clients
whose turns are pure tool calls with little or no text (Cline, Roo Code, JetBrains
Junie/AI Assistant, …), not just plain chat clients. The system prompt, the tools
array, and user text never affect identity, so mutating them between turns is
harmless. Turns are persisted **synchronously** at the end of each request, so turn
*N* is on record before turn *N+1* arrives to be matched against it.

## Turn routing: one model decides, every turn

There is exactly **one routing brain: a tiny GBNF-constrained instruct model**
that classifies every turn as `planner` or `coder`. There is no "rule vs llm"
mode to choose — the model always routes. Keyword rules exist only as an
emergency fallback (see below). The router model is itself just another upstream,
so it can run on any of the supported engines.

On each genuine user turn the router asks the tiny model for a single word,
giving it three signals:

- `pending_plan` — whether an approved plan is waiting to be executed.
- `last_assistant` — a bounded (≤400-char) tail of the previous assistant
  message, so it can tell "the planner just proposed a plan, now run it" from a
  fresh request.
- the current user message.

A GBNF grammar (`root ::= "planner" | "coder"`) makes it physically unable to
answer anything else, and the reply is parsed leniently (case-insensitive, tolerant
of a stray `<think>…</think>` preamble).

Three guards keep the decision coherent — none of them inspect user keywords;
they encode structural facts about how planner and coder relate:

1. **An explicit model alias wins.** If the caller asks for `planner` or `coder`
   directly (e.g. Cline/Roo Plan/Act mapped to a role), that is honored as-is.
2. **No pending plan ⇒ planner only.** The coder exists *only* to execute an
   existing plan; with no plan there is nothing to execute, so `coder` is not a
   valid answer. In that case the grammar is restricted to `planner` alone. The
   model still makes the real planner/coder choice whenever a plan is pending.
3. **Tool-loop continuations stay on the coder.** Within one agentic turn an IDE
   sends *many* requests — one per tool round — feeding tool results back for the
   model to continue. Re-classifying each round would oscillate between coder and
   planner and loop forever (the planner, which has its tools stripped, would
   re-emit the plan mid-execution). So a request that is a continuation (a tool
   result, or an assistant `tool_calls`, appearing after the newest user message)
   is **not** re-routed: only the coder has tools, so a continuation always belongs
   to the coder finishing its work. The model is consulted **once per real user
   turn**, not on every micro-round.

The planner has its `tools`/`tool_choice` stripped so a weak model can't wander
off calling tools instead of producing the plan; the coder keeps its full tool
set. Per-role sampling is forced too (planner near-deterministic at
`temperature 0`, coder at `0.15`).

### Fallback (only when the model can't answer)

If the router model is unreachable, times out, errors, or returns something that
isn't a role, the router falls back to a minimal keyword rule: *pending plan + an
execute-style message → coder, otherwise planner* (never mutate files on a guess).
This is **not** a configurable mode — it is a safety net, and it is made visible:
the response carries `X-LLMRouter-Fallback: 1`, `/admin/status` reports
`last_route_fallback: true`, the dashboard **Router** card flips from green
*"Model routing every turn"* to an amber *"Model unavailable — using keyword
fallback"*, and a `WARN … used keyword fallback` line is logged.

> **The router model must be a generative instruct model**, not an embedding
> model. An embedding model (e.g. `Qwen3-Embedding-0.6B`) can't follow the
> "answer planner or coder" instruction and every turn silently degrades to the
> fallback. Use a generative model such as `Qwen3-0.6B` (the **Use recommended**
> button fills it in). It only ever emits one word, so a sub-1B model keeps
> routing near-instant on any engine.

## Architecture

```
IDE (base_url -> :4000/v1, model=auto)
        │
        ▼
  llmrouter (this binary)            tiny router model (any engine)
    • resolve discussion ───────────▶ classify planner|coder, once per real turn
      (prompt_cache_key, else            (explicit alias wins · no plan ⇒ planner
       assistant-transcript match)        · tool-continuation ⇒ coder
    • build append-only prompt:           · keyword fallback only if it fails)
      [system+contract] + [per-role history] + [new input]
    • forward over OpenAI HTTP (tool_calls relayed verbatim)
    • persist text/plan/summary to SQLite (synchronously)
        │                              │
        ▼                              ▼
  inference server (planner)     inference server (coder)
   llama.cpp | vLLM | SGLang |    llama.cpp | vLLM | SGLang |
   TGI | Ollama                   TGI | Ollama
```

Each role is an independent OpenAI-compatible endpoint. The router doesn't care
which engine answers; it picks the upstream by the role's configured base URL and
model alias, and adapts the request to that engine automatically.

## Warm context across turns

Append-only discipline keeps each role's prefix stable across turns, so an engine
that can reuse a prefilled prefix only has to re-prefill the small new suffix.
How that reuse happens is the one place engines differ:

- **llama.cpp** — the router pins a **warm KV slot per (role, discussion)** with
  `id_slot` + `cache_prompt`, so switching back to a conversation is instant. With
  `--slot-save-path` set, evicted slots are parked to `.bin` files and restored on
  the next turn.
- **vLLM / SGLang / TGI / Ollama** — these batch and cache internally; the router
  detects a non-llama.cpp engine and simply omits the slot fields, letting the
  engine manage concurrency and prefix reuse its own way.

Either way the router never depends on the cache: if it's cold or gone, the prefix
is rebuilt from the SQLite text. The cache is an accelerator, not state.

## Parallel discussions

Every conversation (identified as above) is an independent flow with its own
per-role context in SQLite, so any number of chats run side by side regardless of
engine. The **Slots** value you set per role is a concurrency hint that maps to
each engine's own flag — `-np` (llama.cpp), `--max-num-seqs` (vLLM),
`--max-concurrent-requests` (TGI), or the engine default. On llama.cpp it also
sizes the warm-slot pool: llmrouter assigns one slot per active (role, discussion)
and evicts the least-recently-used one when there are more active discussions than
slots (its KV parked to a `.bin` if a save path is set, otherwise rebuilt from
text on its next turn). Set **Slots** to how many discussions you want served — and,
on llama.cpp, kept warm — at once.

## Build

Requires Go 1.21 or newer. The SQLite driver is pure Go (`modernc.org/sqlite`),
so the result is a self-contained binary — no CGO, no system sqlite.

```bash
cd llmrouter
go mod tidy                # fetches the sqlite driver, writes go.sum (needs network once)
go build -o naanollm-router .
./naanollm-router                # control panel + /v1 on :4000; SQLite DB auto-created
./naanollm-router -data ./state  # optional: keep the db in ./state
```

An existing database from an older build is **migrated automatically** on
startup (the discussions table is rebuilt to the transcript-based shape and a
per-turn fingerprint column is added); no manual step and no data loss. Rows
created before the upgrade have no fingerprints, so their conversations start a
fresh discussion on their next turn — expected, and harmless.

> Upgrading from a build that had a `router.mode` field? It is simply ignored now
> (there is no mode), so your existing `llmrouter.json` keeps working untouched.

### If `go` tries to download a newer toolchain

On an older Go, a higher `go` directive makes the tool fetch a matching
toolchain and fail offline with `toolchain not available`. Pin your local
toolchain and, if needed, lower the directive to your installed version:

```bash
go env -w GOTOOLCHAIN=local   # never auto-download a toolchain
go version                    # confirm it responds, e.g. go1.21.5
go mod edit -go=1.21.0        # match the go.mod directive to your Go (use a full x.y.0)
go mod tidy && go build -o naanollm-router .
```

`go mod tidy` still needs network the first time to pull `modernc.org/sqlite`
from the module proxy (separate from the toolchain). If you're fully offline,
set `GOPROXY` to a local mirror, or ask for the JSON-file persistence variant
that drops the external dependency entirely (stdlib-only binary).

## Run the models from the control panel

You don't start backends by hand. naanollm-router launches and supervises one
inference server per role from the **Engine** tab (planner, coder) and the
**Router** tab (the tiny turn-router). Each role independently picks any of five
engines — the router talks OpenAI HTTP to all of them, so they're interchangeable:

| Engine | Binary you point to | Model source | Notes |
|--------|---------------------|--------------|-------|
| **llama.cpp** | `llama-server` | GGUF (downloaded via `--hf-repo`/`--hf-file`, or a local file) | also gets warm per-discussion KV slots |
| **vLLM** | `vllm` | HF repo id (`vllm serve <repo>`) | high-throughput GPU server |
| **SGLang** | your Python interpreter | HF repo id (`python -m sglang.launch_server`) | "binary" = the venv's `python` |
| **Text Generation Inference** | `text-generation-launcher` | HF repo id | HuggingFace TGI |
| **Ollama** | `ollama` | a tag you've `ollama pull`-ed | daemon bound per role port |

You install or build each engine natively yourself, then paste the **absolute
path to its binary** in the tab. Workflow per role:

1. Pick the engine and paste the binary path, then click **Test** (the tooltip
   shows the per-OS command to locate it; the install link points to that engine's
   docs).
2. Choose the model: for HF-based engines, search Hugging Face and pick a
   repository (and, for llama.cpp, a quant `.gguf` file); for Ollama, type the
   pulled tag.
3. Click **Launch**. Alias, port and concurrency (**Slots**) come from the
   **Endpoints** / **Router** tabs.
4. Tick **Auto-start everything when the server boots** and **Save config** to
   relaunch every role and arm `/v1` automatically next time.

The **Router** tab uses the same picker for the tiny turn-router; the **Use
recommended** button pre-fills a generative `Qwen3-0.6B` on llama.cpp (make sure
it's the generative model, not the `-Embedding-` variant — see the note above).
With autostart on, the router model is launched at boot alongside planner and
coder. Managed processes are stopped on Ctrl-C / SIGTERM.

### Bring your own servers

The panel is the easy path, not a requirement. Because every role is just a base
URL + model alias, you can point any role at an OpenAI-compatible server you
already run (llama.cpp, vLLM, SGLang, TGI, Ollama, `llama-swap`, a remote box…).
Leave the binary path empty, set the role's base URL and model alias on the
**Endpoints** / **Router** tabs, and **Start server** — llmrouter health-checks
the URL and routes to it without managing the process. The warm-slot optimization
still kicks in automatically when that endpoint is a llama.cpp server.

## Configure & start

Open the control panel at `http://localhost:4000/`:

1. Set the planner, coder, and router base URLs, model aliases, and concurrency
   (**Slots**).
2. Optionally set a slot save path (llama.cpp only; must match its
   `--slot-save-path`).
3. Click **Save endpoints** / **Save router**, then **Start server**. Start
   health-checks the upstreams before arming `/v1`.

The rail lights up planner or coder as turns are routed; status shows reachability,
the last route, whether it used the fallback, and the discussion count.

While autostart (or a managed engine launch) is in progress, the UI shows a
distinct **starting** state instead of a flat "stopped": the sidebar status pill
pulses *starting…*, the Engine tab shows an auto-start banner, and each dashboard
card spins on *loading* rather than the alarming red *offline* until its health
endpoint answers. During that window `/v1` returns `503` with a *"server is
starting"* message and a `Retry-After` header instead of *"not started"*.

The **?** button in the top-right opens an in-app help chat: it answers questions
about naanollm-router by streaming from your configured planner model, with this
app's documentation supplied as context (so the planner must be reachable).

## Wire an IDE

- Provider: OpenAI-compatible
- Base URL: `http://localhost:4000/v1`
- API key: anything (ignored locally)
- Model: `auto` (or force `planner` / `coder`)

In Cline/Roo you can also let Plan/Act map to `planner`/`coder` explicitly; with
`auto`, the tiny router model decides each turn (with `pending_plan` and the last
assistant turn as context).

### IDE compatibility notes

Identity is anchored on the assistant transcript including tool_calls, so both
plain-chat clients and native-tool-calling agents are handled the same way:

- **Cline / Roo Code / Continue / Zed / Aider / opencode** — full agent traffic
  (native OpenAI tool calling) flows through the router and stays on one
  discussion across turns.
- **JetBrains AI Assistant / Junie** — works via the IDE's OpenAI-compatible /
  BYOK endpoint; point it at `:4000/v1`.
- **Cursor** — the custom OpenAI base URL is only honored by Cursor's chat / plan
  panel (Cmd/Ctrl+L). Composer, inline edit and autocomplete are locked to
  Cursor's own backend and will **not** route here. That's a Cursor product
  limitation, not a router limitation.

Anything that resends the conversation each turn as standard OpenAI messages
(the normal stateless behavior) is compatible — the router never depends on the
system prompt, the tools array, or user-message wrappers staying constant.

## The handoff contract

- Planner emits its plan as a fenced ```` ```plan {json} ``` ```` block; llmrouter
  stores it and dispatches the scoped subtask to the coder (never the full reasoning).
- Coder ends its final message with a line `FINISH: <2-3 line summary>`; llmrouter
  captures it and appends it to the planner's context on the next planner turn.

Both contracts are folded into the **single leading system message**, alongside
the IDE's own system prompt. They are deliberately *not* added as a separate
system message: many chat templates (Qwen3.x and others) reject any `system`
message that isn't the very first entry, and a second one makes the engine refuse
the request with *"System message must be at the beginning."* Coder result notes
folded back to the planner are added as `user` notes for the same reason.

## Troubleshooting & observability

Quick notes from real debugging sessions:

- **Every turn falls back to keywords / the coder never fires.** The router
  endpoint is almost certainly serving the wrong kind of model. With
  `LLMROUTER_DEBUG=1` you'll see `router classify: reply "…" is neither planner
  nor coder`. Point the Router model at a **generative** model (e.g. `Qwen3-0.6B`),
  not an `-Embedding-` variant, and relaunch it.
- **The coder loops, re-emitting the plan and restarting.** That's mid-turn
  re-routing — handled by the tool-continuation guard (a continuation stays on the
  coder). If you still see it, confirm you rebuilt; with `LLMROUTER_DEBUG=1` a
  continuation should produce **no** `router classify` line at all.
- **A "create a plan" first turn went to the coder.** Shouldn't happen: with no
  pending plan the grammar only allows `planner`. If it does, you're on an older build.
- **Watch routing live.** `LLMROUTER_DEBUG=1` logs each decision
  (`router classify: model=… reply="coder" -> "coder"`), plus `REQ …`
  (`ephemeral=`, `tools=`, `max_tokens=`), `ROUTE …`, and `FWD -> …`.
- **Errors are surfaced, not swallowed.** When an upstream call fails, the router
  logs it (`ERR forward …` / `FWD <- status=…` with the body) and, in streaming
  mode, emits an SSE `error` chunk so the IDE shows the reason instead of a
  silently cut, empty stream.
- ***"System message must be at the beginning"* (HTTP 400).** The model's chat
  template requires a single leading `system` message. The router merges the IDE's
  system prompt and its role contract into one — but **existing discussions in
  `llmrouter.db` keep their old layout**, so delete/rename the DB (or clear
  `model_contexts`) after upgrading to clear stale prefixes.
- **A llama.cpp slot is created but the task never starts.** Usually slot
  contention: the pinned real turn and an unpinned utility request (IDE
  thread-title generation) land on the same server. Make sure that role's
  **Slots** matches the server's `-np`.
- **Reproduce without the IDE.** `curl -N …/v1/chat/completions` with
  `"stream":true` and, if needed, a `tools` array isolates whether the problem is
  the router pipeline or something the IDE adds to the payload.

## Files

```
main.go      entrypoint, one listener for /v1 + control panel, auto-creates/migrates the DB,
             always launches the router model at autostart
config.go    config types, JSON load/save (planner, coder, router — no "mode" field)
store.go     SQLite: discussions, per-role context, plans, summaries (text only);
             prompt_cache_key / transcript-match resolution + per-turn fingerprints + auto-migration
openai.go    OpenAI request/response types (incl. prompt_cache_key)
engines.go   engine adapters (llama.cpp, vLLM, SGLang, TGI, Ollama): build the launch
             command, health path, GGUF awareness; flags whether a role gets KV slot pinning
router.go    routing brain: tiny GBNF model classifies every turn; pending-plan grammar
             guard; explicit-alias and tool-continuation guards; keyword fallback on failure
prompt.go    conversation signature (transcript + tool_calls), last-user / last-assistant /
             tool-continuation detection, append-only prompt building, plan/summary parsing
upstream.go  OpenAI-HTTP client: adds id_slot/cache_prompt only for llama.cpp, streaming tee
             that reassembles tool_calls, slot save/restore; logs the request and any non-200 body
proxy.go     /v1 handler: resolve -> route -> build -> (llama.cpp: assign slot) -> forward -> persist;
             X-LLMRouter-Role/-Fallback headers, surfaces upstream errors (SSE error chunk),
             starting/running/stopped lifecycle phase
manager.go   launches & supervises managed engine processes per role
admin.go     control-panel API + embedded UI (status reports last_route + last_route_fallback)
hf.go        Hugging Face model/file search for the Engine tab pickers
slots.go     llama.cpp slot pool: maps each (role, discussion) to a slot, LRU eviction
web/index.html  the control panel (sidebar dashboard; animated routing conduit; Router card
             shows model-routing vs keyword-fallback state)
```
