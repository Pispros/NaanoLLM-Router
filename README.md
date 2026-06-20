# Naanollm-Router

A tiny, single-binary Go proxy that exposes **one OpenAI-compatible endpoint** and
internally routes each turn to a **planner** or a **coder** model. It keeps each
model's KV cache warm in its own llama.cpp slot, hands the structured plan from
planner to coder, and folds the coder's self-emitted summary back into the
planner — so a single chat thread "switches models" with nothing to configure on
the IDE side.

Point any IDE (Cline, Roo, Zed, Continue, Cursor, Aider, opencode, JetBrains AI
Assistant…) at `http://localhost:4000/v1` with model `auto`. The IDE thinks it's
talking to one model; it's talking to the router.

## The one rule that makes it work

**SQLite stores TEXT + METADATA. It never stores the KV cache.**

- Context (messages, the plan, summaries) → SQLite. Small, queryable, durable, the source of truth.
- KV cache → llama.cpp **hot slots** (RAM/VRAM), or optional `.bin` files on disk via `--slot-save-path`.

A KV cache is a disposable accelerator bound to *one model + one quant*; it is never
shared between models. If a cache is lost, it's rebuilt by re-prefilling from SQLite.
The router moves *text* between models, never KV.

## How a conversation is identified

This is the part that has to be right, because every IDE injects volatile context
differently. The router maps each incoming request back to its conversation by
**matching the assistant transcript**, not by hashing the prompt.

Why not hash the prompt: the system prompt and the first user message are full of
content that changes every single turn — working directory, timestamps, open
tabs, `<environment_details>` blocks, project structure. Hashing any of that mints
a brand-new "discussion" on almost every turn, which fragments one chat into many,
resets all per-conversation state (so the planner→coder handoff dies), and lets
llama.cpp silently graft a stale slot's prefix onto a prompt that has lost its
history. The fix removes prompt content from identity entirely.

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
harmless.

Turns are persisted **synchronously** at the end of each request, so turn *N* is
on record before turn *N+1* arrives to be matched against it.

## Turn routing: one model decides, every turn

There is exactly **one routing brain: a tiny GBNF-constrained instruct model**
that classifies every turn as `planner` or `coder`. There is no "rule vs llm"
mode to choose — the model always routes. Keyword rules exist only as an
emergency fallback (see below).

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

- the response carries `X-LLMRouter-Fallback: 1`,
- `/admin/status` reports `last_route_fallback: true`,
- the dashboard **Router** card flips from green *“Model routing every turn”* to an
  amber *“Model unavailable — using keyword fallback”*,
- a `WARN … used keyword fallback` line is logged.

> **The router model must be a generative instruct model**, not an embedding
> model. An embedding model (e.g. `Qwen3-Embedding-0.6B`) can't follow the
> "answer planner or coder" instruction and every turn silently degrades to the
> fallback. Use the generative `Qwen3-0.6B` (the **Use recommended** button fills
> it in).

## Architecture

```
IDE (base_url -> :4000/v1, model=auto)
        │
        ▼
  llmrouter (this binary)
    • resolve discussion (prompt_cache_key, else assistant-transcript match)
    • route turn: tiny GBNF model classifies planner|coder every turn
        (explicit alias wins · no plan ⇒ planner · tool-continuation ⇒ coder
         · keyword fallback only if the model fails)
    • build append-only prompt: [system+contract] + [per-role history] + [new input]
    • pin the role's warm slot (id_slot), cache_prompt:true
    • stream straight through (tool_calls relayed verbatim)
    • persist text/plan/summary to SQLite (synchronously)
        │                         │
        ▼                         ▼
  llama-server (planner)     llama-server (coder)
   -np N : one slot per        -np N : one slot per
   active discussion           active discussion
            ▲
            │  tiny router model (GBNF), consulted once per real user turn
        llama-server (router)
```

Append-only discipline keeps each model's prefix stable, so llama.cpp reuses the
warm KV and only re-prefills the small new suffix.

## Parallel discussions

Every conversation (identified as above) is an independent flow with its own
per-role context in SQLite. At runtime, llmrouter spreads each upstream's `-np N`
slots across the active discussions: one slot per (role, discussion). When more
discussions are active than there are slots, the least-recently-used one is
evicted — its KV parked to a `.bin` if a slot save path is set, otherwise simply
rebuilt from SQLite text on its next turn. Set `Slots` (the `-np` value) to how
many discussions you want warm at once.

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

## Managed engines (launch models from the panel)

You don't have to start the backends by hand. naanollm-router can launch and
supervise an inference server per role from the **Engine** tab. Each role
(planner, coder, router) independently picks one of five engines:

| Engine | Binary you point to | Models |
|--------|---------------------|--------|
| **llama.cpp** | `llama-server` | GGUF (downloaded via `--hf-repo`/`--hf-file`) |
| **vLLM** | `vllm` | HF repo id (`vllm serve <repo>`) |
| **SGLang** | your Python interpreter | HF repo id (`python -m sglang.launch_server`) |
| **Text Generation Inference** | `text-generation-launcher` | HF repo id |
| **Ollama** | `ollama` | Ollama tags (daemon bound per port; `ollama pull` first) |

You install or build each engine natively yourself, then paste the **absolute
path to its binary** in the Engine tab. For SGLang the "binary" is the Python
interpreter of the env where SGLang is installed (the adapter adds
`-m sglang.launch_server`).

Workflow per role:

1. Pick the engine, paste the binary path, click **Test** (the tooltip shows the
   per-OS command to locate it; the install link points to that engine's docs).
2. For HF-based engines, search Hugging Face and pick a repository (and, for
   llama.cpp, a quant `.gguf` file). For Ollama, type the model tag.
3. Click **Launch**. Alias, port and parallel slots come from the **Endpoints**
   tab.
4. Tick **Auto-start everything when the server boots** and **Save config** to
   relaunch every role and arm `/v1` automatically next time.

Only llama.cpp gets per-discussion KV slot pinning (`id_slot`/`cache_prompt`);
the other engines batch internally, so naanollm-router skips those fields for
them automatically. Managed processes are stopped on Ctrl-C / SIGTERM.

The **router model** is loaded with the same engine picker on the **Router** tab.
It classifies every turn, so a tiny instruct model is plenty — the **Use
recommended** button pre-fills `Qwen3-0.6B` on llama.cpp. (Make sure it's the
generative model, not `Qwen3-Embedding-0.6B`.) With autostart on, the router
model is launched at boot alongside planner and coder.

The manual `llama-server` invocations below still work if you prefer to run the
backends yourself — the Engine tab is optional.

## Run the three models (llama-server)

Keep them resident so switching is instant. Dedicate a slot per process and
enable slot persistence if you want caches to survive a reload.

Planner — Qwen3.5-9B (hybrid/SWA, so add `--swa-full` for correct slot restore):

```bash
llama-server -m qwen3.5-9b-UD-Q4_K_XL.gguf \
  --alias qwen3.5-9b -ngl 99 -fa on \
  --ctx-size 65536 -np 4 --slot-save-path ./kv \
  --swa-full \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --jinja --port 8080
```

Coder — Qwen2.5-Coder 7B (standard transformer, clean prefix reuse):

```bash
llama-server -m qwen2.5-coder-7b-Q4_K_M.gguf \
  --alias qwen2.5-coder-7b -ngl 99 -fa on \
  --ctx-size 65536 -np 4 --slot-save-path ./kv \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --port 8081
```

Router — tiny **generative** instruct model (required; it routes every turn):

```bash
llama-server -m Qwen3-0.6B-Q4_K_M.gguf --alias qwen3-0.6b -ngl 99 --port 8082
```

The router stays light: it sees a short prompt and is capped at a few output
tokens under a one-word grammar, so a 0.6B keeps routing near-instant. If real
decision points (a pending plan, where planner vs coder is a genuine choice)
misclassify, step up to a 1B–3B instruct — it's a pure config change (point the
Router model/URL at the bigger model and relaunch); no code change.

If all three fit in VRAM together you can instead run them under `llama-swap`
with distinct aliases; `llmrouter` selects by base URL/model regardless.

## Configure & start

Open the control panel at `http://localhost:4000/`:

1. Set the planner, coder, and router base URLs, model aliases, and slot counts (`-np`).
2. Optionally set a slot save path (must match each llama-server `--slot-save-path`).
3. Click **Save endpoints** / **Save router**, then **Start server**. Start
   health-checks the upstreams before arming `/v1`.

The rail lights up planner or coder as turns are routed; status shows reachability,
the last route, whether it used the fallback, and the discussion count.

While autostart (or a managed engine launch) is in progress, the UI shows a
distinct **starting** state instead of a flat "stopped": the sidebar status pill
pulses *starting…*, the Engine tab shows an auto-start banner, and each dashboard
card spins on *loading* rather than the alarming red *offline* until its
`/health` answers. During that window `/v1` returns `503` with a *"server is
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
message that isn't the very first entry, and a second one makes llama.cpp refuse
the request with *"System message must be at the beginning."* Coder result notes
folded back to the planner are added as `user` notes for the same reason.

## Troubleshooting & observability

Quick notes from real debugging sessions:

- **Every turn falls back to keywords / the coder never fires.** The router
  endpoint is almost certainly serving the wrong kind of model. With
  `LLMROUTER_DEBUG=1` you'll see `router classify: reply "…" is neither planner
  nor coder`. Point the Router model at a **generative** `Qwen3-0.6B`, not the
  `-Embedding-` variant, and relaunch it.
- **The coder loops, re-emitting the plan and restarting.** That's mid-turn
  re-routing — fixed by the tool-continuation guard (a continuation stays on the
  coder). If you still see it, confirm you rebuilt; with `LLMROUTER_DEBUG=1` a
  continuation should produce **no** `router classify` line at all.
- **A "create a plan" first turn went to the coder.** Shouldn't happen now: with
  no pending plan the grammar only allows `planner`. If it does, you're running an
  older build.
- **Watch routing live.** `LLMROUTER_DEBUG=1` logs each decision
  (`router classify: model=… reply="coder" -> "coder"`), plus `REQ …`
  (`ephemeral=`, `tools=`, `max_tokens=`), `ROUTE … slot=`, and `FWD -> …`.
- **Errors are surfaced, not swallowed.** When an upstream call fails, the router
  logs it (`ERR forward …` / `FWD <- status=…` with the body) and, in streaming
  mode, emits an SSE `error` chunk so the IDE shows the reason instead of a
  silently cut, empty stream.
- ***"System message must be at the beginning"* (HTTP 400).** The model's chat
  template requires a single leading `system` message. The router merges the
  IDE's system prompt and its role contract into one — but **existing discussions
  in `llmrouter.db` keep their old layout**, so delete/rename the DB (or clear
  `model_contexts`) after upgrading to clear stale prefixes.
- **A slot is created but the task never starts.** Usually slot contention: the
  pinned real turn and an unpinned utility request (IDE thread-title generation)
  both land on the same llama-server. Make sure `-np` on each llama-server
  matches the **Slots** count configured for that role.
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
router.go    routing brain: tiny GBNF model classifies every turn; pending-plan grammar
             guard; explicit-alias and tool-continuation guards; keyword fallback on failure
prompt.go    conversation signature (transcript + tool_calls), last-user / last-assistant /
             tool-continuation detection, append-only prompt building, plan/summary parsing
upstream.go  llama-server client: id_slot/cache_prompt, streaming tee that also
             reassembles tool_calls, slot save/restore; logs the outgoing request
             and any non-200 body
proxy.go     /v1 handler: resolve -> route -> build -> assign slot -> forward -> persist;
             X-LLMRouter-Role/-Fallback headers, surfaces upstream errors (SSE error chunk),
             starting/running/stopped lifecycle phase
admin.go     control-panel API + embedded UI (status reports last_route + last_route_fallback)
slots.go     per-role slot pool: maps each (role, discussion) to a slot, LRU eviction
web/index.html  the control panel (sidebar dashboard; Router card shows model-routing
             vs keyword-fallback state)
```
