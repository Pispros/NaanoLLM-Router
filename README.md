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

Because tool_calls are part of the fingerprint, this works for agentic clients
whose turns are pure tool calls with little or no text (Cline, Roo Code, JetBrains
Junie/AI Assistant, …), not just plain chat clients. The system prompt, the tools
array, and user text never affect identity, so mutating them between turns is
harmless.

Turns are persisted **synchronously** at the end of each request, so turn *N* is
on record before turn *N+1* arrives to be matched against it.

## Architecture

```
IDE (base_url -> :4000/v1, model=auto)
        │
        ▼
  llmrouter (this binary)
    • resolve discussion by assistant-transcript match (text + tool_calls)
    • route turn: state rule (+ optional tiny GBNF model) -> planner | coder
    • build append-only prompt: [system+contract] + [per-role history] + [new input]
    • pin the role's warm slot (id_slot), cache_prompt:true
    • stream straight through (tool_calls relayed verbatim)
    • persist text/plan/summary to SQLite (synchronously)
        │                         │
        ▼                         ▼
  llama-server (planner)     llama-server (coder)
   -np N : one slot per        -np N : one slot per
   active discussion           active discussion
```

Append-only discipline keeps each model's prefix stable, so llama.cpp reuses the
warm KV and only re-prefills the small new suffix.

## Parallel discussions

Every conversation (identified by its assistant transcript, see above) is an
independent flow with its own per-role context in SQLite. At runtime, llmrouter
spreads each upstream's `-np N` slots across the active discussions: one slot per
(role, discussion). When more discussions are active than there are slots, the
least-recently-used one is evicted — its KV parked to a `.bin` if a slot save
path is set, otherwise simply rebuilt from SQLite text on its next turn. Set
`Slots` (the `-np` value) to how many discussions you want warm at once.

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
(planner, coder) independently picks one of five engines:

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
   relaunch both roles and arm `/v1` automatically next time.

Only llama.cpp gets per-discussion KV slot pinning (`id_slot`/`cache_prompt`);
the other engines batch internally, so naanollm-router skips those fields for
them automatically. Managed processes are stopped on Ctrl-C / SIGTERM.

When the router is set to **llm** mode (Router tab), you can also load a custom
router model with the same engine picker, and launch it from there. The router
only classifies each turn (planner vs coder), so a tiny instruct model is
plenty — the **Use recommended** button pre-fills Qwen3-0.6B on llama.cpp. With
autostart on, the router model is launched at boot too.

The manual `llama-server` invocations below still work if you prefer to run the
backends yourself — the Engine tab is optional.

## Run the two models (llama-server)

Keep both resident so switching is instant. Dedicate a slot per process and
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

Optional tiny router model (only used when `router.mode = llm`):

```bash
llama-server -m qwen3-0.6b-Q4_K_M.gguf --alias qwen3-0.6b -ngl 99 --port 8082
```

If both models fit in VRAM together (e.g. ~10.5 GB of weights in 16 GB), you can
instead run them under `llama-swap` and give each a different alias; `llmrouter`
selects by base URL/model regardless.

## Configure & start

Open the control panel at `http://localhost:4000/`:

1. Set the planner and coder base URLs, model aliases, and slot counts (`-np`).
2. Pick router mode (`rule` is enough for most cases; `llm` adds a tie-breaker).
3. Optionally set a slot save path (must match each llama-server `--slot-save-path`).
4. Click **Save endpoints**, then **Start server**. Start health-checks both
   upstreams before arming `/v1`.

The rail lights up planner or coder as turns are routed; status shows reachability,
the last route, and the discussion count.

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
`auto`, llmrouter decides from intent + whether a plan is pending.

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

- **Errors are surfaced, not swallowed.** When an upstream call fails, the router
  logs it (`ERR forward …` / `FWD <- status=…` with the body) and, in streaming
  mode, emits an SSE `error` chunk so the IDE shows the reason instead of a
  silently cut, empty stream.
- **`LLMROUTER_DEBUG=1`** dumps each incoming request body and key decisions
  (`REQ …` with `ephemeral=`, `tools=`, `max_tokens=`; `ROUTE … slot=`;
  `FWD -> …` with `id_slot`). Run with it when a client misbehaves but `curl`
  doesn't.
- ***"System message must be at the beginning"* (HTTP 400).** The model's chat
  template requires a single leading `system` message. The router now merges the
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
main.go      entrypoint, one listener for /v1 + control panel, auto-creates/migrates the DB
config.go    config types, JSON load/save (endpoints, slot counts, router)
store.go     SQLite: discussions, per-role context, plans, summaries (text only);
             transcript-match discussion resolution + per-turn fingerprints + auto-migration
openai.go    OpenAI request/response types
router.go    intent routing: state rule + optional GBNF tiny model
prompt.go    conversation signature (transcript + tool_calls), append-only prompt
             building (single merged leading system message), plan/summary parsing
upstream.go  llama-server client: id_slot/cache_prompt, streaming tee that also
             reassembles tool_calls, slot save/restore; logs the outgoing request
             and any non-200 body
proxy.go     /v1 handler: resolve -> route -> build -> assign slot -> forward -> persist;
             request/decision logging, surfaces upstream errors (SSE error chunk),
             starting/running/stopped lifecycle phase
admin.go     control-panel API + embedded UI
slots.go     per-role slot pool: maps each (role, discussion) to a slot, LRU eviction
web/index.html  the control panel (sidebar dashboard)
```
