package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
)

const helpSystemPrompt = `You are the built-in help assistant for "naanollm-router", a local LLM router.
Answer the user's question about the app using ONLY the documentation below.
Be concise and concrete. If the answer is not in the documentation, say you don't
know rather than inventing details. Reply in the same language the user writes in.

=== naanollm-router documentation ===
`

// helpDoc is the in-app knowledge base handed to the planner as context (RAG).
const helpDoc = `naanollm-router is a small single-binary Go proxy. It exposes ONE OpenAI-compatible
endpoint (default http://localhost:4000/v1) and routes each chat turn to one of two
roles: a PLANNER model (reasoning, RAG, builds the plan) or a CODER model (executes
the plan). An IDE points at the router with model "auto" and thinks it is talking to
a single model; it is talking to the router.

CORE RULE: SQLite stores TEXT and METADATA only (messages, the plan, coder summaries).
The KV cache is never stored in SQLite; it lives in each backend's hot slot (RAM/VRAM),
or optional .bin files via --slot-save-path. The router moves text between models,
never KV. A KV cache is a disposable accelerator bound to one model + one quant and is
never shared between models; if it is lost it is rebuilt by re-prefilling from SQLite.

ROUTER (Router tab) decides per turn whether a message goes to the planner or the coder.
- mode "rule" (default): no model, no VRAM, deterministic. It combines one state (is a
  plan pending?) with two keyword regexes on the latest user message. Decision order:
  (1) if the IDE forces model "planner" or "coder", that wins;
  (2) pending plan AND execution intent (execute, implement, go, run it, do it, apply,
      fais-le, vas-y, lance...) -> coder;
  (3) planning intent (plan, analyse, design, architect, explique, pourquoi...) and no
      execution intent -> planner;
  (4) no pending plan -> planner (nothing to execute yet);
  (5) otherwise (pending plan, ambiguous) -> planner. This is the safe default: keep
      talking to the planner rather than mutating files without a clear go-ahead.
- mode "llm": identical to rule, but in the ambiguous case a tiny model (e.g. Qwen3-0.6B),
  constrained by a GBNF grammar so it can only answer "planner" or "coder", breaks the tie.
  You can load a custom router model from the Router tab; the "Use recommended" button
  pre-fills Qwen3-0.6B on llama.cpp.

HANDOFF CONTRACT:
- The planner emits its plan as a fenced "plan" block (JSON with subtasks). The router
  stores it as a pending plan.
- The coder receives the full plan and executes EVERY subtask in order, then ends its
  final message with one line starting exactly with "FINISH:" followed by a 2-3 line
  summary. It emits FINISH exactly once, at the very end.
- That FINISH summary is folded back into the planner's context on the next planner turn,
  so the planner sees what was executed.

ENDPOINTS tab: configure the two resident backends — Base URL, model alias, and parallel
slots (-np, how many discussions a server keeps warm at once). This is also how you
connect a model you launched yourself: just set its Base URL here and ignore the Engine tab.

ENGINE tab (optional): launch and supervise an inference server per role from the panel.
Five engines are supported: llama.cpp (binary "llama-server", GGUF models), vLLM (binary
"vllm"), SGLang (a Python module — point to the Python interpreter), Text Generation
Inference (binary "text-generation-launcher"), and Ollama (binary "ollama"). You install
or build the engine yourself, paste the absolute path to its binary, click Test, pick a
model (search Hugging Face; for llama.cpp also a quant .gguf file; for Ollama type a model
tag you have pulled), then Launch. IMPORTANT: the Engine tab launches a MINIMAL, portable
command with no hardware tuning (no GPU offload, no KV-cache quantization, no context-size,
no tensor-parallel, no slot persistence). For best performance, run each engine as your own
service or script tuned to your hardware and context, then just point the Endpoints tab at
its Base URL — you can skip the Engine tab entirely.

SERVER tab: the listen address (default :4000) and an optional slot save path for KV .bin
persistence (must match each llama-server --slot-save-path). Point your IDE's OpenAI base
URL at http://localhost:4000/v1, model "auto". Changing the listen address needs a binary
restart.

AUTOSTART (Engine tab): when enabled, starting naanollm-router launches the planner and the
coder (and the router model if mode is llm), then arms /v1 once both are healthy. Managed
processes are stopped when naanollm-router exits.

DISCUSSIONS tab: each conversation is an independent flow identified by a stable prefix hash,
with its own per-role context in SQLite and its own warm slot. When more discussions are
active than there are slots, the least-recently-used one is evicted (its KV parked to a .bin
if a slot save path is set, otherwise rebuilt from SQLite on its next turn).

WIRING AN IDE (Cline, Roo, Zed, Continue, Cursor, Aider, opencode...): provider
OpenAI-compatible; base URL http://localhost:4000/v1; API key anything (ignored locally);
model "auto" (or force "planner" / "coder"). In Cline/Roo, Plan/Act can map to planner/coder.

ENGINE-SPECIFIC NOTE: only llama.cpp gets per-discussion KV slot pinning (id_slot /
cache_prompt) and slot save/restore; the other engines batch internally, so the router omits
those fields for them automatically.

UI: dark navy theme, FR/EN language switch (top-right). The sidebar has a status pill and
Start/Stop; Start arms /v1 after health-checking both upstreams. This help chat is answered
by your configured planner model, so the planner must be reachable for it to work.`

// helpChat streams a help answer from the planner, prepending the doc as context.
func (a *Admin) helpChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var in struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		http.Error(w, jsonErr(err.Error()), http.StatusBadRequest)
		return
	}

	cfg := a.cfg.Snapshot()
	up := cfg.Planner
	if !HealthAt(up.BaseURL, engineByID(up.Engine).HealthPath) {
		writeErr(w, http.StatusBadGateway, codedErr("unreachable", "planner", "planner endpoint unreachable"))
		return
	}

	msgs := []map[string]any{{"role": "system", "content": helpSystemPrompt + helpDoc}}
	msgs = append(msgs, in.Messages...)
	reqBody, _ := json.Marshal(map[string]any{
		"model":       up.Model,
		"messages":    msgs,
		"stream":      true,
		"temperature": 0.2,
	})

	preq, _ := http.NewRequestWithContext(r.Context(), "POST", up.BaseURL+"/v1/chat/completions", bytes.NewReader(reqBody))
	preq.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(preq)
	if err != nil {
		writeErr(w, http.StatusBadGateway, codedErr("unreachable", "planner", err.Error()))
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(resp.Body)
		http.Error(w, jsonErr("planner: "+string(msg)), http.StatusBadGateway)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher, _ := w.(http.Flusher)
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		w.Write(sc.Bytes())
		w.Write([]byte("\n"))
		if flusher != nil {
			flusher.Flush()
		}
	}
}
