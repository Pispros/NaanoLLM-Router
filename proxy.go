package main

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
)

// debugOn dumps full request/response bodies when LLMROUTER_DEBUG=1.
var debugOn = os.Getenv("LLMROUTER_DEBUG") == "1"

func snip(b []byte) string {
	const max = 2000
	if len(b) <= max {
		return string(b)
	}
	return string(b[:max]) + "...(" + itoa(int64(len(b))) + "B)"
}

type Proxy struct {
	cfg   *Config
	store *Store

	ready    atomic.Bool // armed: /v1 serves traffic
	starting atomic.Bool // autostart in progress: launching engines / waiting for health

	mu        sync.Mutex
	mgrs      map[string]*slotManager // role -> slot pool
	lastRoute atomic.Value            // string, for the dashboard
	lastFB    atomic.Bool             // last route used the keyword fallback (model failed)
}

func NewProxy(cfg *Config, store *Store) *Proxy {
	p := &Proxy{cfg: cfg, store: store, mgrs: map[string]*slotManager{}}
	p.lastRoute.Store("")
	return p
}

func (p *Proxy) LastRoute() string {
	if v, ok := p.lastRoute.Load().(string); ok {
		return v
	}
	return ""
}

// LastRouteFallback reports whether the most recent routed turn fell back to the
// keyword rules because the router model didn't answer.
func (p *Proxy) LastRouteFallback() bool { return p.lastFB.Load() }

// mgr returns the slot pool for a role, creating it lazily from current config.
func (p *Proxy) mgr(role string) *slotManager {
	p.mu.Lock()
	defer p.mu.Unlock()
	if m := p.mgrs[role]; m != nil {
		return m
	}
	cfg := p.cfg.Snapshot()
	n := cfg.Planner.Slots
	if role == "coder" {
		n = cfg.Coder.Slots
	}
	m := newSlotManager(n)
	p.mgrs[role] = m
	return m
}

func (p *Proxy) SlotStats(role string) (used, size int) {
	return p.mgr(role).stats()
}

// ensureWarm picks the slot for this (role, discussion) and, if KV persistence
// is on, parks the evicted discussion's cache and restores this one's. Returns
// the slot id to pin for the call.
func (p *Proxy) ensureWarm(up Upstream, role string, discID int64) int {
	dec := p.mgr(role).acquire(discID)
	cfg := p.cfg.Snapshot()
	if cfg.SlotSavePath != "" {
		if dec.evicted != 0 { // park the discussion we're displacing
			_ = SaveSlot(up, dec.slot, safeFilename(role, itoa(dec.evicted)))
		}
		if !dec.warm { // try to bring this discussion's KV back (else cold prefill)
			_ = RestoreSlot(up, dec.slot, safeFilename(role, itoa(discID)))
		}
	}
	return dec.slot
}

// ServeHTTP implements POST /v1/chat/completions.
func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !p.ready.Load() {
		if p.starting.Load() {
			w.Header().Set("Retry-After", "5")
			http.Error(w, `{"error":{"message":"server is starting — engines are loading, retry shortly"}}`, http.StatusServiceUnavailable)
			return
		}
		http.Error(w, `{"error":{"message":"server not started — open the control panel and click Start"}}`, http.StatusServiceUnavailable)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		http.Error(w, `{"error":{"message":"invalid JSON"}}`, http.StatusBadRequest)
		return
	}
	var req ChatRequest
	json.Unmarshal(body, &req)

	cfg := p.cfg.Snapshot()

	// --- request visibility -------------------------------------------------
	eph := isEphemeral(raw, req)
	log.Printf("REQ remote=%s model=%v stream=%v msgs=%d tools=%v max_tokens=%v max_completion_tokens=%v ephemeral=%v",
		r.RemoteAddr, raw["model"], req.Stream, len(req.Messages), hasTools(req.Tools),
		raw["max_tokens"], raw["max_completion_tokens"], eph)
	if debugOn {
		log.Printf("REQ body: %s", snip(body))
		for i, m := range req.Messages {
			if m.Role == "assistant" {
				log.Printf("REQ incoming assistant[%d]: %q", i, short(normAssistant(m.Text())))
			}
		}
	}

	// 0. Utility one-shots — IDE thread-title / summary / commit-message
	//    generation and other short, tool-less completions. Zed (and others)
	//    fire these *in addition to* the real turn, pointed at the same model
	//    (Zed's thread_summary_model falls back to the default model). They must
	//    not spawn a discussion or pin a per-discussion slot, otherwise every
	//    prompt burns an extra planner slot on a throwaway request. Forward them
	//    to the planner upstream, unpinned, and don't persist.
	if eph {
		stream := req.Stream
		setStreamHeaders(w, stream)
		w.Header().Set("X-LLMRouter-Role", "utility")
		flusher, _ := w.(http.Flusher)
		flush := func() {
			if flusher != nil {
				flusher.Flush()
			}
		}
		if _, _, ferr := Forward(r.Context(), cfg.Planner, -1, raw, nil, stream, w, flush); ferr != nil {
			log.Printf("ERR utility forward up=%s: %v", cfg.Planner.BaseURL, ferr)
			emitForwardError(w, flush, stream, ferr)
		}
		return
	}

	// 1. Resolve the discussion by matching its assistant transcript (stable
	//    across turns), not by hashing the volatile system/tools/first-user lead.
	sig := ConversationSignature(req)
	discID, err := p.store.ResolveDiscussion(sig)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// 2. Route this turn.
	userMsg := lastUserText(req)
	role, fellBack := Route(cfg, p.store, discID, req.Model, userMsg, lastAssistantText(req), isToolContinuation(req))
	p.lastRoute.Store(role)
	p.lastFB.Store(fellBack)
	if fellBack {
		log.Printf("WARN discID=%d: router model did not answer — used keyword fallback, role=%s", discID, role)
	}
	up := cfg.Planner
	if role == "coder" {
		up = cfg.Coder
	}

	// The planner reasons and emits a ```plan block — it must NOT act. Zed sends
	// its full agentic tool set on every request, and a weak planner model will
	// happily call those tools (listing dirs, etc.) and loop instead of planning,
	// often ending in empty turns and never emitting a plan. Stripping tools for
	// the planner forces a text answer (the plan); the coder keeps its tools and
	// does the execution.
	if role == "planner" {
		delete(raw, "tools")
		delete(raw, "tool_choice")
		delete(raw, "parallel_tool_calls")
	}

	// Per-role sampling. A weak local model (Qwen2.5-9B etc.) follows a strict
	// output format far better near temperature 0 than at a chat default (~0.7).
	// The planner must be deterministic — it emits a machine-readable plan; the
	// coder needs a little freedom to write code but not to wander. We override
	// whatever the IDE sent. (Sensible defaults; could move to config later.)
	switch role {
	case "planner":
		raw["temperature"] = 0.0
		raw["top_p"] = 0.1
	case "coder":
		raw["temperature"] = 0.15
	}

	// 3. Build the append-only prompt for that role (stable prefix preserved).
	full, appended := BuildMessages(p.store, discID, role, sig.SystemJSON, userMsg)

	// 4. Reserve the right slot for THIS discussion and warm its KV. Only
	//    llama.cpp understands id_slot / cache_prompt; other engines batch
	//    internally, so we skip slot pinning for them.
	slot := 0
	if isLlamaCpp(up.Engine) {
		slot = p.ensureWarm(up, role, discID)
	}
	log.Printf("ROUTE discID=%d role=%s up=%s slot=%d", discID, role, up.BaseURL, slot)

	// 5. Forward, streaming straight through to the client.
	stream := req.Stream
	setStreamHeaders(w, stream)
	w.Header().Set("X-LLMRouter-Role", role)
	w.Header().Set("X-LLMRouter-Slot", itoa(int64(slot)))
	if fellBack {
		w.Header().Set("X-LLMRouter-Fallback", "1")
	}

	flusher, _ := w.(http.Flusher)
	flush := func() {
		if flusher != nil {
			flusher.Flush()
		}
	}

	assistant, toolCalls, ferr := Forward(r.Context(), up, slot, raw, full, stream, w, flush)
	if ferr != nil {
		// A client that cancels mid-stream (e.g. the user stops the planner once
		// it has emitted the plan and starts rambling) still received real
		// content, and that content may carry a complete plan. A cancellation
		// with non-empty output is a normal, persistable turn — not a failure —
		// so save it instead of throwing the plan away. Only a genuine upstream
		// error (client still connected) is fatal.
		if r.Context().Err() != nil && (assistant != "" || len(toolCalls) > 0) {
			log.Printf("INFO discID=%d role=%s: client cancelled mid-stream, persisting partial turn (%d chars)", discID, role, len(assistant))
			p.persist(discID, role, appended, userMsg, assistant, toolCalls)
			return
		}
		log.Printf("ERR forward discID=%d role=%s slot=%d up=%s: %v", discID, role, slot, up.BaseURL, ferr)
		emitForwardError(w, flush, stream, ferr)
		return
	}
	if stream && assistant == "" && toolCalls == nil {
		log.Printf("WARN discID=%d role=%s: upstream streamed an empty turn (no content, no tool_calls)", discID, role)
	}

	// 6. Persist text/plan/summary — the durable, reconstructable truth. Done
	//    synchronously so this turn's assistant reply is on record before the
	//    next request arrives and tries to match the transcript against it.
	p.persist(discID, role, appended, userMsg, assistant, toolCalls)
}

func (p *Proxy) persist(discID int64, role string, appended []Message, userMsg, assistant string, toolCalls json.RawMessage) {
	prior, slotFile := p.store.LoadRoleMessages(discID, role)
	msgs := append(prior, appended...)
	msgs = append(msgs, TextMessage("assistant", assistant))
	p.store.SaveRoleMessages(discID, role, msgs, slotFile)

	// Fingerprint this turn exactly as ConversationSignature will fingerprint it
	// when the client echoes it back next turn, so the discussion is recognised.
	anchorSig := turnAnchor(assistant, toolCalls)
	p.store.AppendTurn(discID, role, userMsg, assistant, anchorSig)

	switch role {
	case "planner":
		if plan, ok := ExtractPlan(assistant); ok {
			p.store.SavePlan(discID, plan)
		}
	case "coder":
		if sum, ok := ExtractSummary(assistant); ok {
			p.store.PushSummary(discID, sum) // waits to be folded into the planner
		}
	}
}

// Models implements GET /v1/models, advertising the routing aliases.
func (p *Proxy) Models(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"id": "auto", "object": "model", "owned_by": "naanollm-router"},
			{"id": "planner", "object": "model", "owned_by": "naanollm-router"},
			{"id": "coder", "object": "model", "owned_by": "naanollm-router"},
		},
	})
}

// Start rebuilds slot pools from current config and arms the /v1 endpoint.
func (p *Proxy) Start() error {
	cfg := p.cfg.Snapshot()
	p.mu.Lock()
	p.mgrs = map[string]*slotManager{
		"planner": newSlotManager(cfg.Planner.Slots),
		"coder":   newSlotManager(cfg.Coder.Slots),
	}
	p.mu.Unlock()
	p.starting.Store(false)
	p.ready.Store(true)
	return nil
}

func (p *Proxy) Stop() {
	p.ready.Store(false)
	p.starting.Store(false)
}
func (p *Proxy) Ready() bool { return p.ready.Load() }

// BeginStarting marks the proxy as warming up (engines launching / health
// pending). Set at the top of autostart so the dashboard and /v1 can report a
// "starting" state instead of a flat "stopped" while the binary boots.
func (p *Proxy) BeginStarting() { p.starting.Store(true) }

// AbortStarting clears the warming-up flag when autostart fails to arm /v1.
func (p *Proxy) AbortStarting() { p.starting.Store(false) }

// Starting reports whether autostart is in its loading window.
func (p *Proxy) Starting() bool { return p.starting.Load() }

// Phase is the single overall lifecycle state for the dashboard.
func (p *Proxy) Phase() string {
	switch {
	case p.ready.Load():
		return "running"
	case p.starting.Load():
		return "starting"
	default:
		return "stopped"
	}
}

func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	neg := n < 0
	if neg {
		n = -n
	}
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

func jsonEscape(s string) string {
	b, _ := json.Marshal(s)
	return string(b[1 : len(b)-1])
}

func setStreamHeaders(w http.ResponseWriter, stream bool) {
	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
	} else {
		w.Header().Set("Content-Type", "application/json")
	}
}

// emitForwardError makes an upstream failure visible to the client instead of
// silently closing the (already 200-committed) stream. In streaming mode the
// status line is gone, so the only way to tell the IDE is an SSE error chunk.
func emitForwardError(w http.ResponseWriter, flush func(), stream bool, err error) {
	msg := `{"error":{"message":"` + jsonEscape(err.Error()) + `"}}`
	if stream {
		w.Write([]byte("data: " + msg + "\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
		if flush != nil {
			flush()
		}
		return
	}
	http.Error(w, msg, http.StatusBadGateway)
}

// ephemeralMaxTokens: a request capped at or below this many output tokens and
// carrying no tools is treated as a one-shot utility call (title/summary/commit
// message), not a conversation turn.
const ephemeralMaxTokens = 512

// isEphemeral spots IDE utility completions that should bypass discussion and
// slot bookkeeping. The signal is robust across IDEs: real chat/agent turns
// either advertise tools or don't cap output to a tiny budget, while title /
// summary / commit-message generation does both (no tools, small max_tokens).
func isEphemeral(raw map[string]any, req ChatRequest) bool {
	if hasTools(req.Tools) {
		return false
	}
	// Known meta one-shots (thread title / summary) act on the conversation
	// instead of continuing it. They have no tools but don't always set a small
	// max_tokens, so catch them by their instruction text before the budget check
	// — otherwise each one spawns a throwaway discussion and burns a slot.
	if utilityPromptRe.MatchString(lastUserText(req)) {
		return true
	}
	if n, ok := asPositiveInt(raw["max_tokens"]); ok && n <= ephemeralMaxTokens {
		return true
	}
	if n, ok := asPositiveInt(raw["max_completion_tokens"]); ok && n <= ephemeralMaxTokens {
		return true
	}
	return false
}

func hasTools(raw json.RawMessage) bool {
	s := strings.TrimSpace(string(raw))
	return s != "" && s != "null" && s != "[]"
}

func asPositiveInt(v any) (int, bool) {
	if f, ok := v.(float64); ok && f > 0 {
		return int(f), true
	}
	return 0, false
}
