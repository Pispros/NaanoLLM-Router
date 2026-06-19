package main

import (
	"encoding/json"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
)

type Proxy struct {
	cfg   *Config
	store *Store

	ready atomic.Bool // toggled by the control panel Start/Stop

	mu        sync.Mutex
	mgrs      map[string]*slotManager // role -> slot pool
	lastRoute atomic.Value            // string, for the dashboard
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
	role := Route(cfg, p.store, discID, req.Model, userMsg)
	p.lastRoute.Store(role)
	up := cfg.Planner
	if role == "coder" {
		up = cfg.Coder
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

	// 5. Forward, streaming straight through to the client.
	stream := req.Stream
	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
	} else {
		w.Header().Set("Content-Type", "application/json")
	}
	w.Header().Set("X-LLMRouter-Role", role)
	w.Header().Set("X-LLMRouter-Slot", itoa(int64(slot)))

	flusher, _ := w.(http.Flusher)
	flush := func() {
		if flusher != nil {
			flusher.Flush()
		}
	}

	assistant, toolCalls, ferr := Forward(r.Context(), up, slot, raw, full, stream, w, flush)
	if ferr != nil {
		if !stream {
			http.Error(w, `{"error":{"message":"`+jsonEscape(ferr.Error())+`"}}`, http.StatusBadGateway)
		}
		return
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
	p.ready.Store(true)
	return nil
}
func (p *Proxy) Stop()      { p.ready.Store(false) }
func (p *Proxy) Ready() bool { return p.ready.Load() }

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
