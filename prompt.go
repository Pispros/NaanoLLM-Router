package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"regexp"
	"strings"
)

// Role contracts. Kept tiny so they don't fight an IDE's own large system
// prompt: they're appended as a short extra system note, not a replacement.
const plannerContract = `When you produce an implementation plan, emit it as a fenced block:
` + "```plan" + `
{"subtasks":[{"id":"t1","goal":"...","success":"...","context_refs":["path:lines"],"tools":["read_file","edit_file"],"depends_on":[]}]}
` + "```" + `
Reason freely before it. Put references, not full code, in context_refs.`

const coderContract = `Execute the entire plan — every subtask, in dependency order — using your tools.
Only once all subtasks are complete, end your final message with one line starting
exactly with "FINISH:" followed by a 2-3 line summary of what changed and the
outcome. Emit FINISH exactly once, at the very end. Do not restate the plan.`

var (
	planBlockRe = regexp.MustCompile("(?s)```plan\\s*(.*?)```")
	finishRe    = regexp.MustCompile(`(?m)^FINISH:\s*(.+)$`)
)

// ConvSig is the stable identity derived from an incoming request. Identity is
// anchored on the assistant transcript, which the client echoes back verbatim
// each turn; the system prompt, tools and user text are NOT part of identity
// because IDE/agent clients mutate them every turn (cwd, time, env details).
type ConvSig struct {
	SystemJSON string   // last seen system messages (stored for display; not identity)
	ToolsJSON  string   // last seen tools array (stored for display; not identity)
	AnchorKey  string   // sha256 of the normalized first user message (first-turn dedup)
	Anchors    []string // normalized assistant messages, in order (the continuation key)
}

// envDetailsRe strips the volatile context blocks IDE/agent clients append to
// user turns (open tabs, cwd, timestamps, terminal state, ...). Used only for
// the first-turn dedup anchor; continuation never relies on user text.
var envDetailsRe = regexp.MustCompile(`(?is)<environment_details>.*?</environment_details>`)

// reasoningRe strips model reasoning blocks. Some clients keep them in the
// re-sent history and some drop them; stripping on both sides (store + incoming)
// keeps the per-turn fingerprint stable regardless.
var reasoningRe = regexp.MustCompile(`(?is)<think(?:ing)?>.*?</think(?:ing)?>`)

// ConversationSignature derives a stable identity for the conversation the
// request belongs to. See ConvSig for why only the assistant transcript counts.
func ConversationSignature(req ChatRequest) ConvSig {
	var sys []Message
	var anchors []string
	firstUser := ""
	for _, m := range req.Messages {
		switch m.Role {
		case "system":
			sys = append(sys, m)
		case "assistant":
			anchors = append(anchors, turnAnchor(m.Text(), m.ToolCalls))
		case "user":
			if firstUser == "" {
				firstUser = normUser(m.Text())
			}
		}
	}
	sb, _ := json.Marshal(sys)
	tb := req.Tools
	if len(tb) == 0 {
		tb = []byte("null")
	}
	h := sha256.Sum256([]byte("u0:" + firstUser))
	return ConvSig{
		SystemJSON: string(sb),
		ToolsJSON:  string(tb),
		AnchorKey:  hex.EncodeToString(h[:]),
		Anchors:    anchors,
	}
}

// turnAnchor is the per-assistant-turn identity fingerprint: a hash over the
// (normalized) assistant text PLUS a canonical, id-free form of any tool_calls.
// It is computed identically on stored turns (from our own upstream output, via
// Store) and on incoming requests (from the client's echoed history), so the
// two compare equal no matter which IDE produced the turn or whether it carried
// text, tool_calls, or both.
func turnAnchor(contentText string, toolCallsRaw []byte) string {
	h := sha256.New()
	h.Write([]byte(normAssistant(contentText)))
	h.Write([]byte{0})
	h.Write([]byte(canonicalToolCalls(toolCallsRaw)))
	return hex.EncodeToString(h.Sum(nil))
}

// normUser strips volatile per-turn context so a retried first turn still
// matches its earlier (empty) discussion.
func normUser(s string) string {
	return strings.TrimSpace(envDetailsRe.ReplaceAllString(s, ""))
}

// normAssistant strips reasoning blocks and trims, so assistant text compares
// stably across clients that re-serialize or drop thinking content.
func normAssistant(s string) string {
	return strings.TrimSpace(reasoningRe.ReplaceAllString(s, ""))
}

// canonicalToolCalls reduces an OpenAI tool_calls array to a stable, id-free
// signature: the ordered list of (function name, canonical-JSON arguments).
// Tool-call ids are server-generated and excluded; only the semantic call
// content is kept, which the client echoes back verbatim.
func canonicalToolCalls(raw []byte) string {
	if len(raw) == 0 {
		return ""
	}
	var calls []struct {
		Function struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &calls) != nil || len(calls) == 0 {
		return ""
	}
	var b strings.Builder
	for _, c := range calls {
		b.WriteString(c.Function.Name)
		b.WriteByte('(')
		b.WriteString(canonicalJSON(c.Function.Arguments))
		b.WriteString(")\n")
	}
	return b.String()
}

// canonicalJSON re-encodes a JSON value so whitespace and object key-order
// differences between clients don't change the signature.
func canonicalJSON(raw json.RawMessage) string {
	s := strings.TrimSpace(string(raw))
	if s == "" {
		return ""
	}
	var v any
	if json.Unmarshal([]byte(s), &v) != nil {
		return s
	}
	out, err := json.Marshal(v)
	if err != nil {
		return s
	}
	return string(out)
}

func systemMessages(systemJSON string) []Message {
	var sys []Message
	json.Unmarshal([]byte(systemJSON), &sys)
	return sys
}

// lastUserText returns the newest user message in the incoming request.
func lastUserText(req ChatRequest) string {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			return req.Messages[i].Text()
		}
	}
	return ""
}

// BuildMessages assembles the append-only prompt for the chosen role.
//
// Layout (stable prefix first, so llama.cpp reuses the warm KV):
//
//	[ system + role contract ]      <- frozen, cached
//	[ prior per-role messages  ]    <- grown append-only across turns
//	[ this turn's new input     ]   <- the only part that re-prefills
//
// It also returns the freshly appended messages so the caller can persist them.
func BuildMessages(store *Store, discID int64, role, systemJSON, userMsg string) (full []Message, appended []Message) {
	prior, _ := store.LoadRoleMessages(discID, role)

	// First time we touch this role in this discussion: seed the frozen prefix
	// (system + the short role contract). It becomes the stable, cached lead.
	if len(prior) == 0 {
		contract := plannerContract
		if role == "coder" {
			contract = coderContract
		}
		appended = append(appended, systemMessages(systemJSON)...)
		appended = append(appended, TextMessage("system", contract))
	}

	switch role {
	case "coder":
		// Inject the scoped subtask drawn from the latest plan — never the
		// planner's full reasoning. This is what keeps the coder's prefill tiny.
		if plan, ok := store.LatestUnconsumedPlan(discID); ok {
			task := "Execute this plan in full: complete every subtask in order, then emit a single FINISH line at the very end.\n\n" + plan
			if userMsg != "" {
				task += "\n\nUser note: " + userMsg
			}
			m := TextMessage("user", task)
			appended = append(appended, m)
			store.MarkPlansConsumed(discID)
		} else if userMsg != "" {
			m := TextMessage("user", userMsg)
			appended = append(appended, m)
		}

	default: // planner
		// Fold any coder summaries into the planner's context before the new
		// user turn, so the planner "sees" what was executed (append-only).
		for _, sum := range store.PopSummaries(discID) {
			appended = append(appended, TextMessage("system", "[coder result] "+sum))
		}
		if userMsg != "" {
			appended = append(appended, TextMessage("user", userMsg))
		}
	}

	full = append(prior, appended...)
	return full, appended
}

// ExtractPlan pulls a fenced ```plan ...``` block from planner output, if any.
func ExtractPlan(assistant string) (string, bool) {
	m := planBlockRe.FindStringSubmatch(assistant)
	if len(m) < 2 {
		return "", false
	}
	s := strings.TrimSpace(m[1])
	if !json.Valid([]byte(s)) {
		return "", false
	}
	return s, true
}

// ExtractSummary pulls the coder's self-emitted "FINISH: ..." line.
func ExtractSummary(assistant string) (string, bool) {
	m := finishRe.FindStringSubmatch(assistant)
	if len(m) < 2 {
		return "", false
	}
	return strings.TrimSpace(m[1]), true
}
