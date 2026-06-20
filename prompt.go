package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// Role contracts. Kept tiny so they don't fight an IDE's own large system
// prompt: they're appended as a short extra system note, not a replacement.
const plannerContract = `You are the planner. Produce a plan and nothing else.
Emit exactly one fenced block, then STOP — no preamble, no prose after it, and never start implementing or calling tools:
` + "```plan" + `
{"subtasks":[{"id":"t1","goal":"...","success":"...","context_refs":["path:lines"],"tools":["read_file","edit_file"],"depends_on":[]}]}
` + "```" + `
Use references, not code, in context_refs. After the closing fence, output nothing further.`

const coderContract = `You are the executor, not a planner. The plan is already approved — your job is to make the changes, not to discuss them.

- Act immediately: start calling tools on the first subtask. Do not deliberate about whether the plan is correct or complete.
- You have the full tool set (create_directory, write_file, edit_file, read_file, terminal, …). Use whatever each subtask needs. If a file or directory does not exist yet, CREATE it.
- The subtask goals are what matter. Any tool hints carried over from planning are NOT restrictions — ignore them if they get in the way.
- Work in dependency order. Do not skip, redesign, replan, re-explain, or propose alternatives.
- Keep prose to a minimum — prefer tool calls over explanation.

When every subtask is complete, output exactly one final line:
FINISH: <2-3 line summary of what changed>
Do not restate the plan.`

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
	// ClientKey is the client-supplied stable conversation id (OpenAI
	// prompt_cache_key), when the client sends one. It is content-independent and
	// per-thread, so it is a stronger identity than the transcript: it is used as
	// the PRIMARY discussion key, with the transcript anchors kept as a fallback
	// for clients that don't send it.
	ClientKey string
}

// envDetailsRe strips the volatile context blocks IDE/agent clients append to
// user turns (open tabs, cwd, timestamps, terminal state, ...). Used only for
// the first-turn dedup anchor; continuation never relies on user text.
var envDetailsRe = regexp.MustCompile(`(?is)<environment_details>.*?</environment_details>`)

// reasoningRe strips model reasoning blocks. Some clients keep them in the
// re-sent history and some drop them; stripping on both sides (store + incoming)
// keeps the per-turn fingerprint stable regardless.
var reasoningRe = regexp.MustCompile(`(?is)<think(?:ing)?>.*?</think(?:ing)?>`)

// utilityPromptRe spots IDE meta one-shots that operate ON the conversation
// rather than continuing it — thread-title and thread-summary generation (Zed
// fires these alongside the real turn). They carry no tools and no
// prompt_cache_key, and don't always cap max_tokens, so they slip past the
// token-budget check in isEphemeral and would otherwise spawn a throwaway
// discussion and pin a slot. Anchoring on "...conversation" avoids catching a
// genuine "summarize this file/function" task.
var utilityPromptRe = regexp.MustCompile(`(?i)\b(title|summary|summari[sz]e|recap)\b[\s\S]{0,40}(\bthis conversation\b|\b(for|of|about) (the|this|our) conversation\b)`)

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
		ClientKey:  strings.TrimSpace(req.PromptCacheKey),
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

// lastAssistantText returns the newest assistant message text in the request,
// trimmed — used to give the router a little context about what just happened.
func lastAssistantText(req ChatRequest) string {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "assistant" {
			return strings.TrimSpace(req.Messages[i].Text())
		}
	}
	return ""
}

// isToolContinuation reports whether this request is a mid-turn agentic step
// (the client fed tool results back and wants the model to continue) rather
// than a fresh user turn. The signal: after the newest user message there is a
// tool result or an assistant message that issued tool_calls. Only the coder is
// given tools, so a continuation always belongs to the coder finishing its
// work — it must NOT be re-routed, or execution oscillates with the planner and
// loops. New user turns (nothing after the last user message) are not
// continuations and are routed normally.
func isToolContinuation(req ChatRequest) bool {
	lastUser := -1
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			lastUser = i
			break
		}
	}
	for i := lastUser + 1; i < len(req.Messages); i++ {
		m := req.Messages[i]
		if m.Role == "tool" || m.ToolCallID != "" {
			return true
		}
		if m.Role == "assistant" && len(m.ToolCalls) > 0 && string(m.ToolCalls) != "null" {
			return true
		}
	}
	return false
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
		// Merge the IDE's system message(s) AND our role contract into a SINGLE
		// leading system message. Many chat templates (Qwen3.x, etc.) reject any
		// system message that isn't the very first entry, so appending the
		// contract as a second system message makes llama.cpp fail to apply the
		// template ("System message must be at the beginning") whenever the IDE
		// also sent a system prompt. One combined system message is template-safe
		// and preserves the same stable, cacheable prefix.
		var sys strings.Builder
		for _, m := range systemMessages(systemJSON) {
			if txt := m.Text(); txt != "" {
				if sys.Len() > 0 {
					sys.WriteString("\n\n")
				}
				sys.WriteString(txt)
			}
		}
		if sys.Len() > 0 {
			sys.WriteString("\n\n")
		}
		sys.WriteString(contract)
		appended = append(appended, TextMessage("system", sys.String()))
	}

	switch role {
	case "coder":
		// Inject the scoped subtask drawn from the latest plan — never the
		// planner's full reasoning. This is what keeps the coder's prefill tiny.
		if plan, ok := store.LatestUnconsumedPlan(discID); ok {
			task := "Execute this plan now. Complete every subtask in dependency order using your tools, then emit a single FINISH line at the very end.\n\n" + planToChecklist(plan)
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
		// user turn, so the planner "sees" what was executed (append-only). These
		// go in as USER notes, not system messages: a system message here would
		// be mid-conversation and break templates that require system to be first.
		for _, sum := range store.PopSummaries(discID) {
			appended = append(appended, TextMessage("user", "[coder result] "+sum))
		}
		if userMsg != "" {
			appended = append(appended, TextMessage("user", userMsg))
		}
	}

	full = append(prior, appended...)
	return full, appended
}

// planToChecklist renders a plan as a clean, ordered instruction list for the
// coder: id, goal, success criterion and dependency order only. The planner's
// per-subtask "tools" and "context_refs" metadata is deliberately dropped — the
// coder already has the client's full tool set, and weak models misread those
// hints as hard restrictions and stall (e.g. "tools=[read_file] but I need to
// write a file…"). Falls back to the raw JSON if it can't be parsed.
func planToChecklist(planJSON string) string {
	var p struct {
		Subtasks []struct {
			ID        string   `json:"id"`
			Goal      string   `json:"goal"`
			Success   string   `json:"success"`
			DependsOn []string `json:"depends_on"`
		} `json:"subtasks"`
	}
	if json.Unmarshal([]byte(planJSON), &p) != nil || len(p.Subtasks) == 0 {
		return planJSON
	}
	var b strings.Builder
	for i, st := range p.Subtasks {
		fmt.Fprintf(&b, "%d. ", i+1)
		if st.ID != "" {
			fmt.Fprintf(&b, "[%s] ", st.ID)
		}
		b.WriteString(strings.TrimSpace(st.Goal))
		if len(st.DependsOn) > 0 {
			fmt.Fprintf(&b, " (after %s)", strings.Join(st.DependsOn, ", "))
		}
		if s := strings.TrimSpace(st.Success); s != "" {
			b.WriteString("\n   done when: " + s)
		}
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

// ExtractPlan pulls a plan from planner output. Preferred shape is a fenced
// ```plan ...``` block, but weak models often drop the label, wrap it as
// ```json, or emit the bare object — so we fall back to scanning for the first
// valid JSON object that carries a "subtasks" key.
func ExtractPlan(assistant string) (string, bool) {
	// 1. Preferred: the fenced ```plan body. The fence delimiters survive even
	//    when the JSON inside is malformed, so repair it here directly — a stray
	//    missing quote desyncs the brace scanner below, so that path can't.
	if m := planBlockRe.FindStringSubmatch(assistant); len(m) >= 2 {
		body := strings.TrimSpace(m[1])
		if json.Valid([]byte(body)) {
			return body, true
		}
		if r := repairJSON(body); json.Valid([]byte(r)) && strings.Contains(r, `"subtasks"`) {
			return r, true
		}
	}
	// 2. No (usable) fence: scan for a subtasks object in the raw text.
	if s, ok := findSubtasksObject(assistant); ok {
		return s, true
	}
	// 3. Last resort: repair the whole text, then re-scan — handles a malformed
	//    bare object whose stray quote desynced the first scan.
	if r := repairJSON(assistant); r != assistant {
		if s, ok := findSubtasksObject(r); ok {
			return s, true
		}
	}
	return "", false
}

// findSubtasksObject returns the first balanced {...} in s that carries a
// "subtasks" key and parses as JSON. If the object is balanced but invalid, a
// bounded repair (see repairJSON) is attempted and re-validated, so weak-model
// JSON is rescued without ever returning unparseable garbage.
func findSubtasksObject(s string) (string, bool) {
	for i := 0; i < len(s); i++ {
		if s[i] != '{' {
			continue
		}
		depth, inStr, esc := 0, false, false
		for j := i; j < len(s); j++ {
			c := s[j]
			if inStr {
				switch {
				case esc:
					esc = false
				case c == '\\':
					esc = true
				case c == '"':
					inStr = false
				}
				continue
			}
			switch c {
			case '"':
				inStr = true
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					cand := s[i : j+1]
					if strings.Contains(cand, `"subtasks"`) {
						if json.Valid([]byte(cand)) {
							return cand, true
						}
						if r := repairJSON(cand); json.Valid([]byte(r)) && strings.Contains(r, `"subtasks"`) {
							return r, true
						}
					}
					j = len(s) // this object isn't it; restart from the next '{'
				}
			}
		}
	}
	return "", false
}

// missingKeyQuoteRe spots an object key that lost its opening quote
// (`,depends_on":` instead of `,"depends_on":`) — a key position is a `{` or `,`
// followed by a bareword and a closing quote+colon. A correctly quoted key has a
// `"` right after the delimiter, so it never matches.
var missingKeyQuoteRe = regexp.MustCompile(`([{,]\s*)([A-Za-z_]\w*)("\s*:)`)

// trailingCommaRe spots a comma immediately before a closing } or ].
var trailingCommaRe = regexp.MustCompile(`,(\s*[}\]])`)

// repairJSON makes a best-effort pass at the two malformations weak models most
// often produce in otherwise-valid JSON: a key missing its opening quote and a
// trailing comma. It is only used as a fallback after strict parsing fails, and
// callers re-validate the result, so a bad repair yields "no plan", not garbage.
func repairJSON(s string) string {
	s = missingKeyQuoteRe.ReplaceAllString(s, `${1}"${2}${3}`)
	s = trailingCommaRe.ReplaceAllString(s, `${1}`)
	return s
}

// ExtractSummary pulls the coder's self-emitted "FINISH: ..." line.
func ExtractSummary(assistant string) (string, bool) {
	m := finishRe.FindStringSubmatch(assistant)
	if len(m) < 2 {
		return "", false
	}
	return strings.TrimSpace(m[1]), true
}
