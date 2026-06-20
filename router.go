package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

// The router has exactly ONE brain: the tiny GBNF-constrained model. Every turn
// is classified by it. The keyword rules below are NOT a routing mode — they are
// an emergency fallback used only when the model can't answer (unreachable,
// timeout, empty/invalid output). When the fallback fires, Route reports it so
// the UI can flag that routing is running degraded.

// execFallback matches a user telling a pending plan to go ahead (FR + EN). Verb
// stems consume the rest of the word with \w* so the trailing \b lands at the
// real word end (without it, "ex[eé]cut\b" can't match "execute"/"exécute").
// This is touched only on the fallback path.
var execFallback = regexp.MustCompile(`(?i)\b(ex[eé]cut\w*|impl[eé]ment\w*|fais[- ]?le|vas[- ]?y|go|lance\w*|proceed\w*|run it|do it|apply|build it|ship it)\b`)

// Route decides which role handles this turn.
//   - an explicit model alias ("planner"/"coder") from the caller always wins
//   - a tool-loop continuation stays on the coder (never re-routed mid-turn)
//   - otherwise the tiny router model classifies the turn (the default path)
//   - only if the model fails do the keyword rules pick a role (fallback=true)
//
// lastAssistant is a trimmed snapshot of the previous assistant message, passed
// to the router model as light context. It returns the chosen role and whether
// the keyword fallback had to be used.
func Route(cfg Config, store *Store, discID int64, requested, userMsg, lastAssistant string, toolContinuation bool) (role string, fallback bool) {
	switch strings.ToLower(strings.TrimSpace(requested)) {
	case "planner":
		return "planner", false
	case "coder":
		return "coder", false
	}

	// A continuation is the same turn still executing. Only the coder has tools,
	// so it belongs to the coder finishing its work — re-routing it is what made
	// execution oscillate with the planner and loop. Keep it on the coder.
	if toolContinuation {
		return "coder", false
	}

	pending := store.HasPendingPlan(discID)

	// Default path: ask the model. This is the only routing brain.
	if r, ok := classifyLLM(cfg, userMsg, lastAssistant, pending); ok {
		return r, false
	}

	// Fallback path: the model didn't answer. Keep it minimal and safe —
	// only hand off to the coder when there is a plan to execute and the user
	// clearly asked to run it; otherwise stay on the planner (never mutate
	// files on a guess).
	if pending && execFallback.MatchString(userMsg) {
		return "coder", true
	}
	return "planner", true
}

// classifyLLM asks the tiny model for one word, constrained by a GBNF grammar
// so it can only answer "planner" or "coder". The bool is false when the model
// is unreachable, errors, times out, or returns anything we can't read as a
// role — that is the signal to fall back. Every failure mode is logged with the
// raw reply so a misbehaving router model (wrong type, thinking preamble, etc.)
// is visible instead of silently degrading to the keyword rules.
func classifyLLM(cfg Config, userMsg, lastAssistant string, pending bool) (string, bool) {
	// The coder only ever executes an existing plan. With no pending plan there
	// is nothing to execute, so "coder" is not a valid answer — constrain the
	// grammar to "planner" alone. The model still decides; we just never offer
	// it a structurally impossible option. With a pending plan it gets the real
	// planner/coder choice.
	grammar := `root ::= "planner" | "coder"`
	if !pending {
		grammar = `root ::= "planner"`
	}

	// Light context: the tail of the previous assistant message helps the model
	// tell "the planner just proposed a plan, now run it" from a fresh request.
	// Bounded so routing stays a cheap, near-instant call.
	user := "pending_plan=" + boolStr(pending) + "\n"
	if lastAssistant != "" {
		user += "last_assistant: " + clip(lastAssistant, 400) + "\n"
	}
	user += "user: " + userMsg

	body := map[string]any{
		"model": cfg.Router.Model,
		"messages": []map[string]string{
			{"role": "system", "content": routerSystemPrompt},
			{"role": "user", "content": user},
		},
		"temperature": 0,
		// "planner"/"coder" can tokenize to more than one token, so 1 truncates
		// the word and the reply never matches. A small budget is plenty; the
		// grammar keeps the model to exactly one of the two words.
		"max_tokens":   8,
		"grammar":      grammar,
		"cache_prompt": true,
	}
	b, _ := json.Marshal(body)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", cfg.Router.BaseURL+"/v1/chat/completions", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Printf("router classify: request to %s failed: %v", cfg.Router.BaseURL, err)
		return "", false
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		log.Printf("router classify: %s returned HTTP %d: %s", cfg.Router.BaseURL, resp.StatusCode, snip(raw))
		return "", false
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(raw, &out) != nil || len(out.Choices) == 0 {
		log.Printf("router classify: unreadable reply from %s: %s", cfg.Router.BaseURL, snip(raw))
		return "", false
	}
	reply := out.Choices[0].Message.Content
	role := normalizeRouterReply(reply)
	if debugOn {
		log.Printf("router classify: model=%s reply=%q -> %q", cfg.Router.Model, reply, role)
	}
	if role == "" {
		// A correct reply is "planner"/"coder". Anything else usually means the
		// router endpoint is serving the wrong kind of model (e.g. an embedding
		// model can't follow the instruction) or the grammar wasn't applied.
		log.Printf("router classify: reply %q is neither planner nor coder — check that %s serves a generative instruct model; falling back",
			strings.TrimSpace(reply), cfg.Router.Model)
		return "", false
	}
	return role, true
}

// normalizeRouterReply maps a router reply to a role, tolerant of case,
// surrounding whitespace, and a Qwen-style <think>…</think> preamble (kept as a
// safety net for the case the grammar isn't honored by the engine).
func normalizeRouterReply(s string) string {
	s = strings.ToLower(s)
	if i := strings.LastIndex(s, "</think>"); i >= 0 {
		s = s[i+len("</think>"):]
	}
	s = strings.TrimSpace(s)
	switch {
	case strings.Contains(s, "coder"):
		return "coder"
	case strings.Contains(s, "planner"):
		return "planner"
	}
	return ""
}

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// clip truncates s to at most n bytes, on a rune boundary, adding an ellipsis.
func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := n
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…"
}

const routerSystemPrompt = `You are a router for a coding assistant. Read the user's latest message and the pending_plan flag.

pending_plan=true  : an approved plan is waiting to be executed.
pending_plan=false : no plan exists yet, so there is nothing to execute.

Answer with exactly one word:
- "coder"   only when pending_plan=true AND the user is telling you to execute / implement / run the existing plan.
- "planner" in every other case — writing or revising a plan, analysis, questions, and ALWAYS when pending_plan=false.`
