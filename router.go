package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// execIntent matches a user signalling "go ahead and do it" (FR + EN).
var execIntent = regexp.MustCompile(`(?i)\b(ex[eé]cut|impl[eé]ment|fais[- ]?le|vas[- ]?y|go|lance|proceed|act\b|run it|do it|apply|build it|ship it)\b`)

// planIntent matches a user asking to think / plan / analyse.
var planIntent = regexp.MustCompile(`(?i)\b(plan|analys|brainstorm|r[eé]fl[eé]ch|architect|design|comment|pourquoi|explique|strat[eé]gie)\b`)

// Route decides which role handles this turn.
//   - forced model name ("planner"/"coder") always wins
//   - otherwise: pending plan + execution intent => coder
//   - otherwise: planner (analysis, questions, follow-ups, new plans)
//   - mode "llm": the tiny model only breaks genuinely ambiguous ties
func Route(cfg Config, store *Store, discID int64, requested, userMsg string) string {
	switch strings.ToLower(strings.TrimSpace(requested)) {
	case "planner":
		return "planner"
	case "coder":
		return "coder"
	}

	pending := store.HasPendingPlan(discID)
	exec := execIntent.MatchString(userMsg)
	plan := planIntent.MatchString(userMsg)

	switch {
	case pending && exec:
		return "coder"
	case plan && !exec:
		return "planner"
	case !pending:
		return "planner" // nothing to execute yet
	}

	// Ambiguous: pending plan, no clear signal either way.
	if cfg.Router.Mode == "llm" {
		if r := classifyLLM(cfg, userMsg, pending); r != "" {
			return r
		}
	}
	// Safe default: keep talking to the planner rather than mutating files.
	return "planner"
}

// classifyLLM asks the tiny model for a single token, constrained by a GBNF
// grammar so it literally cannot answer anything but "planner" or "coder".
func classifyLLM(cfg Config, userMsg string, pending bool) string {
	body := map[string]any{
		"model": cfg.Router.Model,
		"messages": []map[string]string{
			{"role": "system", "content": routerSystemPrompt},
			{"role": "user", "content": "pending_plan=" + boolStr(pending) + "\nuser: " + userMsg},
		},
		"temperature": 0,
		"max_tokens":  1,
		"grammar":     `root ::= "planner" | "coder"`,
		"cache_prompt": true,
	}
	b, _ := json.Marshal(body)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", cfg.Router.BaseURL+"/v1/chat/completions", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.NewDecoder(resp.Body).Decode(&out) != nil || len(out.Choices) == 0 {
		return ""
	}
	switch strings.TrimSpace(out.Choices[0].Message.Content) {
	case "coder":
		return "coder"
	case "planner":
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

const routerSystemPrompt = `You are a router. Read the user's latest message and whether a plan is pending.
Answer with exactly one word: "coder" if the user wants the pending plan executed/implemented, otherwise "planner".`
