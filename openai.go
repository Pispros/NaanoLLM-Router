package main

import "encoding/json"

// Message is an OpenAI chat message. Content can be a string or an array of
// content parts; we keep it as RawMessage and extract text when needed.
type Message struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
}

// Text returns a best-effort plain-text view of the message content.
func (m Message) Text() string {
	if len(m.Content) == 0 {
		return ""
	}
	// string form
	var s string
	if json.Unmarshal(m.Content, &s) == nil {
		return s
	}
	// array of {type:"text", text:"..."} parts
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(m.Content, &parts) == nil {
		var out string
		for _, p := range parts {
			out += p.Text
		}
		return out
	}
	return ""
}

func TextMessage(role, text string) Message {
	b, _ := json.Marshal(text)
	return Message{Role: role, Content: b}
}

// ChatRequest is the subset of the OpenAI body we read. Everything else (tools,
// temperature, top_p, ...) is preserved by forwarding the raw map unchanged.
type ChatRequest struct {
	Model    string          `json:"model"`
	Messages []Message       `json:"messages"`
	Stream   bool            `json:"stream"`
	Tools    json.RawMessage `json:"tools,omitempty"`
	// PromptCacheKey is the OpenAI stable client identifier (it replaces the
	// legacy `user` field). Clients that mint one per conversation thread — Zed
	// emits a fixed UUID for every turn of a thread — hand us a content-free,
	// collision-free conversation id. When present it is the primary discussion
	// key; see Store.ResolveDiscussion.
	PromptCacheKey string `json:"prompt_cache_key,omitempty"`
}
