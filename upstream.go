package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Forward calls the chosen upstream. It starts from the original request map
// (so tools, temperature, top_p, etc. pass through untouched), then overrides
// messages, model, id_slot and cache_prompt. The response is streamed straight
// back to the client; a tee captures the assistant output for persistence.
//
// Returns the full assistant text AND any tool_calls (as raw OpenAI JSON) once
// the response completes. The tool_calls are needed because agentic IDE clients
// (Cline, Roo, JetBrains, ...) emit turns that are tool_calls with little or no
// text; those calls are what makes such a turn identifiable across requests.
func Forward(ctx context.Context, up Upstream, slot int, raw map[string]any, msgs []Message, stream bool, w io.Writer, flush func()) (string, json.RawMessage, error) {
	if msgs != nil {
		raw["messages"] = msgs // nil => forward the client's messages untouched (utility passthrough)
	}
	raw["model"] = up.Model
	raw["stream"] = stream
	if isLlamaCpp(up.Engine) {
		raw["cache_prompt"] = true // reuse a warm prefix (llama.cpp)
		if slot >= 0 {
			raw["id_slot"] = slot // pin this (role, discussion)'s slot; slot<0 = let the engine choose
		}
	}

	b, _ := json.Marshal(raw)
	log.Printf("FWD -> %s/v1/chat/completions model=%v stream=%v id_slot=%v cache_prompt=%v bytes=%d",
		up.BaseURL, raw["model"], raw["stream"], raw["id_slot"], raw["cache_prompt"], len(b))
	req, err := http.NewRequestWithContext(ctx, "POST", up.BaseURL+"/v1/chat/completions", bytes.NewReader(b))
	if err != nil {
		return "", nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(resp.Body)
		log.Printf("FWD <- %s status=%s body=%s", up.Name, resp.Status, snip(msg))
		return "", nil, fmt.Errorf("upstream %s: %s: %s", up.Name, resp.Status, string(msg))
	}

	if !stream {
		body, _ := io.ReadAll(resp.Body)
		w.Write(body)
		text, tools := contentAndToolsFromResponse(body)
		return text, tools, nil
	}

	// Streaming: forward every byte verbatim while reassembling delta.content
	// and delta.tool_calls so we can persist the turn's identity fingerprint.
	var sa streamAcc
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		w.Write(line)
		w.Write([]byte("\n"))
		if flush != nil {
			flush()
		}
		if bytes.HasPrefix(line, []byte("data: ")) {
			payload := bytes.TrimSpace(line[len("data: "):])
			if bytes.Equal(payload, []byte("[DONE]")) {
				continue
			}
			sa.add(payload)
		}
	}
	text, tools := sa.result()
	return text, tools, sc.Err()
}

// streamAcc reassembles a streamed assistant message: content is concatenated,
// and tool_calls are merged by their index (id/name arrive once, arguments are
// streamed in fragments). It mirrors how OpenAI-style servers chunk responses.
type streamAcc struct {
	text  strings.Builder
	tools []*tcAcc
	byIdx map[int]int
}

type tcAcc struct {
	id   string
	name string
	args strings.Builder
}

func (a *streamAcc) add(payload []byte) {
	var chunk struct {
		Choices []struct {
			Delta struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					Index    int    `json:"index"`
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if json.Unmarshal(payload, &chunk) != nil || len(chunk.Choices) == 0 {
		return
	}
	d := chunk.Choices[0].Delta
	a.text.WriteString(d.Content)
	for _, tc := range d.ToolCalls {
		if a.byIdx == nil {
			a.byIdx = map[int]int{}
		}
		i, ok := a.byIdx[tc.Index]
		if !ok {
			i = len(a.tools)
			a.byIdx[tc.Index] = i
			a.tools = append(a.tools, &tcAcc{})
		}
		if tc.ID != "" {
			a.tools[i].id = tc.ID
		}
		if tc.Function.Name != "" {
			a.tools[i].name = tc.Function.Name
		}
		a.tools[i].args.WriteString(tc.Function.Arguments)
	}
}

func (a *streamAcc) result() (string, json.RawMessage) {
	if len(a.tools) == 0 {
		return a.text.String(), nil
	}
	type fn struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	}
	type call struct {
		ID       string `json:"id,omitempty"`
		Type     string `json:"type"`
		Function fn     `json:"function"`
	}
	out := make([]call, len(a.tools))
	for i, t := range a.tools {
		out[i] = call{ID: t.id, Type: "function", Function: fn{Name: t.name, Arguments: t.args.String()}}
	}
	b, _ := json.Marshal(out)
	return a.text.String(), b
}

func contentAndToolsFromResponse(body []byte) (string, json.RawMessage) {
	var out struct {
		Choices []struct {
			Message struct {
				Content   string          `json:"content"`
				ToolCalls json.RawMessage `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(body, &out) == nil && len(out.Choices) > 0 {
		return out.Choices[0].Message.Content, out.Choices[0].Message.ToolCalls
	}
	return "", nil
}

// Health pings an upstream's /health (llama-server and most OpenAI servers).
func Health(baseURL string) bool { return HealthAt(baseURL, "/health") }

// HealthAt pings an arbitrary health path (engines differ: Ollama uses
// /api/version, the rest use /health).
func HealthAt(baseURL, path string) bool {
	if baseURL == "" {
		return false
	}
	if path == "" {
		path = "/health"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", baseURL+path, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// SaveSlot / RestoreSlot persist this role's KV to/from a .bin on disk, used
// only when more discussions are active than there are slots, or to survive a
// reload. Requires llama-server started with --slot-save-path.
func SaveSlot(up Upstream, slot int, filename string) error {
	return slotAction(up, slot, "save", filename)
}
func RestoreSlot(up Upstream, slot int, filename string) error {
	return slotAction(up, slot, "restore", filename)
}

func slotAction(up Upstream, slot int, action, filename string) error {
	u := fmt.Sprintf("%s/slots/%d?action=%s", up.BaseURL, slot, action)
	body, _ := json.Marshal(map[string]string{"filename": filename})
	req, _ := http.NewRequest("POST", u, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("slot %s failed: %s", action, resp.Status)
	}
	return nil
}

func safeFilename(parts ...string) string {
	return url.PathEscape(strings.Join(parts, "_")) + ".bin"
}
