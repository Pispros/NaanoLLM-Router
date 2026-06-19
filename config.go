package main

import (
	"encoding/json"
	"os"
	"sync"
)

// Upstream is one llama-server backend (one resident model = one role).
type Upstream struct {
	Name      string `json:"name"`       // human label, e.g. "planner"
	Engine    string `json:"engine"`     // inference engine adapter: llamacpp|vllm|sglang|tgi|ollama
	BinPath   string `json:"bin_path"`   // absolute path to this engine's binary (falls back to Config.LlamaBin)
	BaseURL   string `json:"base_url"`   // e.g. http://127.0.0.1:8080
	Model     string `json:"model"`      // model alias the upstream answers to
	Slots     int    `json:"slots"`      // number of parallel slots (-np) this upstream exposes
	HFRepo    string `json:"hf_repo"`    // HuggingFace repo to download+run
	HFFile    string `json:"hf_file"`    // specific .gguf file in that repo (quant, llama.cpp only)
	ModelPath string `json:"model_path"` // local model path (used when HFRepo is empty)
	ExtraArgs string `json:"extra_args"` // extra flags appended to the launch command, space-separated
}

// RouterCfg decides which role handles a turn.
//
//	mode "rule" : pure state + keyword logic, zero VRAM, deterministic.
//	mode "llm"  : rule first, tiny model (GBNF-constrained) only for ambiguous turns.
type RouterCfg struct {
	Mode      string `json:"mode"`       // "rule" | "llm"
	BaseURL   string `json:"base_url"`   // tiny router model endpoint (mode "llm")
	Model     string `json:"model"`      // tiny router model alias
	Engine    string `json:"engine"`     // inference engine adapter for a managed router model
	BinPath   string `json:"bin_path"`   // absolute path to that engine's binary (falls back to Config.LlamaBin)
	HFRepo    string `json:"hf_repo"`    // HuggingFace repo for the router model
	HFFile    string `json:"hf_file"`    // specific .gguf file (quant, llama.cpp only)
	ModelPath string `json:"model_path"` // local model path (used when HFRepo is empty)
	ExtraArgs string `json:"extra_args"` // extra launch flags, space-separated
	Slots     int    `json:"slots"`      // parallel slots for the router (routing is light; 1 is enough)
}

type Config struct {
	ListenAddr   string    `json:"listen_addr"`    // where the OpenAI API is exposed, e.g. :4000
	SlotSavePath string    `json:"slot_save_path"` // dir for KV .bin (must match llama-server --slot-save-path)
	LlamaBin     string    `json:"llama_bin"`      // path to the llama-server binary (for managed launch)
	AutoStart    bool      `json:"autostart"`      // launch planner+coder and arm /v1 at boot
	Planner      Upstream  `json:"planner"`
	Coder        Upstream  `json:"coder"`
	Router       RouterCfg `json:"router"`

	path string        // backing file, not serialized
	mu   *sync.RWMutex // pointer so Snapshot() doesn't copy a lock
}

func DefaultConfig(path string) *Config {
	return &Config{
		ListenAddr:   ":4000",
		SlotSavePath: "",
		Planner:      Upstream{Name: "planner", Engine: "llamacpp", BaseURL: "http://127.0.0.1:8080", Model: "qwen3.5-9b", Slots: 4},
		Coder:        Upstream{Name: "coder", Engine: "llamacpp", BaseURL: "http://127.0.0.1:8081", Model: "qwen2.5-coder-7b", Slots: 4},
		Router:       RouterCfg{Mode: "rule", BaseURL: "http://127.0.0.1:8082", Model: "qwen3-0.6b", Engine: "llamacpp", Slots: 1},
		path:         path,
		mu:           &sync.RWMutex{},
	}
}

func LoadConfig(path string) (*Config, error) {
	c := DefaultConfig(path)
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return c, c.Save() // write defaults on first run
		}
		return nil, err
	}
	if err := json.Unmarshal(b, c); err != nil {
		return nil, err
	}
	c.path = path
	return c, nil
}

func (c *Config) Save() error {
	c.mu.RLock()
	b, err := json.MarshalIndent(c, "", "  ")
	c.mu.RUnlock()
	if err != nil {
		return err
	}
	return os.WriteFile(c.path, b, 0o644)
}

// Snapshot returns a copy safe to read without holding the lock.
func (c *Config) Snapshot() Config {
	c.mu.RLock()
	defer c.mu.RUnlock()
	cp := *c
	return cp
}

// Update applies a partial config (from the UI) and persists it.
func (c *Config) Update(n Config) error {
	c.mu.Lock()
	c.ListenAddr = n.ListenAddr
	c.SlotSavePath = n.SlotSavePath
	c.LlamaBin = n.LlamaBin
	c.AutoStart = n.AutoStart
	c.Planner = n.Planner
	c.Coder = n.Coder
	c.Router = n.Router
	c.mu.Unlock()
	return c.Save()
}

func (c *Config) role(name string) Upstream {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if name == "coder" {
		return c.Coder
	}
	return c.Planner
}
