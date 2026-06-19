package main

import (
	"strconv"
	"strings"
)

// Engine is an inference-server adapter. Each one knows how to turn a role's
// Upstream config into a command line for that server, where its health probe
// lives, and whether it consumes GGUF files (which changes model discovery).
type Engine struct {
	ID         string
	Name       string
	HealthPath string
	UsesGGUF   bool
	// Build returns the process arguments and any extra environment variables
	// for launching one role on the given port.
	Build func(up Upstream, port string) (args []string, env []string, err error)
}

// modelRef is the model identifier most engines accept directly: a HuggingFace
// repo id when set, otherwise a local path.
func modelRef(up Upstream) (string, error) {
	if strings.TrimSpace(up.HFRepo) != "" {
		return strings.TrimSpace(up.HFRepo), nil
	}
	if strings.TrimSpace(up.ModelPath) != "" {
		return strings.TrimSpace(up.ModelPath), nil
	}
	return "", codedErr("no_model", up.Name, "no model selected for %s — pick one in the Engine tab", up.Name)
}

var engines = map[string]Engine{
	// llama.cpp — the GGUF reference server. Pins KV per slot, downloads GGUF
	// from HuggingFace itself via --hf-repo/--hf-file.
	"llamacpp": {
		ID: "llamacpp", Name: "llama.cpp", HealthPath: "/health", UsesGGUF: true,
		Build: func(up Upstream, port string) ([]string, []string, error) {
			var args []string
			switch {
			case strings.TrimSpace(up.HFRepo) != "":
				args = append(args, "--hf-repo", strings.TrimSpace(up.HFRepo))
				if strings.TrimSpace(up.HFFile) != "" {
					args = append(args, "--hf-file", strings.TrimSpace(up.HFFile))
				}
			case strings.TrimSpace(up.ModelPath) != "":
				args = append(args, "-m", strings.TrimSpace(up.ModelPath))
			default:
				return nil, nil, codedErr("no_model", up.Name, "no model selected for %s", up.Name)
			}
			if strings.TrimSpace(up.Model) != "" {
				args = append(args, "--alias", up.Model)
			}
			args = append(args, "--host", hostOf(up.BaseURL), "--port", port,
				"-np", strconv.Itoa(maxInt(up.Slots, 1)), "--jinja")
			return args, nil, nil
		},
	},

	// vLLM — high-throughput GPU server. `vllm serve <repo>`; pulls from HF.
	"vllm": {
		ID: "vllm", Name: "vLLM", HealthPath: "/health", UsesGGUF: false,
		Build: func(up Upstream, port string) ([]string, []string, error) {
			ref, err := modelRef(up)
			if err != nil {
				return nil, nil, err
			}
			args := []string{"serve", ref, "--host", hostOf(up.BaseURL), "--port", port}
			if strings.TrimSpace(up.Model) != "" {
				args = append(args, "--served-model-name", up.Model)
			}
			if up.Slots > 0 {
				args = append(args, "--max-num-seqs", strconv.Itoa(up.Slots))
			}
			return args, nil, nil
		},
	},

	// SGLang — launched as a Python module, so the "binary" is the interpreter
	// (e.g. /path/to/venv/bin/python). Pulls from HF.
	"sglang": {
		ID: "sglang", Name: "SGLang", HealthPath: "/health", UsesGGUF: false,
		Build: func(up Upstream, port string) ([]string, []string, error) {
			ref, err := modelRef(up)
			if err != nil {
				return nil, nil, err
			}
			args := []string{"-m", "sglang.launch_server", "--model-path", ref,
				"--host", hostOf(up.BaseURL), "--port", port}
			if strings.TrimSpace(up.Model) != "" {
				args = append(args, "--served-model-name", up.Model)
			}
			return args, nil, nil
		},
	},

	// Text Generation Inference (HuggingFace). `text-generation-launcher`.
	"tgi": {
		ID: "tgi", Name: "Text Generation Inference", HealthPath: "/health", UsesGGUF: false,
		Build: func(up Upstream, port string) ([]string, []string, error) {
			ref, err := modelRef(up)
			if err != nil {
				return nil, nil, err
			}
			args := []string{"--model-id", ref, "--hostname", hostOf(up.BaseURL), "--port", port}
			if up.Slots > 0 {
				args = append(args, "--max-concurrent-requests", strconv.Itoa(up.Slots))
			}
			return args, nil, nil
		},
	},

	// Ollama — a daemon bound to this role's port via OLLAMA_HOST. The model is
	// selected per request by alias (an Ollama tag the user has already pulled).
	"ollama": {
		ID: "ollama", Name: "Ollama", HealthPath: "/api/version", UsesGGUF: false,
		Build: func(up Upstream, port string) ([]string, []string, error) {
			env := []string{"OLLAMA_HOST=" + hostOf(up.BaseURL) + ":" + port}
			return []string{"serve"}, env, nil
		},
	},
}

func engineByID(id string) Engine {
	if e, ok := engines[strings.TrimSpace(id)]; ok {
		return e
	}
	return engines["llamacpp"]
}

// isLlamaCpp reports whether a role runs on llama.cpp (the only engine that
// understands id_slot / cache_prompt and KV slot save/restore).
func isLlamaCpp(id string) bool {
	id = strings.TrimSpace(id)
	return id == "" || id == "llamacpp"
}

// hostOf extracts the host from a base URL, defaulting to 127.0.0.1.
func hostOf(baseURL string) string {
	s := baseURL
	if i := strings.Index(s, "://"); i >= 0 {
		s = s[i+3:]
	}
	if i := strings.IndexByte(s, '/'); i >= 0 {
		s = s[:i]
	}
	if i := strings.LastIndexByte(s, ':'); i >= 0 {
		s = s[:i]
	}
	if strings.TrimSpace(s) == "" {
		return "127.0.0.1"
	}
	return s
}
