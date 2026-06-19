package main

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// Manager owns the inference-server child processes launched from the control
// panel (one per role: "planner", "coder"). The engine adapter decides the
// command line; the manager only keeps the process alive and reapable.
type Manager struct {
	mu    sync.Mutex
	procs map[string]*exec.Cmd
}

func NewManager() *Manager { return &Manager{procs: map[string]*exec.Cmd{}} }

// portOf pulls the port out of a base URL like http://127.0.0.1:8080[/...].
func portOf(baseURL string) string {
	s := baseURL
	if i := strings.Index(s, "://"); i >= 0 {
		s = s[i+3:]
	}
	if i := strings.IndexByte(s, '/'); i >= 0 {
		s = s[:i]
	}
	if i := strings.LastIndexByte(s, ':'); i >= 0 {
		return s[i+1:]
	}
	return ""
}

// Launch (re)starts the server process for one role using its engine adapter.
// bin is the absolute path the user provided for that engine's binary (for
// SGLang this is the Python interpreter; the adapter adds the module args).
func (m *Manager) Launch(bin string, up Upstream) error {
	if strings.TrimSpace(bin) == "" {
		return codedErr("bin_missing", up.Name, "binary path is not set for %s", up.Name)
	}
	port := portOf(up.BaseURL)
	if port == "" {
		return codedErr("bad_port", up.Name, "could not read a port from base URL %q", up.BaseURL)
	}

	eng := engineByID(up.Engine)
	args, env, err := eng.Build(up, port)
	if err != nil {
		return err
	}
	if strings.TrimSpace(up.ExtraArgs) != "" {
		args = append(args, strings.Fields(up.ExtraArgs)...)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if c := m.procs[up.Name]; c != nil && c.Process != nil {
		_ = c.Process.Kill()
		delete(m.procs, up.Name)
	}

	cmd := exec.Command(bin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if len(env) > 0 {
		cmd.Env = append(os.Environ(), env...)
	}
	if err := cmd.Start(); err != nil {
		return codedErr("start_failed", up.Name, "start %s (%s): %v", up.Name, eng.Name, err)
	}
	m.procs[up.Name] = cmd

	// Reap on exit so Running() reflects reality.
	go func(name string, c *exec.Cmd) {
		_ = c.Wait()
		m.mu.Lock()
		if m.procs[name] == c {
			delete(m.procs, name)
		}
		m.mu.Unlock()
	}(up.Name, cmd)
	return nil
}

func (m *Manager) Stop(role string) {
	m.mu.Lock()
	c := m.procs[role]
	delete(m.procs, role)
	m.mu.Unlock()
	if c != nil && c.Process != nil {
		_ = c.Process.Kill()
	}
}

func (m *Manager) StopAll() {
	m.mu.Lock()
	cmds := make([]*exec.Cmd, 0, len(m.procs))
	for _, c := range m.procs {
		cmds = append(cmds, c)
	}
	m.procs = map[string]*exec.Cmd{}
	m.mu.Unlock()
	for _, c := range cmds {
		if c.Process != nil {
			_ = c.Process.Kill()
		}
	}
}

func (m *Manager) Running(role string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	c := m.procs[role]
	return c != nil && c.Process != nil
}

// TestBin runs `<bin> --version` and returns the reported version text. Works
// for llama-server, vllm, text-generation-launcher, ollama and python.
func TestBin(bin string) (string, error) {
	if strings.TrimSpace(bin) == "" {
		return "", codedErr("bin_empty", "", "path is empty")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, bin, "--version").CombinedOutput()
	text := strings.TrimSpace(string(out))
	if err != nil {
		// Some binaries print the version to stderr and exit non-zero; any
		// output means the binary is runnable, so treat it as success.
		if text != "" {
			return firstLine(text), nil
		}
		return "", err
	}
	if text == "" {
		text = "ok (no version output)"
	}
	return firstLine(text), nil
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return s
}

// WaitHealthy polls an upstream's health path until it answers or the timeout hits.
func WaitHealthy(baseURL, path string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if HealthAt(baseURL, path) {
			return true
		}
		time.Sleep(500 * time.Millisecond)
	}
	return false
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
