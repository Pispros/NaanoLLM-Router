package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

func main() {
	configPath := flag.String("config", "llmrouter.json", "path to the config file")
	dataDir := flag.String("data", ".", "data directory (sqlite db lives here, created if missing)")
	flag.Parse()

	if err := os.MkdirAll(*dataDir, 0o755); err != nil {
		log.Fatalf("data dir: %v", err)
	}

	cfg, err := LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	// The SQLite store (text + metadata only) is created or reused at startup.
	dbPath := filepath.Join(*dataDir, "llmrouter.db")
	store, err := OpenStore(dbPath)
	if err != nil {
		log.Fatalf("store: %v", err)
	}
	defer store.Close()

	mgr := NewManager()
	proxy := NewProxy(cfg, store)
	admin := NewAdmin(cfg, proxy, mgr)

	// Stop managed llama-server processes on Ctrl-C / SIGTERM.
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
		<-sig
		log.Printf("shutting down — stopping managed llama-server processes")
		mgr.StopAll()
		_ = store.Close()
		os.Exit(0)
	}()

	// Optional: at boot, launch planner+coder and arm /v1 once both are healthy.
	if snap := cfg.Snapshot(); snap.AutoStart {
		go autoStart(snap, mgr, proxy)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", proxy.ServeHTTP)
	mux.HandleFunc("/v1/models", proxy.Models)
	admin.Register(mux)

	addr := cfg.Snapshot().ListenAddr
	log.Printf("naanollm-router listening on %s", addr)
	log.Printf("  control panel : http://localhost%s/", addr)
	log.Printf("  OpenAI base   : http://localhost%s/v1", addr)
	log.Printf("  database      : %s", dbPath)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}

// autoStart launches both managed backends with their per-role engine binary,
// waits for them to become healthy (model downloads can take a while), then
// arms the /v1 endpoint.
func autoStart(snap Config, mgr *Manager, proxy *Proxy) {
	for _, up := range []Upstream{snap.Planner, snap.Coder} {
		bin := up.BinPath
		if strings.TrimSpace(bin) == "" {
			bin = snap.LlamaBin
		}
		if err := mgr.Launch(bin, up); err != nil {
			log.Printf("autostart %s: %v", up.Name, err)
		}
	}
	if snap.Router.Mode == "llm" {
		ru := routerUpstream(snap)
		bin := ru.BinPath
		if strings.TrimSpace(bin) == "" {
			bin = snap.LlamaBin
		}
		if err := mgr.Launch(bin, ru); err != nil {
			log.Printf("autostart router: %v", err)
		}
	}
	okP := WaitHealthy(snap.Planner.BaseURL, engineByID(snap.Planner.Engine).HealthPath, 5*time.Minute)
	okC := WaitHealthy(snap.Coder.BaseURL, engineByID(snap.Coder.Engine).HealthPath, 5*time.Minute)
	if okP && okC {
		_ = proxy.Start()
		log.Printf("autostart: planner+coder healthy, /v1 armed")
		return
	}
	log.Printf("autostart: upstreams not healthy (planner=%v coder=%v) — arm manually from the panel", okP, okC)
}
