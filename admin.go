package main

import (
	"embed"
	"encoding/json"
	"io/fs"
	"net/http"
	"strings"
	"time"
)

// Embeds the whole web/ tree (index.html, tailwind.css, fonts.css, fonts/*.woff2).
// Using the directory form means a not-yet-added file (e.g. a tailwind.css you
// build separately) won't break compilation — it just 404s until present.
//go:embed web
var webFS embed.FS

type Admin struct {
	cfg   *Config
	proxy *Proxy
	mgr   *Manager
}

func NewAdmin(cfg *Config, proxy *Proxy, mgr *Manager) *Admin {
	return &Admin{cfg: cfg, proxy: proxy, mgr: mgr}
}

func (a *Admin) Register(mux *http.ServeMux) {
	mux.HandleFunc("/", a.index)

	// Static assets served straight from the embedded web/ dir.
	if static, err := fs.Sub(webFS, "web"); err == nil {
		fileServer := http.FileServer(http.FS(static))
		mux.Handle("/tailwind.css", fileServer)
		mux.Handle("/fonts.css", fileServer)
		mux.Handle("/fonts/", fileServer)
	}

	mux.HandleFunc("/admin/config", a.config)
	mux.HandleFunc("/admin/status", a.status)
	mux.HandleFunc("/admin/discussions", a.discussions)
	mux.HandleFunc("/admin/start", a.start)
	mux.HandleFunc("/admin/stop", a.stop)
	mux.HandleFunc("/admin/engine/test", a.engineTest)
	mux.HandleFunc("/admin/engine/launch", a.engineLaunch)
	mux.HandleFunc("/admin/engine/stop", a.engineStop)
	mux.HandleFunc("/admin/hf/search", a.hfSearch)
	mux.HandleFunc("/admin/hf/files", a.hfFiles)
	mux.HandleFunc("/admin/help/chat", a.helpChat)
}

func (a *Admin) index(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	b, _ := webFS.ReadFile("web/index.html")
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(b)
}

func (a *Admin) config(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, a.cfg.Snapshot())
	case http.MethodPost:
		var n Config
		if err := json.NewDecoder(r.Body).Decode(&n); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := a.cfg.Update(n); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]any{"ok": true})
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// binFor returns the binary path for a role, falling back to the shared default.
func binFor(cfg Config, up Upstream) string {
	if strings.TrimSpace(up.BinPath) != "" {
		return strings.TrimSpace(up.BinPath)
	}
	return cfg.LlamaBin
}

// routerUpstream adapts the router config into an Upstream so the same engine
// machinery can launch a managed router model.
func routerUpstream(cfg Config) Upstream {
	r := cfg.Router
	slots := r.Slots
	if slots <= 0 {
		slots = 1
	}
	return Upstream{
		Name: "router", Engine: r.Engine, BinPath: r.BinPath, BaseURL: r.BaseURL,
		Model: r.Model, Slots: slots, HFRepo: r.HFRepo, HFFile: r.HFFile,
		ModelPath: r.ModelPath, ExtraArgs: r.ExtraArgs,
	}
}

func (a *Admin) upStatus(u Upstream, used, size int, process bool) map[string]any {
	eng := engineByID(u.Engine)
	return map[string]any{
		"configured": u.BaseURL != "" && u.Model != "",
		"reachable":  HealthAt(u.BaseURL, eng.HealthPath),
		"model":      u.Model,
		"base_url":   u.BaseURL,
		"slots_used": used,
		"slots_size": size,
		"process":    process, // a managed engine process is alive for this role
		"engine":     eng.ID,
		"engine_name": eng.Name,
		"bin_path":   u.BinPath,
		"hf_repo":    u.HFRepo,
		"hf_file":    u.HFFile,
	}
}

func (a *Admin) status(w http.ResponseWriter, r *http.Request) {
	cfg := a.cfg.Snapshot()
	pUsed, pSize := a.proxy.SlotStats("planner")
	cUsed, cSize := a.proxy.SlotStats("coder")

	// The router model is the only routing brain, so it is always required.
	routerConfigured := cfg.Router.BaseURL != "" && cfg.Router.Model != ""
	routerReachable := HealthAt(cfg.Router.BaseURL, engineByID(cfg.Router.Engine).HealthPath)

	writeJSON(w, map[string]any{
		"running":     a.proxy.Ready(),
		"phase":       a.proxy.Phase(),    // "running" | "starting" | "stopped"
		"starting":    a.proxy.Starting(), // autostart loading window
		"listen_addr": cfg.ListenAddr,
		"llama_bin":   cfg.LlamaBin,
		"autostart":   cfg.AutoStart,
		"planner":     a.upStatus(cfg.Planner, pUsed, pSize, a.mgr.Running("planner")),
		"coder":       a.upStatus(cfg.Coder, cUsed, cSize, a.mgr.Running("coder")),
		"router": map[string]any{
			"configured": routerConfigured,
			"reachable":  routerReachable,
			"process":    a.mgr.Running("router"),
			"engine":     engineByID(cfg.Router.Engine).ID,
			"bin_path":   cfg.Router.BinPath,
			"hf_repo":    cfg.Router.HFRepo,
			"hf_file":    cfg.Router.HFFile,
		},
		"slot_save_path":     cfg.SlotSavePath,
		"last_route":         a.proxy.LastRoute(),
		"last_route_fallback": a.proxy.LastRouteFallback(),
		"discussions":        a.proxy.store.CountDiscussions(),
	})
}

func (a *Admin) discussions(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, a.proxy.store.ListDiscussions(50))
}

func (a *Admin) start(w http.ResponseWriter, r *http.Request) {
	cfg := a.cfg.Snapshot()
	if !HealthAt(cfg.Planner.BaseURL, engineByID(cfg.Planner.Engine).HealthPath) {
		writeErr(w, http.StatusBadGateway, codedErr("unreachable", "planner", "planner endpoint unreachable"))
		return
	}
	if !HealthAt(cfg.Coder.BaseURL, engineByID(cfg.Coder.Engine).HealthPath) {
		writeErr(w, http.StatusBadGateway, codedErr("unreachable", "coder", "coder endpoint unreachable"))
		return
	}
	a.proxy.Start()
	writeJSON(w, map[string]any{"running": true})
}

func (a *Admin) stop(w http.ResponseWriter, r *http.Request) {
	a.proxy.Stop()
	writeJSON(w, map[string]any{"running": false})
}

// ---- managed engine processes ----

func (a *Admin) engineTest(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Bin string `json:"bin"`
	}
	_ = json.NewDecoder(r.Body).Decode(&in)
	bin := strings.TrimSpace(in.Bin)
	if bin == "" {
		bin = a.cfg.Snapshot().LlamaBin
	}
	ver, err := TestBin(bin)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, map[string]any{"ok": true, "version": ver})
}

func (a *Admin) engineLaunch(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Role string `json:"role"`
	}
	_ = json.NewDecoder(r.Body).Decode(&in)
	if in.Role != "planner" && in.Role != "coder" && in.Role != "router" {
		http.Error(w, jsonErr("role must be planner, coder or router"), http.StatusBadRequest)
		return
	}
	cfg := a.cfg.Snapshot()
	var up Upstream
	switch in.Role {
	case "coder":
		up = cfg.Coder
	case "router":
		up = routerUpstream(cfg)
	default:
		up = cfg.Planner
	}
	if err := a.mgr.Launch(binFor(cfg, up), up); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	// Give it a moment to fail fast (bad binary / args). Model downloads can
	// take minutes, so we don't block on full health here; dashboard polling
	// reports reachability as it comes up.
	time.Sleep(1500 * time.Millisecond)
	if !a.mgr.Running(in.Role) {
		writeErr(w, http.StatusBadGateway, codedErr("exited", in.Role, "%s exited immediately — check the binary path and the engine logs", in.Role))
		return
	}
	writeJSON(w, map[string]any{"ok": true, "running": true})
}

func (a *Admin) engineStop(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Role string `json:"role"`
	}
	_ = json.NewDecoder(r.Body).Decode(&in)
	a.mgr.Stop(in.Role)
	writeJSON(w, map[string]any{"ok": true})
}

// ---- HuggingFace discovery ----

func (a *Admin) hfSearch(w http.ResponseWriter, r *http.Request) {
	q := strings.TrimSpace(r.URL.Query().Get("q"))
	if q == "" {
		writeJSON(w, []HFModel{})
		return
	}
	ggufOnly := r.URL.Query().Get("gguf") == "1"
	res, err := HFSearch(q, ggufOnly)
	if err != nil {
		http.Error(w, jsonErr(err.Error()), http.StatusBadGateway)
		return
	}
	writeJSON(w, res)
}

func (a *Admin) hfFiles(w http.ResponseWriter, r *http.Request) {
	repo := strings.TrimSpace(r.URL.Query().Get("repo"))
	if repo == "" {
		http.Error(w, jsonErr("repo is required"), http.StatusBadRequest)
		return
	}
	files, err := HFFiles(repo)
	if err != nil {
		http.Error(w, jsonErr(err.Error()), http.StatusBadGateway)
		return
	}
	writeJSON(w, files)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func jsonErr(msg string) string {
	b, _ := json.Marshal(map[string]string{"error": msg})
	return string(b)
}
