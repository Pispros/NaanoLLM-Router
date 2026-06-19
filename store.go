package main

import (
	"database/sql"
	"encoding/json"
	"time"

	_ "modernc.org/sqlite"
)

// Store holds TEXT and METADATA only. The KV-cache itself never lives here —
// it stays in llama.cpp hot slots (RAM/VRAM) or, optionally, in .bin files on
// disk referenced by slot_file. SQLite is the durable, reconstructable truth.
type Store struct {
	db *sql.DB
}

const schema = `
CREATE TABLE IF NOT EXISTS discussions (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  anchor_key  TEXT NOT NULL,                 -- normalized first-user hash (first-turn dedup only)
  system_json TEXT NOT NULL,                 -- last seen system messages (display/debug; NOT identity)
  tools_json  TEXT NOT NULL,                 -- last seen tools array (display/debug; NOT identity)
  created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_discussions_anchor ON discussions(anchor_key);
CREATE TABLE IF NOT EXISTS turns (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  discussion_id INTEGER NOT NULL,
  seq           INTEGER NOT NULL,
  role          TEXT NOT NULL,               -- planner | coder
  user_msg      TEXT,
  assistant_msg TEXT,
  anchor_sig    TEXT NOT NULL DEFAULT '',    -- per-turn identity fingerprint (text + tool_calls)
  created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_disc ON turns(discussion_id, seq);
CREATE TABLE IF NOT EXISTS plans (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  discussion_id INTEGER NOT NULL,
  plan_json     TEXT NOT NULL,
  consumed      INTEGER NOT NULL DEFAULT 0,  -- 1 once dispatched to the coder
  created_at    INTEGER NOT NULL
);
-- One append-only message list PER (discussion, role): this is the per-model
-- context the user described. The matching KV lives in slot_file / a hot slot.
CREATE TABLE IF NOT EXISTS model_contexts (
  discussion_id INTEGER NOT NULL,
  role          TEXT NOT NULL,
  messages_json TEXT NOT NULL DEFAULT '[]',
  slot_file     TEXT NOT NULL DEFAULT '',    -- path of the persisted KV .bin, if any
  last_used     INTEGER NOT NULL,
  PRIMARY KEY (discussion_id, role)
);
-- Coder's self-emitted FINISH summary, waiting to be appended to the planner.
CREATE TABLE IF NOT EXISTS pending_summaries (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  discussion_id INTEGER NOT NULL,
  summary       TEXT NOT NULL,
  consumed      INTEGER NOT NULL DEFAULT 0,
  created_at    INTEGER NOT NULL
);
`

func OpenStore(path string) (*Store, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1) // sqlite: serialize writers, avoids "database is locked"
	if err := migrate(db); err != nil {
		return nil, err
	}
	if _, err := db.Exec(schema); err != nil {
		return nil, err
	}
	return &Store{db: db}, nil
}

// migrate brings an existing DB up to the current schema without data loss. It
// runs before the schema DDL: the discussions rebuild needs the old table gone
// before CREATE, and the turns column add must precede any new inserts.
func migrate(db *sql.DB) error {
	if err := migrateDiscussions(db); err != nil {
		return err
	}
	if tableExists(db, "turns") && !columnExists(db, "turns", "anchor_sig") {
		if _, err := db.Exec(`ALTER TABLE turns ADD COLUMN anchor_sig TEXT NOT NULL DEFAULT ''`); err != nil {
			return err
		}
	}
	return nil
}

// migrateDiscussions upgrades the pre-transcript schema (discussions keyed by a
// UNIQUE prefix_hash) to the new anchor_key shape, preserving existing rows. It
// runs before the schema DDL so the recreated table picks up the new columns.
func migrateDiscussions(db *sql.DB) error {
	if !tableExists(db, "discussions") {
		return nil // fresh DB; schema below creates the new shape
	}
	if columnExists(db, "discussions", "anchor_key") {
		return nil // already migrated
	}
	if !columnExists(db, "discussions", "prefix_hash") {
		return nil // unknown shape; leave it alone
	}
	stmts := []string{
		`ALTER TABLE discussions RENAME TO discussions_legacy`,
		`CREATE TABLE discussions (
		   id          INTEGER PRIMARY KEY AUTOINCREMENT,
		   anchor_key  TEXT NOT NULL,
		   system_json TEXT NOT NULL,
		   tools_json  TEXT NOT NULL,
		   created_at  INTEGER NOT NULL
		 )`,
		`INSERT INTO discussions(id, anchor_key, system_json, tools_json, created_at)
		   SELECT id, prefix_hash, system_json, tools_json, created_at FROM discussions_legacy`,
		`DROP TABLE discussions_legacy`,
	}
	for _, q := range stmts {
		if _, err := db.Exec(q); err != nil {
			return err
		}
	}
	return nil
}

func tableExists(db *sql.DB, name string) bool {
	var n string
	err := db.QueryRow(`SELECT name FROM sqlite_master WHERE type='table' AND name=?`, name).Scan(&n)
	return err == nil
}

func columnExists(db *sql.DB, table, col string) bool {
	rows, err := db.Query(`PRAGMA table_info(` + table + `)`)
	if err != nil {
		return false
	}
	defer rows.Close()
	for rows.Next() {
		var (
			cid     int
			name    string
			ctype   string
			notnull int
			dflt    sql.NullString
			pk      int
		)
		if rows.Scan(&cid, &name, &ctype, &notnull, &dflt, &pk) != nil {
			continue
		}
		if name == col {
			return true
		}
	}
	return false
}

func (s *Store) Close() error { return s.db.Close() }

func now() int64 { return time.Now().Unix() }

// ResolveDiscussion maps an incoming request to a stable discussion id by
// matching the conversation's assistant transcript (which the client echoes
// back verbatim each turn) against what we've recorded. The system prompt,
// tools and user messages are deliberately ignored for identity, because IDE /
// agent clients mutate them every turn (cwd, timestamps, <environment_details>),
// which is what previously fragmented one conversation into many discussions.
//
//   - continuation: the longest discussion whose recorded assistant sequence is
//     a prefix of the incoming one wins (same conversation, one more turn).
//   - first turn (no assistant messages yet): reuse an empty discussion sharing
//     the same normalized first-user anchor (handles retries), else create one.
//
// Each anchor is turnAnchor(text, tool_calls), so turns that are pure tool_calls
// (common with Cline/Roo/JetBrains agents) are still strong, distinct anchors.
func (s *Store) ResolveDiscussion(sig ConvSig) (int64, error) {
	if len(sig.Anchors) == 0 {
		return s.resolveFirstTurn(sig)
	}
	if id, ok := s.matchByTranscript(sig.Anchors); ok {
		return id, nil
	}
	// Assistant history present but unrecognised (edited history, or a chat that
	// predates the router): start a fresh discussion anchored on its first user.
	return s.createDiscussion(sig)
}

func (s *Store) resolveFirstTurn(sig ConvSig) (int64, error) {
	var id int64
	err := s.db.QueryRow(`
		SELECT d.id FROM discussions d
		WHERE d.anchor_key = ?
		  AND (SELECT COUNT(*) FROM turns t WHERE t.discussion_id = d.id) = 0
		ORDER BY d.id DESC LIMIT 1`, sig.AnchorKey).Scan(&id)
	if err == nil {
		return id, nil
	}
	if err != sql.ErrNoRows {
		return 0, err
	}
	return s.createDiscussion(sig)
}

func (s *Store) createDiscussion(sig ConvSig) (int64, error) {
	res, err := s.db.Exec(
		`INSERT INTO discussions(anchor_key, system_json, tools_json, created_at) VALUES(?,?,?,?)`,
		sig.AnchorKey, sig.SystemJSON, sig.ToolsJSON, now())
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

// matchByTranscript finds the discussion whose recorded assistant messages form
// the longest prefix of `anchors`. Returns (id, true) on a match of length >= 1.
func (s *Store) matchByTranscript(anchors []string) (int64, bool) {
	rows, err := s.db.Query(`SELECT id FROM discussions ORDER BY id`)
	if err != nil {
		return 0, false
	}
	var ids []int64
	for rows.Next() {
		var id int64
		if rows.Scan(&id) == nil {
			ids = append(ids, id)
		}
	}
	rows.Close()

	var bestID int64
	bestLen := 0
	for _, id := range ids {
		stored := s.anchorSequence(id)
		if len(stored) == 0 || len(stored) > len(anchors) {
			continue // empty, or longer than incoming => cannot be a prefix
		}
		n := commonPrefixLen(stored, anchors)
		if n == len(stored) && n > bestLen { // stored is fully a prefix of anchors
			bestLen = n
			bestID = id
		}
	}
	return bestID, bestLen >= 1
}

// anchorSequence returns this discussion's per-turn fingerprints, in turn order.
// They were computed at persist time by turnAnchor over the actual assistant
// output (text + tool_calls), so they compare directly to the incoming anchors.
func (s *Store) anchorSequence(discID int64) []string {
	rows, err := s.db.Query(
		`SELECT anchor_sig FROM turns WHERE discussion_id=? ORDER BY seq`, discID)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var a sql.NullString
		if rows.Scan(&a) == nil {
			out = append(out, a.String)
		}
	}
	return out
}

func commonPrefixLen(a, b []string) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

func (s *Store) nextSeq(discID int64) int {
	var n sql.NullInt64
	s.db.QueryRow(`SELECT MAX(seq) FROM turns WHERE discussion_id=?`, discID).Scan(&n)
	if n.Valid {
		return int(n.Int64) + 1
	}
	return 0
}

func (s *Store) AppendTurn(discID int64, role, userMsg, assistantMsg, anchorSig string) error {
	_, err := s.db.Exec(
		`INSERT INTO turns(discussion_id, seq, role, user_msg, assistant_msg, anchor_sig, created_at) VALUES(?,?,?,?,?,?,?)`,
		discID, s.nextSeq(discID), role, userMsg, assistantMsg, anchorSig, now())
	return err
}

func (s *Store) SavePlan(discID int64, planJSON string) error {
	_, err := s.db.Exec(
		`INSERT INTO plans(discussion_id, plan_json, created_at) VALUES(?,?,?)`,
		discID, planJSON, now())
	return err
}

// LatestUnconsumedPlan returns the most recent plan not yet dispatched to the coder.
func (s *Store) LatestUnconsumedPlan(discID int64) (string, bool) {
	var p string
	err := s.db.QueryRow(
		`SELECT plan_json FROM plans WHERE discussion_id=? AND consumed=0 ORDER BY id DESC LIMIT 1`,
		discID).Scan(&p)
	if err != nil {
		return "", false
	}
	return p, true
}

func (s *Store) MarkPlansConsumed(discID int64) {
	s.db.Exec(`UPDATE plans SET consumed=1 WHERE discussion_id=?`, discID)
}

func (s *Store) HasPendingPlan(discID int64) bool {
	_, ok := s.LatestUnconsumedPlan(discID)
	return ok
}

// --- per-role context (the splittable, model-specific text history) ----------

func (s *Store) LoadRoleMessages(discID int64, role string) ([]Message, string) {
	var raw, slotFile string
	err := s.db.QueryRow(
		`SELECT messages_json, slot_file FROM model_contexts WHERE discussion_id=? AND role=?`,
		discID, role).Scan(&raw, &slotFile)
	if err != nil {
		return nil, ""
	}
	var msgs []Message
	json.Unmarshal([]byte(raw), &msgs)
	return msgs, slotFile
}

func (s *Store) SaveRoleMessages(discID int64, role string, msgs []Message, slotFile string) error {
	b, _ := json.Marshal(msgs)
	_, err := s.db.Exec(`
		INSERT INTO model_contexts(discussion_id, role, messages_json, slot_file, last_used)
		VALUES(?,?,?,?,?)
		ON CONFLICT(discussion_id, role) DO UPDATE SET
		  messages_json=excluded.messages_json,
		  slot_file=CASE WHEN excluded.slot_file<>'' THEN excluded.slot_file ELSE model_contexts.slot_file END,
		  last_used=excluded.last_used`,
		discID, role, string(b), slotFile, now())
	return err
}

// --- coder summary handoff ---------------------------------------------------

func (s *Store) PushSummary(discID int64, summary string) error {
	_, err := s.db.Exec(
		`INSERT INTO pending_summaries(discussion_id, summary, created_at) VALUES(?,?,?)`,
		discID, summary, now())
	return err
}

// PopSummaries returns and consumes all pending coder summaries for a discussion.
func (s *Store) PopSummaries(discID int64) []string {
	rows, err := s.db.Query(
		`SELECT id, summary FROM pending_summaries WHERE discussion_id=? AND consumed=0 ORDER BY id`,
		discID)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var ids []int64
	var out []string
	for rows.Next() {
		var id int64
		var sum string
		if rows.Scan(&id, &sum) == nil {
			ids = append(ids, id)
			out = append(out, sum)
		}
	}
	for _, id := range ids {
		s.db.Exec(`UPDATE pending_summaries SET consumed=1 WHERE id=?`, id)
	}
	return out
}

func (s *Store) CountDiscussions() int {
	var n int
	s.db.QueryRow(`SELECT COUNT(*) FROM discussions`).Scan(&n)
	return n
}

type DiscInfo struct {
	ID        int64 `json:"id"`
	CreatedAt int64 `json:"created_at"`
	Turns     int   `json:"turns"`
	LastUsed  int64 `json:"last_used"`
}

func (s *Store) ListDiscussions(limit int) []DiscInfo {
	rows, err := s.db.Query(`
		SELECT d.id, d.created_at,
		  (SELECT COUNT(*) FROM turns t WHERE t.discussion_id=d.id),
		  COALESCE((SELECT MAX(last_used) FROM model_contexts m WHERE m.discussion_id=d.id), d.created_at)
		FROM discussions d
		ORDER BY 4 DESC
		LIMIT ?`, limit)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []DiscInfo
	for rows.Next() {
		var d DiscInfo
		if rows.Scan(&d.ID, &d.CreatedAt, &d.Turns, &d.LastUsed) == nil {
			out = append(out, d)
		}
	}
	return out
}
