package main

import "sync"

// slotManager shares one upstream's N parallel slots (-np) across active
// discussions. Each discussion that touches this role gets a slot; when the
// pool is full, the least-recently-used discussion is evicted so a new one can
// take its slot. The evicted discussion's KV is optionally parked to a .bin and
// otherwise simply rebuilt from SQLite text on its next turn.
type slotManager struct {
	mu     sync.Mutex
	size   int
	bySlot []int64 // slotID -> discussionID (0 = free)
	order  []int64 // discussionIDs, least-recent first ... most-recent last
}

func newSlotManager(size int) *slotManager {
	if size < 1 {
		size = 1
	}
	return &slotManager{size: size, bySlot: make([]int64, size)}
}

type slotDecision struct {
	slot    int   // the slot to use for this turn
	warm    bool  // discussion was already loaded in this slot (KV hot)
	evicted int64 // discussion pushed out to make room (0 = none)
}

func (m *slotManager) acquire(discID int64) slotDecision {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Already resident: reuse its warm slot.
	for s, d := range m.bySlot {
		if d == discID {
			m.touch(discID)
			return slotDecision{slot: s, warm: true}
		}
	}
	// Free slot available.
	for s, d := range m.bySlot {
		if d == 0 {
			m.bySlot[s] = discID
			m.touch(discID)
			return slotDecision{slot: s, warm: false}
		}
	}
	// Pool full: evict the least-recently-used discussion.
	victim := m.order[0]
	slot := m.slotOf(victim)
	m.removeOrder(victim)
	m.bySlot[slot] = discID
	m.touch(discID)
	return slotDecision{slot: slot, warm: false, evicted: victim}
}

func (m *slotManager) slotOf(discID int64) int {
	for s, d := range m.bySlot {
		if d == discID {
			return s
		}
	}
	return -1
}

func (m *slotManager) touch(discID int64) {
	m.removeOrder(discID)
	m.order = append(m.order, discID)
}

func (m *slotManager) removeOrder(discID int64) {
	for i, d := range m.order {
		if d == discID {
			m.order = append(m.order[:i], m.order[i+1:]...)
			return
		}
	}
}

func (m *slotManager) stats() (used, size int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, d := range m.bySlot {
		if d != 0 {
			used++
		}
	}
	return used, m.size
}
