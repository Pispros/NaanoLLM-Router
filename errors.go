package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
)

// CodedError carries a stable machine code (and optional role) alongside an
// English fallback message, so the control panel can localize it.
type CodedError struct {
	Code string
	Role string
	Msg  string
}

func (e *CodedError) Error() string { return e.Msg }

func codedErr(code, role, format string, a ...any) *CodedError {
	return &CodedError{Code: code, Role: role, Msg: fmt.Sprintf(format, a...)}
}

// writeErr emits {"error","code","role"} as JSON; code/role are populated when
// err is a *CodedError, so the frontend can pick a translated message.
func writeErr(w http.ResponseWriter, status int, err error) {
	code, role := "", ""
	var ce *CodedError
	if errors.As(err, &ce) {
		code, role = ce.Code, ce.Role
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error(), "code": code, "role": role})
}
