package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

// HFModel is a trimmed-down HuggingFace model record for the picker UI.
type HFModel struct {
	ID        string `json:"id"`
	Downloads int    `json:"downloads"`
	Likes     int    `json:"likes"`
}

func hfGet(u string, v any) error {
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", u, nil)
	req.Header.Set("Accept", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("huggingface: %s", resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(v)
}

// HFSearch returns model repositories matching the query, most-downloaded
// first. When ggufOnly is set it restricts to GGUF repos (for llama.cpp);
// otherwise it returns general text-generation repos (for vLLM/TGI/SGLang).
func HFSearch(query string, ggufOnly bool) ([]HFModel, error) {
	q := url.Values{}
	q.Set("search", query)
	if ggufOnly {
		q.Set("filter", "gguf")
	} else {
		q.Set("pipeline_tag", "text-generation")
	}
	q.Set("sort", "downloads")
	q.Set("direction", "-1")
	q.Set("limit", "24")
	out := []HFModel{}
	if err := hfGet("https://huggingface.co/api/models?"+q.Encode(), &out); err != nil {
		return nil, err
	}
	return out, nil
}

type hfModelInfo struct {
	Siblings []struct {
		Rfilename string `json:"rfilename"`
	} `json:"siblings"`
}

// HFFiles returns the .gguf file names available in a repository.
func HFFiles(repo string) ([]string, error) {
	var info hfModelInfo
	if err := hfGet("https://huggingface.co/api/models/"+repo, &info); err != nil {
		return nil, err
	}
	files := []string{}
	for _, s := range info.Siblings {
		if strings.HasSuffix(strings.ToLower(s.Rfilename), ".gguf") {
			files = append(files, s.Rfilename)
		}
	}
	sort.Strings(files)
	return files, nil
}
