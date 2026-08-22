package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"

	"github.com/ollama/ollama/api"
)

type RerankRequest struct {
	Model string `json:"model"`
	// Query is the rerank query. It is nil when the model's Ollama template
	// already rendered the query into each prompt (template-based rerankers).
	Query *string `json:"query"`
	// Prompts holds one entry per document: either the raw document text
	// (when Query is set) or the fully templated query+document prompt.
	Prompts []string `json:"prompts"`
}

type RerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float32 `json:"relevance_score"`
}

type RerankResponse struct {
	Results []RerankResult `json:"results"`
}

// Rerank scores each prompt against the query using llama-server's native
// /v1/rerank endpoint. The server must have been launched with --reranking,
// which requires a model converted with rank pooling; llama-server owns the
// model-specific query/document prompt formatting and pooling.
func (s *llamaServerRunner) Rerank(ctx context.Context, req RerankRequest, fn func(RerankResponse)) error {
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return err
	}
	defer s.sem.Release(1)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return err
	} else if status != ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status)
	}

	// Pre-templated prompts are passed through as documents with an empty
	// query; llama-server still wraps them in the model's rerank format, so
	// prefer raw query+documents whenever the caller has them.
	var query string
	if req.Query != nil {
		query = *req.Query
	}
	data, err := json.Marshal(map[string]any{
		"model":     req.Model,
		"query":     query,
		"documents": req.Prompts,
	})
	if err != nil {
		return fmt.Errorf("error marshaling rerank data: %w", err)
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/rerank", s.port), bytes.NewBuffer(data))
	if err != nil {
		return fmt.Errorf("error creating rerank request: %w", err)
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(r)
	if err != nil {
		return fmt.Errorf("do rerank request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("error reading rerank response: %w", err)
	}

	if resp.StatusCode >= 400 {
		slog.Error("llm rerank error", "status", resp.StatusCode, "response", string(body))
		return api.StatusError{StatusCode: resp.StatusCode, ErrorMessage: string(body)}
	}

	var rr RerankResponse
	if err := json.Unmarshal(body, &rr); err != nil {
		return fmt.Errorf("unmarshal rerank response: %w", err)
	}
	fn(rr)

	return nil
}
