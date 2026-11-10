package server

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"sort"
	"strings"
	"sync"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

// errNotReranker is returned for models that llama-server cannot score
// with rank pooling.
var errNotReranker = errors.New("does not support rerank: the GGUF has no classification head (cls.* tensors); use a reranker converted for llama.cpp")

// rerankCapable caches checkRerankModel results per GGUF path set. Blob paths
// are content-addressed, so a cached answer never goes stale.
var rerankCapable sync.Map // map[string]bool

// checkRerankModel reports whether m can produce meaningful rerank scores.
//
// llama-server's rank pooling projects the pooled hidden state through the
// model's classification head (cls.weight and/or cls.output.weight). A GGUF
// without one — e.g. a Qwen3-Reranker converted as a plain causal LM — is
// still accepted by llama-server with --reranking, but then returns raw
// hidden-state values as scores, so reject it up front.
func checkRerankModel(m *Model) error {
	if !m.isGGUF() || m.ModelPath == "" {
		return errNotReranker
	}

	key := strings.Join(m.modelPaths(), "\x00")
	if ok, found := rerankCapable.Load(key); found {
		if !ok.(bool) {
			return errNotReranker
		}
		return nil
	}

	f, err := gguf.ReadModel(m.ModelPath, 1024, m.ModelShardPaths...)
	if err != nil {
		return fmt.Errorf("read model metadata: %w", err)
	}
	ok := len(f.Tensors().Items("cls.")) > 0
	rerankCapable.Store(key, ok)
	if !ok {
		return errNotReranker
	}
	return nil
}

func (s *Server) RerankHandler(c *gin.Context) {
	var req api.RerankRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	switch {
	case len(req.Documents) == 0:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "Document cannot be empty"})
		return
	case req.Query == "":
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "Query cannot be empty"})
		return
	case req.TopN < 0:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "TopN cannot be negative"})
		return
	}
	ref, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		if errors.Is(err, errModelRequired) {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}
	if ref.Source == modelSourceCloud {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "rerank requires a local model"})
		return
	}
	mm, err := s.getExistingModelForRunner(ref.Name.String(), "")
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
			return
		}
		handleScheduleError(c, req.Model, err)
		return
	}
	if err := checkRerankModel(mm); err != nil {
		if errors.Is(err, errNotReranker) {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("model '%s' %s", req.Model, err)})
			return
		}
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if req.Options == nil {
		req.Options = make(map[string]any)
	}
	req.Options["reranking"] = true
	r, m, _, err := s.scheduleRunnerForModel(c.Request.Context(), mm, []model.Capability{}, req.Options, req.KeepAlive, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	llmReq := llm.RerankRequest{
		Model: req.Model,
	}
	useTemplate := false
	if m.Template != nil {
		vars, err := m.Template.Vars()
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		useTemplate = slices.Contains(vars, "query") && slices.Contains(vars, "document")
	}
	if useTemplate {
		for _, doc := range req.Documents {
			var b bytes.Buffer
			var values template.Values
			values.Query = req.Query
			values.Document = doc
			if err := m.Template.Execute(&b, values); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			llmReq.Prompts = append(llmReq.Prompts, b.String())
		}
	} else {
		llmReq.Query = &req.Query
		llmReq.Prompts = req.Documents
	}
	err = r.Rerank(c.Request.Context(), llmReq, func(rr llm.RerankResponse) {
		// Scores pass through untouched: llama-server's /v1/rerank guarantees
		// higher = more relevant for every model family (BGE-style logits,
		// Qwen3-style sigmoid probabilities, ...).
		sort.SliceStable(rr.Results, func(i, j int) bool {
			return rr.Results[i].RelevanceScore > rr.Results[j].RelevanceScore
		})

		var topn int
		if req.TopN == 0 {
			topn = len(rr.Results) // if TopN is unset, return all results
		} else {
			topn = min(len(rr.Results), req.TopN)
		}
		topResults := rr.Results[:topn]

		rsp := api.RerankResponse{
			Model:   req.Model,
			Results: make([]api.RerankResult, topn),
		}

		for i, result := range topResults {
			rsp.Results[i].Index = result.Index
			rsp.Results[i].RelevanceScore = result.RelevanceScore
		}

		c.JSON(http.StatusOK, rsp)
	})

	if err != nil {
		slog.Info(fmt.Sprintf("rerank failed: %v", err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to rerank"})
		return
	}
}
