package server

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/fs/gguf"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/types/model"
)

func writeRerankTestGGUF(t *testing.T, tensorNames ...string) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "model.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	var tensors []*gguftest.Tensor
	for _, name := range tensorNames {
		tensors = append(tensors, &gguftest.Tensor{
			Name:     name,
			Type:     gguf.TensorTypeF32,
			Shape:    []uint64{1, 1},
			WriterTo: bytes.NewReader(make([]byte, 4)),
		})
	}
	if err := gguftest.Write(f, gguftest.KV{"general.architecture": "test"}, tensors); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestCheckRerankModel(t *testing.T) {
	cases := []struct {
		name    string
		tensors []string
		wantErr error
	}{
		// BGE / XLM-RoBERTa cross-encoders: dense head + output projection.
		{"cls and cls.output", []string{"token_embd.weight", "cls.weight", "cls.bias", "cls.output.weight", "cls.output.bias"}, nil},
		// Qwen3-Reranker converted by llama.cpp: output projection only.
		{"cls.output only", []string{"token_embd.weight", "cls.output.weight"}, nil},
		// jina-reranker-v1: single-layer head, no cls.output.
		{"cls only", []string{"token_embd.weight", "cls.weight"}, nil},
		// Causal LM conversion: llama-server would return hidden-state garbage.
		{"no classification head", []string{"token_embd.weight", "output.weight", "output_norm.weight"}, errNotReranker},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{ModelPath: writeRerankTestGGUF(t, tt.tensors...)}
			if err := checkRerankModel(m); !errors.Is(err, tt.wantErr) {
				t.Fatalf("checkRerankModel() = %v, want %v", err, tt.wantErr)
			}
			// Second call is served from the cache and must agree.
			if err := checkRerankModel(m); !errors.Is(err, tt.wantErr) {
				t.Fatalf("cached checkRerankModel() = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func TestCheckRerankModelRejectsNonGGUF(t *testing.T) {
	m := &Model{
		ModelPath: writeRerankTestGGUF(t, "cls.output.weight"),
		Config:    model.ConfigV2{ModelFormat: "safetensors"},
	}
	if err := checkRerankModel(m); !errors.Is(err, errNotReranker) {
		t.Fatalf("checkRerankModel() = %v, want %v", err, errNotReranker)
	}
	if err := checkRerankModel(&Model{}); !errors.Is(err, errNotReranker) {
		t.Fatalf("checkRerankModel(no path) = %v, want %v", err, errNotReranker)
	}
}

func TestCheckRerankModelUnreadableFile(t *testing.T) {
	m := &Model{ModelPath: filepath.Join(t.TempDir(), "missing.gguf")}
	err := checkRerankModel(m)
	if err == nil || errors.Is(err, errNotReranker) {
		t.Fatalf("checkRerankModel() = %v, want a read error", err)
	}
}
