package pooling_test

import (
	"bytes"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/ml/nn/pooling"
)

func setup(tb testing.TB, n int) ml.Backend {
	tb.Helper()

	f, err := os.CreateTemp(tb.TempDir(), "*.bin")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	if err := fsggml.WriteGGUF(f, fsggml.KV{
		"general.architecture": "test",
		"test.block_count":     uint32(1),
	}, []*fsggml.Tensor{
		{Name: "blk.0.weight", Shape: []uint64{1}, WriterTo: bytes.NewBuffer(make([]byte, 4))},
	}); err != nil {
		tb.Fatal(err)
	}

	b, err := ggml.New(f.Name(), ml.BackendParams{AllocMemory: true})
	if err != nil {
		tb.Fatal(err)
	}

	return b
}

func TestForward(t *testing.T) {
	cases := map[pooling.Type][]float32{
		pooling.TypeMean: {4, 5, 6, 7, 8, 9, 10, 11},
		pooling.TypeCLS:  {0, 1, 2, 3, 4, 5, 6, 7},
		pooling.TypeLast: {8, 9, 10, 11, 12, 13, 14, 15},
	}
	for typ, want := range cases {
		t.Run(typ.String(), func(t *testing.T) {
			b := setup(t, 99)
			defer b.Close()

			ctx := b.NewContext()
			defer ctx.Close()

			tt := ctx.Input().Arange(0, 16, 1, ml.DTypeF32).Reshape(ctx, 8, 2)
			tt = typ.Forward(ctx, tt, nil, nil, nil)

			ctx.Forward(tt).Compute(tt)
			if diff := cmp.Diff(want, tt.Floats()); diff != "" {
				t.Error(diff)
			}
		})
	}
}

// TestRankSingleSequenceFallback verifies backward compatibility: when
// sequences is nil/empty, TypeRank falls back to the global last position
// (matches the previous behavior — same as TypeLast for shape [8, 2]).
func TestRankSingleSequenceFallback(t *testing.T) {
	want := []float32{8, 9, 10, 11, 12, 13, 14, 15}

	b := setup(t, 99)
	defer b.Close()

	ctx := b.NewContext()
	defer ctx.Close()

	tt := ctx.Input().Arange(0, 16, 1, ml.DTypeF32).Reshape(ctx, 8, 2)
	tt = pooling.TypeRank.Forward(ctx, tt, nil, nil, nil)

	ctx.Forward(tt).Compute(tt)
	if diff := cmp.Diff(want, tt.Floats()); diff != "" {
		t.Errorf("nil sequences fallback: %s", diff)
	}
}

// TestRankSingleSequenceExplicit verifies that passing a sequences slice
// containing one unique seq id produces the same result as the nil fallback
// (i.e. the last position of that seq).
func TestRankSingleSequenceExplicit(t *testing.T) {
	want := []float32{8, 9, 10, 11, 12, 13, 14, 15}

	b := setup(t, 99)
	defer b.Close()

	ctx := b.NewContext()
	defer ctx.Close()

	tt := ctx.Input().Arange(0, 16, 1, ml.DTypeF32).Reshape(ctx, 8, 2)
	tt = pooling.TypeRank.Forward(ctx, tt, []int{0, 0}, nil, nil)

	ctx.Forward(tt).Compute(tt)
	if diff := cmp.Diff(want, tt.Floats()); diff != "" {
		t.Errorf("single-seq explicit: %s", diff)
	}
}

// TestRankMultiSequence is the regression test for the batched-rerank bug.
// Two sequences in one batch, 2 tokens each. The pooling layer must gather
// the last token of each unique sequence (positions 1 and 3) and return
// both rows. Before the fix, only the global last position (3) was returned.
func TestRankMultiSequence(t *testing.T) {
	// 4 token positions, embed_dim=8 → tensor shape [8, 4].
	// sequences = [0, 0, 1, 1] → unique seqs [0, 1], last positions [1, 3].
	// Expected output rows: position 1 (values 8..15) and position 3 (24..31).
	want := []float32{
		8, 9, 10, 11, 12, 13, 14, 15,
		24, 25, 26, 27, 28, 29, 30, 31,
	}

	b := setup(t, 99)
	defer b.Close()

	ctx := b.NewContext()
	defer ctx.Close()

	tt := ctx.Input().Arange(0, 32, 1, ml.DTypeF32).Reshape(ctx, 8, 4)
	tt = pooling.TypeRank.Forward(ctx, tt, []int{0, 0, 1, 1}, nil, nil)

	ctx.Forward(tt).Compute(tt)
	if diff := cmp.Diff(want, tt.Floats()); diff != "" {
		t.Errorf("multi-seq: %s", diff)
	}
}

// TestRankMultiSequenceFirstAppearanceOrder verifies that the output order
// follows the order each unique seq id FIRST appears in `sequences`, not
// numerical order. This matters because llama.cpp's pooled output uses
// first-appearance ordering, and the runner's pool index assignment in
// forwardBatch matches that — so the test pins the expected order.
func TestRankMultiSequenceFirstAppearanceOrder(t *testing.T) {
	// 4 tokens. sequences = [7, 3, 7, 3] → unique seqs in order: [7, 3].
	// Last position of seq 7: index 2 (values 16..23).
	// Last position of seq 3: index 3 (values 24..31).
	// Output rows in first-appearance order: seq 7 row, then seq 3 row.
	want := []float32{
		16, 17, 18, 19, 20, 21, 22, 23,
		24, 25, 26, 27, 28, 29, 30, 31,
	}

	b := setup(t, 99)
	defer b.Close()

	ctx := b.NewContext()
	defer ctx.Close()

	tt := ctx.Input().Arange(0, 32, 1, ml.DTypeF32).Reshape(ctx, 8, 4)
	tt = pooling.TypeRank.Forward(ctx, tt, []int{7, 3, 7, 3}, nil, nil)

	ctx.Forward(tt).Compute(tt)
	if diff := cmp.Diff(want, tt.Floats()); diff != "" {
		t.Errorf("first-appearance order: %s", diff)
	}
}

// TestLastMultiSequence: same as TestRankMultiSequence but for TypeLast (no
// cls/clsOut head). Verifies sequence-aware Last pooling for batched embeds.
func TestLastMultiSequence(t *testing.T) {
	// 4 tokens, embed_dim=8. sequences=[0,0,1,1] → last positions [1, 3].
	want := []float32{
		8, 9, 10, 11, 12, 13, 14, 15,
		24, 25, 26, 27, 28, 29, 30, 31,
	}

	b := setup(t, 99)
	defer b.Close()

	ctx := b.NewContext()
	defer ctx.Close()

	tt := ctx.Input().Arange(0, 32, 1, ml.DTypeF32).Reshape(ctx, 8, 4)
	tt = pooling.TypeLast.Forward(ctx, tt, []int{0, 0, 1, 1}, nil, nil)

	ctx.Forward(tt).Compute(tt)
	if diff := cmp.Diff(want, tt.Floats()); diff != "" {
		t.Errorf("Last multi-seq: %s", diff)
	}
}

// TestCLSMultiSequence verifies that TypeCLS picks the FIRST position of each
// unique sequence (position 0 of seq 0, position 2 of seq 1), not position 0
// of the whole batch.
func TestCLSMultiSequence(t *testing.T) {
	// 4 tokens, embed_dim=8. sequences=[0,0,1,1] → first positions [0, 2].
	want := []float32{
		0, 1, 2, 3, 4, 5, 6, 7,
		16, 17, 18, 19, 20, 21, 22, 23,
	}

	b := setup(t, 99)
	defer b.Close()

	ctx := b.NewContext()
	defer ctx.Close()

	tt := ctx.Input().Arange(0, 32, 1, ml.DTypeF32).Reshape(ctx, 8, 4)
	tt = pooling.TypeCLS.Forward(ctx, tt, []int{0, 0, 1, 1}, nil, nil)

	ctx.Forward(tt).Compute(tt)
	if diff := cmp.Diff(want, tt.Floats()); diff != "" {
		t.Errorf("CLS multi-seq: %s", diff)
	}
}

// TestMeanMultiSequence verifies that TypeMean averages tokens PER SEQUENCE,
// not over the entire batch.
func TestMeanMultiSequence(t *testing.T) {
	// 4 tokens, embed_dim=8. sequences=[0,0,1,1].
	// Mean of seq 0 (positions 0, 1) = mean of {[0..7], [8..15]} = [4..11].
	// Mean of seq 1 (positions 2, 3) = mean of {[16..23], [24..31]} = [20..27].
	want := []float32{
		4, 5, 6, 7, 8, 9, 10, 11,
		20, 21, 22, 23, 24, 25, 26, 27,
	}

	b := setup(t, 99)
	defer b.Close()

	ctx := b.NewContext()
	defer ctx.Close()

	tt := ctx.Input().Arange(0, 32, 1, ml.DTypeF32).Reshape(ctx, 8, 4)
	tt = pooling.TypeMean.Forward(ctx, tt, []int{0, 0, 1, 1}, nil, nil)

	ctx.Forward(tt).Compute(tt)
	if diff := cmp.Diff(want, tt.Floats()); diff != "" {
		t.Errorf("Mean multi-seq: %s", diff)
	}
}

// TestMeanMultiSequenceUnequalLengths verifies segmented mean works when
// sequences have different lengths.
func TestMeanMultiSequenceUnequalLengths(t *testing.T) {
	// 4 tokens, embed_dim=8. sequences=[0,1,1,1].
	// Mean of seq 0 (1 token at position 0) = [0..7].
	// Mean of seq 1 (3 tokens at positions 1, 2, 3) =
	//   ([8..15] + [16..23] + [24..31]) / 3 = [16..23].
	want := []float32{
		0, 1, 2, 3, 4, 5, 6, 7,
		16, 17, 18, 19, 20, 21, 22, 23,
	}

	b := setup(t, 99)
	defer b.Close()

	ctx := b.NewContext()
	defer ctx.Close()

	tt := ctx.Input().Arange(0, 32, 1, ml.DTypeF32).Reshape(ctx, 8, 4)
	tt = pooling.TypeMean.Forward(ctx, tt, []int{0, 1, 1, 1}, nil, nil)

	ctx.Forward(tt).Compute(tt)
	if diff := cmp.Diff(want, tt.Floats()); diff != "" {
		t.Errorf("Mean multi-seq unequal: %s", diff)
	}
}
