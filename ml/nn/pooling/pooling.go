package pooling

import (
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type Type uint32

const (
	TypeNone Type = iota
	TypeMean
	TypeCLS
	TypeLast
	TypeRank
)

func (t Type) String() string {
	switch t {
	case TypeMean:
		return "Mean"
	case TypeCLS:
		return "CLS"
	case TypeLast:
		return "Last"
	case TypeRank:
		return "Rank"
	default:
		return "Unknown"
	}
}

// Forward applies the configured pooling to hiddenStates.
//
// sequences is the per-token sequence ID slice from input.Batch.Sequences.
// It is required for sequence-aware pooling when multiple sequences are
// batched together (OLLAMA_NUM_PARALLEL > 1). When nil/empty, each pooling
// type falls back to the previous global-batch behavior — correct only for
// single-sequence batches.
func (t Type) Forward(ctx ml.Context,
	hiddenStates ml.Tensor,
	sequences []int,
	cls *nn.Linear,
	clsOut *nn.Linear,
) ml.Tensor {
	switch t {
	case TypeMean:
		return meanPerSequence(ctx, hiddenStates, sequences)
	case TypeCLS:
		return firstPositionPerSequence(ctx, hiddenStates, sequences)
	case TypeLast:
		return lastPositionPerSequence(ctx, hiddenStates, sequences)
	case TypeRank:
		hiddenStates = lastPositionPerSequence(ctx, hiddenStates, sequences)
		if cls != nil && cls.Weight != nil {
			hiddenStates = cls.Forward(ctx, hiddenStates)
			hiddenStates = hiddenStates.Tanh(ctx)
		}
		if clsOut != nil && clsOut.Weight != nil {
			hiddenStates = clsOut.Forward(ctx, hiddenStates)
		}
		return hiddenStates
	default:
		panic("unknown pooling type")
	}
}

// lastPositionPerSequence gathers the last token position of each unique
// sequence in `sequences` (in order of first appearance) from hiddenStates.
// If sequences is nil/empty, falls back to the global last position so
// single-sequence callers see the previous behavior.
func lastPositionPerSequence(ctx ml.Context, hiddenStates ml.Tensor, sequences []int) ml.Tensor {
	if len(sequences) == 0 {
		return hiddenStates.Slice(ctx, 1, hiddenStates.Dim(1)-1, hiddenStates.Dim(1), 1)
	}

	seen := make(map[int]struct{}, 8)
	order := make([]int, 0, 8)
	lastPos := make(map[int]int32, 8)
	for i, sid := range sequences {
		if _, ok := seen[sid]; !ok {
			seen[sid] = struct{}{}
			order = append(order, sid)
		}
		lastPos[sid] = int32(i)
	}

	indices := make([]int32, len(order))
	for i, sid := range order {
		indices[i] = lastPos[sid]
	}
	idx := ctx.Input().FromInts(indices, len(indices))
	return hiddenStates.Rows(ctx, idx)
}

// firstPositionPerSequence gathers the first token position of each unique
// sequence in `sequences` (in order of first appearance) from hiddenStates.
// If sequences is nil/empty, falls back to position 0 of the whole batch
// (the previous TypeCLS behavior).
func firstPositionPerSequence(ctx ml.Context, hiddenStates ml.Tensor, sequences []int) ml.Tensor {
	if len(sequences) == 0 {
		return hiddenStates.Slice(ctx, 1, 0, 1, 1)
	}

	seen := make(map[int]struct{}, 8)
	indices := make([]int32, 0, 8)
	for i, sid := range sequences {
		if _, ok := seen[sid]; !ok {
			seen[sid] = struct{}{}
			indices = append(indices, int32(i))
		}
	}
	idx := ctx.Input().FromInts(indices, len(indices))
	return hiddenStates.Rows(ctx, idx)
}

// meanPerSequence computes the per-sequence mean of token hidden states.
// For each unique seq (in order of first appearance), gathers its tokens
// with Rows and reduces via Permute→Mean→Permute. Concatenates the per-seq
// means along dim 1 to produce shape [embed_dim, n_unique_seqs].
//
// If sequences is nil/empty, falls back to a global mean over the whole
// batch (the previous TypeMean behavior).
//
// Performance note: this builds a graph with O(n_unique_seqs) ops, which is
// fine for typical embedding workloads (1-8 batched sequences). If the user
// ever batches very many sequences and segmented mean becomes a bottleneck,
// it can be replaced with a single mask-matmul without changing this
// function's signature.
func meanPerSequence(ctx ml.Context, hiddenStates ml.Tensor, sequences []int) ml.Tensor {
	if len(sequences) == 0 {
		hiddenStates = hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).Mean(ctx)
		return hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	}

	seen := make(map[int]struct{}, 8)
	order := make([]int, 0, 8)
	groups := make(map[int][]int32, 8)
	for i, sid := range sequences {
		if _, ok := seen[sid]; !ok {
			seen[sid] = struct{}{}
			order = append(order, sid)
		}
		groups[sid] = append(groups[sid], int32(i))
	}

	var result ml.Tensor
	for _, sid := range order {
		idx := ctx.Input().FromInts(groups[sid], len(groups[sid]))
		seqTokens := hiddenStates.Rows(ctx, idx)
		seqMean := seqTokens.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).Mean(ctx)
		seqMean = seqMean.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
		if result == nil {
			result = seqMean
		} else {
			result = result.Concat(ctx, seqMean, 1)
		}
	}
	return result
}
