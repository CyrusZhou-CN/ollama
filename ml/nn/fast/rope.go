package fast

import (
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/fast/rope"
)

func RoPE(ctx ml.Context, t, positionIDs ml.Tensor, dim uint32, base, scale float32, options ...func(*rope.Options)) ml.Tensor {
	if t, ok := t.(rope.RoPE); ok {
		return t.RoPE(ctx, positionIDs, dim, base, scale, options...)
	}

	panic("RoPE not implemented for this tensor type")
}
