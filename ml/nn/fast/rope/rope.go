package rope

import "github.com/ollama/ollama/ml"

type RoPE interface {
	RoPE(ctx ml.Context, positionIDs ml.Tensor, dim uint32, base, scale float32, options ...func(*Options)) ml.Tensor
}

// Options contains optional parameters for RoPE function
type Options struct {
	OriginalContextLength uint32
	Type                  uint32
	Factors               ml.Tensor
}

// WithOriginalContextLength sets a custom context length
func WithOriginalContextLength(len uint32) func(*Options) {
	return func(opts *Options) {
		opts.OriginalContextLength = len
	}
}

// WithType sets RoPE type to NeoX
func WithTypeNeoX() func(*Options) {
	return func(opts *Options) {
		opts.Type = 2
	}
}

// WithFactors sets custom rope factors
func WithFactors(factors ml.Tensor) func(*Options) {
	return func(opts *Options) {
		if factors != nil {
			opts.Factors = factors
		}
	}
}
