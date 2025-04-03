package convert

import (
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type llama4Model struct {
	ModelParameters
	TextModel struct {
		llamaModel
		NumExpertsPerToken     uint32 `json:"num_experts_per_tok"`
		NumLocalExperts        uint32 `json:"num_local_experts"`
		InterleaveMOELayerStep uint32 `json:"interleave_moe_layer_step"`
		UseQKNorm              bool   `json:"use_qk_norm"`
		IntermediateSizeMLP    uint32 `json:"intermediate_size_mlp"`
	} `json:"text_config"`
	VisionModel struct {
		NumHiddenLayers   uint32  `json:"num_hidden_layers"`
		IntermediateSize  uint32  `json:"intermediate_size"`
		NumAttentionHeads uint32  `json:"num_attention_heads"`
		ImageSize         uint32  `json:"image_size"`
		PatchSize         uint32  `json:"patch_size"`
		RopeTheta         float32 `json:"rope_theta"`
		NormEpsilon       float32 `json:"norm_eps"`
		PixelShuffleRatio float32 `json:"pixel_shuffle_ratio"`
	} `json:"vision_config"`
}

// KV implements ModelConverter.
func (p *llama4Model) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "llama4"

	for k, v := range p.TextModel.KV(t) {
		if strings.HasPrefix(k, "llama.") {
			kv[strings.ReplaceAll(k, "llama.", "llama4.")] = v
		}
	}

	kv["llama4.intermediate_size"] = p.TextModel.IntermediateSizeMLP
	kv["llama4.intermediate_size_moe"] = p.TextModel.IntermediateSize

	kv["llama4.expert_count"] = p.TextModel.NumLocalExperts
	kv["llama4.expert_used_count"] = p.TextModel.NumExpertsPerToken
	kv["llama4.interleave_moe_layer_step"] = p.TextModel.InterleaveMOELayerStep
	kv["llama4.use_qk_norm"] = p.TextModel.UseQKNorm

	kv["llama4.vision.block_count"] = p.VisionModel.NumHiddenLayers
	kv["llama4.vision.embedding_length"] = p.VisionModel.IntermediateSize
	kv["llama4.vision.attention.head_count"] = p.VisionModel.NumAttentionHeads
	kv["llama4.vision.image_size"] = p.VisionModel.ImageSize
	kv["llama4.vision.patch_size"] = p.VisionModel.PatchSize
	kv["llama4.vision.rope.freq_base"] = p.VisionModel.RopeTheta
	kv["llama4.vision.layer_norm_epsilon"] = p.VisionModel.NormEpsilon
	kv["llama4.vision.pixel_shuffle_ratio"] = p.VisionModel.PixelShuffleRatio
	return kv
}

// Replacements implements ModelConverter.
func (p *llama4Model) Replacements() []string {
	return append(
		p.TextModel.Replacements(),
		"language_model.", "",
		"vision_model", "v",
		"multi_modal_projector", "mm",
		"feed_forward.", "ffn_",
		"shared_expert.down_proj", "down_shexp",
		"shared_expert.gate_proj", "gate_shexp",
		"shared_expert.up_proj", "up_shexp",
		"experts.down_proj", "down_exps.weight",
		"experts.gate_up_proj", "gate_up_exps.weight",
		"router", "gate_inp",
	)
}

// Tensors implements ModelConverter.
func (p *llama4Model) Tensors(ts []Tensor) []ggml.Tensor {
	var out []ggml.Tensor

	var textTensors []Tensor
	for _, t := range ts {
		if strings.HasPrefix(t.Name(), "v.") || strings.HasPrefix(t.Name(), "mm.") {
			out = append(out, ggml.Tensor{
				Name:     t.Name(),
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
		} else if strings.Contains(t.Name(), "ffn_gate_up_exps") {
			// gate and up projectors are fused
			// dims[1], dims[2] must be swapped
			// [experts, hidden_size, intermediate_size * 2] --> [experts, intermediate_size, hidden_size]
			halfDim := int(t.Shape()[2])

			// clone tensor since we need separate repackers
			gate := t.Clone()
			gate.SetRepacker(p.repack(nil, nil, tensor.S(0, halfDim/2)))

			up := t.Clone()
			up.SetRepacker(p.repack(nil, nil, tensor.S(halfDim/2, halfDim)))

			newShape := append([]uint64{}, t.Shape()...)
			newShape[1], newShape[2] = newShape[2]/2, newShape[1]

			out = append(out, ggml.Tensor{
				Name:     strings.ReplaceAll(gate.Name(), "ffn_gate_up_exps", "ffn_gate_exps"),
				Kind:     gate.Kind(),
				Shape:    newShape,
				WriterTo: gate,
			}, ggml.Tensor{
				Name:     strings.ReplaceAll(up.Name(), "ffn_gate_up_exps", "ffn_up_exps"),
				Kind:     up.Kind(),
				Shape:    newShape,
				WriterTo: up,
			})
		} else if strings.Contains(t.Name(), "ffn_down_exps") {
			// dims[1], dims[2] must be swapped
			// [experts, intermediate_size, hidden_size] --> [experts, hidden_size, intermediate_size]
			t.SetRepacker(p.repack())
			shape := t.Shape()
			shape[1], shape[2] = shape[2], shape[1]
			out = append(out, ggml.Tensor{
				Name:     t.Name(),
				Kind:     t.Kind(),
				Shape:    shape,
				WriterTo: t,
			})
		} else {
			textTensors = append(textTensors, t)
		}
	}

	p.TextModel.skipRepack = true
	out = append(out, p.TextModel.Tensors(textTensors)...)
	return out
}

func (p *llama4Model) repack(slice ...tensor.Slice) Repacker {
	return func(name string, data []float32, shape []uint64) ([]float32, error) {
		dims := make([]int, len(shape))
		for i, dim := range shape {
			dims[i] = int(dim)
		}

		var t tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
		t, err := t.Slice(slice...)
		if err != nil {
			return nil, err
		}

		if err := t.T(0, 2, 1); err != nil {
			return nil, err
		}

		t = tensor.Materialize(t)
		if err := t.Reshape(t.Shape().TotalSize()); err != nil {
			return nil, err
		}

		return native.VectorF32(t.(*tensor.Dense))
	}
}
