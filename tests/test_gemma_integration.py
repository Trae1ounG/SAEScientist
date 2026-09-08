"""CPU integration with a tiny random Gemma model; no pretrained weights or downloads."""
import pytest
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

from sae_scientist.steering import steer


def test_gemma_residual_intervention_and_hook_cleanup():
    torch.manual_seed(0)
    config = Gemma2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=32)
    model = Gemma2ForCausalLM(config).eval()
    layer = model.model.layers[0]
    inputs = torch.tensor([[2, 3, 4]])
    observed = []

    def capture(_module, _inputs, output):
        value = output if torch.is_tensor(output) else output[0]
        observed.append(value.clone())

    with torch.inference_mode():
        handle = layer.register_forward_hook(capture)
        model(inputs, use_cache=False)
        handle.remove()
        with steer(layer, torch.ones(32), 0.5):
            handle = layer.register_forward_hook(capture)
            model(inputs, use_cache=False)
            handle.remove()
    torch.testing.assert_close(observed[1] - observed[0], torch.full_like(observed[0], 0.5))
    assert not layer._forward_hooks
    with pytest.raises(RuntimeError, match="synthetic"):
        with steer(layer, torch.ones(32), 0.5):
            raise RuntimeError("synthetic")
    assert not layer._forward_hooks
