import unittest

import torch
import comfy.model_patcher
import comfy.supported_models

import pipeline_h3 as pipeline


def tiny_h3(vsa):
    """Two-block MiniMax H3 DiT; `vsa` adds the to_gate_compress layers FastH3 checkpoints ship."""
    cfg = comfy.supported_models.MiniMaxH3({
        "image_model": "minimax_h3", "hidden_size": 256, "num_layers": 2, "token_refiner_num_layers": 1,
        "num_attention_heads": 2, "attention_head_dim": 128, "ffn_hidden_size": 256, "text_dim": 64,
        "timestep_input_dim": 32, "time_embed_hidden_size": 64, "time_embed_dim": 64,
        "rope_inv_freq_len": 16, "gate_compress": vsa,
    })
    cfg.set_inference_dtype(torch.float32, None)
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(cfg.get_model({}), cpu, cpu)


def block_patches(model):
    replace = model.model_options["transformer_options"].get("patches_replace", {})
    return sorted(replace.get("dit", {}))


def has_vsa(model):
    import comfy.patcher_extension

    return "block_sparse_attention" in model.callbacks.get(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, {})


ALL_BLOCKS = [("double_block", 0), ("double_block", 1)]


class AttentionPatchTests(unittest.TestCase):
    def test_base_model_gets_kitchen_attention_and_keeps_res_multistep(self):
        model, sampler = pipeline._patch_attention(tiny_h3(vsa=False), [])
        self.assertEqual(sampler, "res_multistep")
        self.assertIn("optimized_attention_override", model.model_options["transformer_options"])
        self.assertFalse(has_vsa(model))
        self.assertEqual(block_patches(model), ALL_BLOCKS)  # long-sequence chunking wrappers only

    def test_fasth3_gets_vsa_on_every_block_and_euler(self):
        model, sampler = pipeline._patch_attention(tiny_h3(vsa=True), [])
        self.assertEqual(sampler, "euler")
        self.assertTrue(has_vsa(model))
        self.assertEqual(block_patches(model), ALL_BLOCKS)

    def test_fasth3_i2v_uses_the_v2_recipe(self):
        model, sampler = pipeline._patch_attention(tiny_h3(vsa=True), [], "i2v")
        self.assertEqual(sampler, "res_multistep")
        self.assertEqual(model.model_options["transformer_options"]["minimax_h3_sigma_shift_video"], 10.0)
        self.assertTrue(has_vsa(model))

    def test_fasth3_stays_dense_under_controlnet(self):
        positive = [[torch.zeros(1), {"control": object()}]]
        model, sampler = pipeline._patch_attention(tiny_h3(vsa=True), positive)
        self.assertEqual(sampler, "euler")
        self.assertEqual(block_patches(model), [])


class ChunkedBlockTests(unittest.TestCase):
    def test_chunked_block_matches_comfy_block(self):
        from unittest import mock

        from comfy.ldm.minimax.model import rope_rotation_table

        torch.manual_seed(0)
        block = tiny_h3(vsa=False).get_model_object("diffusion_model").blocks[0]
        for name, param in block.named_parameters():  # comfy ops leave weights uninitialised
            param.data.copy_(torch.ones_like(param) if "norm" in name else torch.randn_like(param) * 0.05)
        n = 300
        x = torch.randn(n, 256)
        t_emb = torch.randn(2, 64)
        segments = [(0, 120, 0), (120, 300, torch.arange(180) % 2 * 3)]  # a plain row and per-token rows
        rope = rope_rotation_table(torch.randn(n, 32), torch.float32)
        with torch.no_grad():
            expected = block(x.clone(), t_emb, segments, rope)
            with mock.patch.object(pipeline, "CHUNK_TOKENS", 64):
                got = pipeline._chunked_block(block, x.clone(), t_emb, segments, rope, {})
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
