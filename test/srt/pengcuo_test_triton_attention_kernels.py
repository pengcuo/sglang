import random
import unittest

import torch

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd,
    decode_attention_fwd_grouped,
    decode_attention_fwd_normal,
)
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd,
    redundant_attention,
)
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)

import triton
import flashinfer
import torch.nn.functional as F


class TestTritonAttention(unittest.TestCase):

    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setUp(self):
        # Set seeds before each test method
        self._set_all_seeds(42)

    def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):
        dtype = torch.bfloat16

        b_seq_len_prefix = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
        )
        b_seq_len_extend = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
        )
        b_seq_len = b_seq_len_prefix + b_seq_len_extend
        max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

        b_req_idx = torch.arange(B, dtype=torch.int32, device="cuda")
        b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
        kv_indices = torch.zeros(
            (b_seq_len_prefix.sum().item(),), dtype=torch.int32, device="cuda"
        )

        for i in range(B):
            kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
            )

        total_token_num = torch.sum(b_seq_len).item()
        extend_token_num = torch.sum(b_seq_len_extend).item()
        k_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)
        v_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)

        k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
        v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
        q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        for i in range(B):
            extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = torch.empty(
                (b_seq_len_extend[i], H_Q, D), dtype=dtype, device="cuda"
            ).normal_(mean=0.1, std=0.2)

        o_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        o_extend_mask = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
        )
        o_redundant = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
        )

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
        qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

        custom_mask = None
        mask_offsets = None

        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            mask_offsets,
            max_len_extend,
        )

        b_seq_mask_len = b_seq_len_extend * b_seq_len
        custom_mask = torch.ones(
            (b_seq_mask_len.sum().item(),), dtype=torch.bool, device="cuda"
        )
        mask_offsets = torch.zeros((B + 1,), dtype=torch.int64, device="cuda")
        mask_offsets[1 : B + 1] = torch.cumsum(b_seq_mask_len[:B], dim=0)
        for i in range(B):
            causal_mask = (
                torch.tril(
                    torch.ones(b_seq_len_extend[i], b_seq_len_extend[i]), diagonal=0
                )
                == 1
            )
            prefix_mask = torch.ones(
                b_seq_len_extend[i], b_seq_len_prefix[i], dtype=torch.bool
            )
            mask_flatten = torch.cat([prefix_mask, causal_mask], dim=1).flatten()
            custom_mask[mask_offsets[i] : mask_offsets[i + 1]] = mask_flatten

        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend_mask,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            mask_offsets,
            max_len_extend,
        )

        redundant_attention(
            q_extend,
            o_redundant,
            k_buffer,
            v_buffer,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            b_seq_len_prefix,
            max_len_in_batch,
        )

        self.assertTrue(torch.allclose(o_extend, o_redundant, rtol=1e-2))
        self.assertTrue(torch.allclose(o_extend_mask, o_redundant, rtol=1e-2))

    def test_extend_attention(self):

        # Define the varying parameter values
        attention_values = [128, 96, 80, 13]

        # Loop through the values and call the method
        for value in attention_values:
            self._test_extend_attention_once(19, 12331, 12, 4, value)

    def _test_context_attention_once(self, head_dim, is_causal):
        # Set up a simple test case
        num_heads = 4
        seq_lens = [8, 12]
        max_seq_len = max(seq_lens)

        # Create random input tensors
        q = torch.randn(sum(seq_lens), num_heads, head_dim, device="cuda")
        k = torch.randn(sum(seq_lens), num_heads, head_dim, device="cuda")
        v = torch.randn(sum(seq_lens), num_heads, head_dim, device="cuda")
        o = torch.zeros(sum(seq_lens), num_heads, head_dim, device="cuda")

        # Create b_start_loc and b_seq_len tensors
        b_start_loc = torch.tensor([0, seq_lens[0]], device="cuda")
        b_seq_len = torch.tensor(seq_lens, device="cuda")

        context_attention_fwd(
            q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=is_causal
        )

        cu_seq_lens = [0] * (len(seq_lens) + 1)
        for i, seq_len in enumerate(seq_lens):
            cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

        for i in range(len(seq_lens)):
            start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
            o_torch = torch.nn.functional.scaled_dot_product_attention(
                q[start:end].permute(1, 0, 2),
                k[start:end].permute(1, 0, 2),
                v[start:end].permute(1, 0, 2),
                is_causal=is_causal,
            ).permute(1, 0, 2)

            cos_sim = torch.nn.functional.cosine_similarity(
                o[start:end].flatten(), o_torch.flatten(), dim=0
            )
            self.assertTrue(cos_sim.item() > 1 - (1e-5))
            self.assertTrue(torch.allclose(o[start:end], o_torch, atol=1e-2))

    def test_context_attention(self):
        head_dim = [128, 96, 80, 13]

        for dim in head_dim:
            for is_causal in [True, False]:
                self._test_context_attention_once(dim, is_causal)

    def _test_decode_attention_once(self, seq_len, B, H_Q, H_KV, D, D_V):
        print("#" * 20)

        dtype = torch.bfloat16
        # seq_len = 16 * 1024  # This represents the number of tokens already in the sequence
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        num_kv_splits = 8
        
        print(f"seq_len: {seq_len}")

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
        # v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device="cuda")
        v_buffer = k_buffer[..., :D_V]

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")

        b_seq_len = torch.full((B,), seq_len, device="cuda")

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
        kv_indices = torch.arange(total_tokens, device="cuda")

        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1),
            dtype=torch.float32,
            device="cuda",
        )

        decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            num_kv_splits,
            sm_scale,
        )
        print(f"q: {q.shape} {q.dtype}")
        print(f"k_buffer: {k_buffer.shape} {k_buffer.dtype}")
        print(f"v_buffer: {v_buffer.shape} {v_buffer.dtype}")
        print(f"o: {o.shape} {o.dtype}")
        print(f"kv_indptr: {kv_indptr.shape} {kv_indptr.dtype}")
        print(f"kv_indices: {kv_indices.shape} {kv_indices.dtype}")
        print(f"attn_logits: {attn_logits.shape} {attn_logits.dtype}")
        print(num_kv_splits)
        
        for i in range(100):
            decode_attention_fwd(
                q,
                k_buffer,
                v_buffer,
                o,
                kv_indptr,
                kv_indices,
                attn_logits,
                num_kv_splits,
                sm_scale,
            )
        torch.cuda.synchronize()
        
        fn = lambda: decode_attention_fwd(
                q,
                k_buffer,
                v_buffer,
                o,
                kv_indptr,
                kv_indices,
                attn_logits,
                num_kv_splits,
                sm_scale,
            )
        
        us = triton.testing.do_bench(fn) * 1e3
        sglang_us = us
        print(f"sglang {us:.2f} us")
        
        decode_attention_fwd(
                q,
                k_buffer,
                v_buffer,
                o,
                kv_indptr,
                kv_indices,
                attn_logits,
                num_kv_splits,
                sm_scale,
            )
        print(f"sglang output : {o.shape} {o.dtype} {o.abs().sum()}")
        sglang_output = o.clone().contiguous()
        
        
        if 1:

            mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0),
                backend="fa3"
            )
            context_len = seq_len
            batch_size = B
            q_indptr = torch.arange(0, B + 1).to(0).int() # for decode, each query length is 1
            kv_lens = torch.full((batch_size,), context_len, dtype=torch.int32).to(0)
            kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * context_len
            kv_indices = torch.arange(0, batch_size * context_len).to(0).int()
            q_nope = q[..., :D_V].clone().contiguous()
            q_pe = q[..., D_V:].clone().contiguous()
            
            num_local_heads = H_Q
            head_dim_ckv = D_V
            head_dim_kpe = D - D_V
            page_size = 1
            # num_local_heads = 128 // 16
            # batch_size = 16
            # head_dim_ckv = 512
            # head_dim_kpe = 64
            # page_size = 1
            # ckv = torch.randn(
            #     batch_size * context_len, 1, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
            # )
            ckv = k_buffer[..., :D_V].clone().contiguous()
            
            # kpe = torch.zeros(
            #     batch_size * context_len, 1, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
            # )
            kpe = k_buffer[..., D_V:].clone().contiguous()
            # print(f"ckv: {ckv.shape} {ckv.dtype}")
            # print(f"kpe: {kpe.shape} {kpe.dtype}")

            # # sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
  
            # print(f"num_local_heads {num_local_heads}")
            
            print(f"H_Q {H_Q}")
            print(f"D {D}")
            print(f"D_V {D_V}")
            print(f"num_local_heads {num_local_heads}")
            
            
            # num_local_heads = 128 // 16
            # batch_size = 16
            # head_dim_ckv = 512
            # head_dim_kpe = 64
            # page_size = 1
            # context_len = seq_len
            # q_indptr = torch.arange(0, batch_size + 1).to(0).int() # for decode, each query length is 1
            # kv_lens = torch.full((batch_size,), context_len, dtype=torch.int32).to(0)
            # kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * context_len
            # kv_indices = torch.arange(0, batch_size * context_len).to(0).int()
            # q_nope = torch.randn(
            #     batch_size * 1, num_local_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
            # )
            # q_pe = torch.zeros(
            #     batch_size * 1, num_local_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
            # )
            # ckv = torch.randn(
            #     batch_size * context_len, 1, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
            # )
            # kpe = torch.zeros(
            #     batch_size * context_len, 1, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
            # )
            # sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption


            
            mla_wrapper.plan(
                q_indptr,
                kv_indptr,
                kv_indices,
                kv_lens,
                num_local_heads,
                head_dim_ckv,
                head_dim_kpe,
                page_size,
                True,  # causal
                sm_scale,
                q_nope.dtype,
                ckv.dtype,
            )

            print(f"q_inptr : {q_indptr.shape} {q_indptr.dtype}")
            print(f"kv_indptr : {kv_indptr.shape} {kv_indptr.dtype}")
            print(f"kv_indices : {kv_indices.shape} {kv_indices.dtype}")
            print(f"kv_lens : {kv_lens}")
            print(f"num_local_heads : {num_local_heads}")
            print(f"head_dim_ckv : {head_dim_ckv}")
            print(f"head_dim_kpe : {head_dim_kpe}")
            print(f"page_size : {page_size}")
            o = mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

            print(o.shape)

            warmup_times = 100
            for i in range(warmup_times):
                o = mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

            torch.cuda.synchronize()
            
            flashinfer_output = mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)
            
            print(f"sglang {sglang_output.shape} {sglang_output.dtype} {sglang_output.abs().sum()}")
            print(f"flashinfer {flashinfer_output.shape} {flashinfer_output.dtype} {flashinfer_output.abs().sum()}")
            
            
            
            sglang_output = sglang_output.double().reshape(-1)
            flashinfer_output = flashinfer_output.double().reshape(-1)
            
            similarity = F.cosine_similarity(sglang_output, flashinfer_output, dim=0)
            
            print(f"flashinfer version {flashinfer.__version__}")
            print(f"similarity {similarity}")



            fn = lambda: mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)
            us = triton.testing.do_bench(fn) * 1e3
            print(f"sglang : {sglang_us:.2f} us")
            print(f"flashinfer : {us:.2f} us")

    def test_decode_attention(self):
        # Here we just to ensure there is no error
        # TODO: correctnesss test

        # Test configurations
        configs = [
            # (2, 4, 4, 64),  # MHA
            # (2, 4, 2, 64),  # GQA
            # (2, 4, 4, 80),  # Non-standard head dim
            # (2, 4, 4, 13),  # Prime number head dim
            # (16, 16, 1, 512 + 64, 512)
            # (64, 16, 1, 512 + 64, 512)
            (16, 16, 1, 512 + 64, 512)
        ]
        
        
            # (16, 8, 8, 512 + 64, 512)
        

        # ss = [16, 14, 12, 10, 8]
        ss = [16]
        ss = [item * 1024 for item in ss]
        print("xxxx")
        print(configs)
        for S in ss:
            for B, H_Q, H_KV, D, D_V in configs:
                self._test_decode_attention_once(S, B, H_Q, H_KV, D, D_V)

    def _test_grouped_decode_attention_once(self, B, S, H_Q, H_KV, D, D_V):
        return

        dtype = torch.bfloat16
        seq_len = S  # This represents the number of tokens already in the sequence
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        num_kv_splits = 8

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
        v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device="cuda")

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")
        o_grouped = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")

        b_seq_len = torch.full((B,), seq_len, device="cuda")

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
        kv_indices = torch.arange(total_tokens, device="cuda")

        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1),
            dtype=torch.float32,
            device="cuda",
        )

        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            num_kv_splits,
            sm_scale,
        )

        attn_logits1 = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1),
            dtype=torch.float32,
            device="cuda",
        )

        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o_grouped,
            kv_indptr,
            kv_indices,
            attn_logits1,
            num_kv_splits,
            sm_scale,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_grouped.flatten(), dim=0
        )
        print(cos_sim.item())
        self.assertTrue(cos_sim.item() > 0.99)
        self.assertTrue(torch.allclose(o, o_grouped, atol=3e-2))

    def test_grouped_decode_attention(self):
        # seq_lens = [5, 100, 128, 500]
        seq_lens = [16 * 1024]
        configs = [
            # (2, 16, 16, 64, 64),
            # (2, 16, 1, 64, 64),
            # (2, 64, 1, 13, 13),
            # (2, 128, 1, 80, 80),
            # (2, 128, 2, 512, 512),
            # (2, 128, 1, 576, 512),
            # (2, 128, 1, 576, 512),
            (16, 8, 8, 512 + 64, 512)
        ]

        for S in seq_lens:
            for B, H_Q, H_KV, D, D_V in configs:
                self._test_grouped_decode_attention_once(B, S, H_Q, H_KV, D, D_V)


if __name__ == "__main__":
    unittest.main()
