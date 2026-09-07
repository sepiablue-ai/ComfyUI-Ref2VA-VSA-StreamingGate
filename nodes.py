from __future__ import annotations

import os
import logging
import re
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import folder_paths
import comfy_kitchen as ck

from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy_extras.nodes_sparse_attention import (
    SparseAttnPatch,
    HEAD_DIM,
    BLOCK_SIZE,
    PRODUCER_CHUNK,
    install_override,
    h3_eligible
)

log = logging.getLogger("Ref2VAStreamingVSA")

PROFILE_VSA = os.environ.get("VSA_PROFILE", "0") == "1"

class VSAProfiler:
    def __init__(self):
        self.enabled = PROFILE_VSA
        self.records = []
        self.current_step = -1
        self._current_step_record = None

    def reset(self):
        self.records = []
        self.current_step = -1
        self._current_step_record = None

    def new_step(self, timestep=None):
        if not self.enabled:
            return
        self.current_step += 1
        self._current_step_record = {
            "step_idx": self.current_step,
            "timestep": float(timestep) if timestep is not None else None,
            "blocks": []
        }
        self.records.append(self._current_step_record)

    def start_block(self, block_index):
        if not self.enabled:
            return None
        if self._current_step_record is None or (block_index == 0 and len(self._current_step_record["blocks"]) >= 50):
            self.new_step()
        block_rec = {
            "block_index": block_index,
            "gate_h2d": None,
            "reorder_gather": [],
            "gate_linear": [],
            "qkv_proj": [],
            "sol_total": None,
            "output_unpermute": None,
            "attn_out_proj": None,
            "ffn_mlp": None,
            "block_other": [],
            "block_total": None,
        }
        self._current_step_record["blocks"].append(block_rec)
        return block_rec

    def get_current_block_record(self, block_index):
        if not self.enabled or self._current_step_record is None:
            return None
        for b in reversed(self._current_step_record["blocks"]):
            if b["block_index"] == block_index:
                return b
        return None

    def finish_and_report(self, output_json_path=None):
        if not self.enabled or not self.records:
            return None
        torch.cuda.synchronize()

        step_summaries = []
        total_4step = {
            "gate_h2d_ms": 0.0,
            "reorder_gather_ms": 0.0,
            "gate_linear_ms": 0.0,
            "qkv_proj_ms": 0.0,
            "sol_total_ms": 0.0,
            "sol_kernel_and_coarse_residual_ms": 0.0,
            "output_unpermute_ms": 0.0,
            "attn_out_proj_ms": 0.0,
            "ffn_mlp_ms": 0.0,
            "block_other_ms": 0.0,
            "block_total_ms": 0.0,
        }

        for step_rec in self.records:
            s_sum = {
                "step_idx": step_rec["step_idx"],
                "timestep": step_rec["timestep"],
                "gate_h2d_ms": 0.0,
                "reorder_gather_ms": 0.0,
                "gate_linear_ms": 0.0,
                "qkv_proj_ms": 0.0,
                "sol_total_ms": 0.0,
                "output_unpermute_ms": 0.0,
                "attn_out_proj_ms": 0.0,
                "ffn_mlp_ms": 0.0,
                "block_other_ms": 0.0,
                "block_total_ms": 0.0,
            }
            def _safe_elapsed(ev_start, ev_end):
                if ev_start is None or ev_end is None:
                    return 0.0
                try:
                    return ev_start.elapsed_time(ev_end)
                except Exception:
                    return 0.0

            for b in step_rec["blocks"]:
                if b["gate_h2d"]:
                    s_sum["gate_h2d_ms"] += _safe_elapsed(b["gate_h2d"][0], b["gate_h2d"][1])
                for s, e in b["reorder_gather"]:
                    s_sum["reorder_gather_ms"] += _safe_elapsed(s, e)
                for s, e in b["gate_linear"]:
                    s_sum["gate_linear_ms"] += _safe_elapsed(s, e)
                for s, e in b["qkv_proj"]:
                    s_sum["qkv_proj_ms"] += _safe_elapsed(s, e)
                if b["sol_total"]:
                    s_sum["sol_total_ms"] += _safe_elapsed(b["sol_total"][0], b["sol_total"][1])
                if b["output_unpermute"]:
                    s_sum["output_unpermute_ms"] += _safe_elapsed(b["output_unpermute"][0], b["output_unpermute"][1])
                if b.get("attn_out_proj"):
                    s_sum["attn_out_proj_ms"] += _safe_elapsed(b["attn_out_proj"][0], b["attn_out_proj"][1])
                if b.get("ffn_mlp"):
                    s_sum["ffn_mlp_ms"] += _safe_elapsed(b["ffn_mlp"][0], b["ffn_mlp"][1])
                for s, e in b.get("block_other", []):
                    s_sum["block_other_ms"] += _safe_elapsed(s, e)
                if b.get("block_total"):
                    s_sum["block_total_ms"] += _safe_elapsed(b["block_total"][0], b["block_total"][1])

            s_sum["sol_kernel_and_coarse_residual_ms"] = (
                s_sum["sol_total_ms"]
                - s_sum["reorder_gather_ms"]
                - s_sum["gate_linear_ms"]
                - s_sum["qkv_proj_ms"]
            )
            step_summaries.append(s_sum)

            for k in total_4step:
                total_4step[k] += s_sum[k]

        n_steps = len(step_summaries)
        avg_1step = {k: v / max(n_steps, 1) for k, v in total_4step.items()}

        total_vsa_time_ms = (
            total_4step["gate_h2d_ms"]
            + total_4step["sol_total_ms"]
            + total_4step["output_unpermute_ms"]
        )

        pct = {}
        for k, v in total_4step.items():
            pct[k] = (v / total_vsa_time_ms * 100.0) if total_vsa_time_ms > 0 else 0.0

        report_data = {
            "num_steps": n_steps,
            "total_vsa_time_ms": total_vsa_time_ms,
            "total_4step": total_4step,
            "avg_1step": avg_1step,
            "percentage_of_vsa_time": pct,
            "step_summaries": step_summaries,
        }

        print("\n" + "=" * 84)
        print("                 VSA & DiT BLOCK CUDA EVENT PROFILING REPORT")
        print("=" * 84)
        print(f"Total steps profiled: {n_steps}")
        print(f"Total VSA Attention Time (H2D + Sol + Unpermute): {total_vsa_time_ms:.2f} ms ({total_vsa_time_ms / 1000:.2f} s)")
        print(f"Total Block Time (50 blocks x {n_steps} steps): {total_4step['block_total_ms']:.2f} ms ({total_4step['block_total_ms'] / 1000:.2f} s)")
        print("-" * 84)
        print(f"{'Item':<34} | {'4step Total':<12} | {'1step Avg':<12} | {'% of VSA Time':<12}")
        print("-" * 84)
        print(f"{'gate H2D':<34} | {total_4step['gate_h2d_ms']:>9.2f} ms | {avg_1step['gate_h2d_ms']:>9.2f} ms | {pct['gate_h2d_ms']:>10.2f} %")
        print(f"{'reorder/gather':<34} | {total_4step['reorder_gather_ms']:>9.2f} ms | {avg_1step['reorder_gather_ms']:>9.2f} ms | {pct['reorder_gather_ms']:>10.2f} %")
        print(f"{'gate linear':<34} | {total_4step['gate_linear_ms']:>9.2f} ms | {avg_1step['gate_linear_ms']:>9.2f} ms | {pct['gate_linear_ms']:>10.2f} %")
        print(f"{'QKV projection':<34} | {total_4step['qkv_proj_ms']:>9.2f} ms | {avg_1step['qkv_proj_ms']:>9.2f} ms | {pct['qkv_proj_ms']:>10.2f} %")
        print(f"{'sol kernel + coarse residual':<34} | {total_4step['sol_kernel_and_coarse_residual_ms']:>9.2f} ms | {avg_1step['sol_kernel_and_coarse_residual_ms']:>9.2f} ms | {pct['sol_kernel_and_coarse_residual_ms']:>10.2f} %")
        print(f"{'output unpermute':<34} | {total_4step['output_unpermute_ms']:>9.2f} ms | {avg_1step['output_unpermute_ms']:>9.2f} ms | {pct['output_unpermute_ms']:>10.2f} %")
        print(f"{'attn out_proj':<34} | {total_4step['attn_out_proj_ms']:>9.2f} ms | {avg_1step['attn_out_proj_ms']:>9.2f} ms | {pct['attn_out_proj_ms']:>10.2f} %")
        print("-" * 84)
        print(f"{'sol_total (sum of inner 4)':<34} | {total_4step['sol_total_ms']:>9.2f} ms | {avg_1step['sol_total_ms']:>9.2f} ms | {pct['sol_total_ms']:>10.2f} %")
        print("-" * 84)
        print(f"{'DiT Block FFN (MLP)':<34} | {total_4step['ffn_mlp_ms']:>9.2f} ms | {avg_1step['ffn_mlp_ms']:>9.2f} ms | {'-':>12}")
        print(f"{'DiT Block Other (AdaLN/Norm/Mod)':<34} | {total_4step['block_other_ms']:>9.2f} ms | {avg_1step['block_other_ms']:>9.2f} ms | {'-':>12}")
        print(f"{'DiT Block Total (Attention+FFN+Other)':<34} | {total_4step['block_total_ms']:>9.2f} ms | {avg_1step['block_total_ms']:>9.2f} ms | {'-':>12}")
        print("=" * 84)

        for s in step_summaries:
            print(f"[Step {s['step_idx']} (t={s['timestep']})] VSA: {s['sol_total_ms']:.1f}ms (QKV: {s['qkv_proj_ms']:.1f}ms, Gate: {s['gate_linear_ms']:.1f}ms, Sol: {s['sol_kernel_and_coarse_residual_ms']:.1f}ms, OutP: {s['attn_out_proj_ms']:.1f}ms) | FFN: {s['ffn_mlp_ms']:.1f}ms | BlkTot: {s['block_total_ms']:.1f}ms")
        print("=" * 84 + "\n")

        import os
        actual_output_path = os.environ.get("VSA_PROFILE_PATH", output_json_path)
        if actual_output_path:
            import json
            try:
                with open(actual_output_path, "w", encoding="utf-8") as f:
                    json.dump(report_data, f, indent=2)
                print(f"[Ref2VAStreamingVSA] Profile written to {actual_output_path}")
            except Exception as e:
                print(f"[Ref2VAStreamingVSA] Failed to write profile JSON: {e}")

        return report_data

PROFILER = VSAProfiler()


def _extract_gate_weights(state):
    patterns = (
        re.compile(r"^transformer_blocks\.(\d+)\.attn\.to_gate_compress\.set_weight$"),
        re.compile(r"^transformer_blocks\.(\d+)\.attn\.to_gate_compress\.weight$"),
        re.compile(r"^diffusion_model\.blocks\.(\d+)\.attn\.to_gate_compress\.weight$"),
        re.compile(r"^blocks\.(\d+)\.attn\.to_gate_compress\.weight$"),
    )
    gates = {}
    for key, value in state.items():
        block_index = None
        for pattern in patterns:
            match = pattern.match(str(key))
            if match:
                block_index = int(match.group(1))
                break
        if block_index is None:
            continue
        if value.ndim != 2:
            raise ValueError(
                f"FastH3 VSA gate must be a matrix, got {key} with shape {tuple(value.shape)}"
            )
        if block_index in gates:
            raise ValueError(f"duplicate FastH3 VSA gate for block {block_index}")
        gates[block_index] = value
    return gates


def h3_streaming_vsa_attention(attn, x, rope_freqs, transformer_options, patch: SparseAttnPatch, block_index, gate_cpu_tensor, gate_mode="int8", skip_padding_qkv=False):
    """H3 attention through chunked producer with streaming gate weight (BF16 or INT8):
    1. gate_gpu is transferred from CPU (pinned) memory ONCE per block.
    2. All 4096-token chunks in this block reuse gate_gpu.
    3. gate_gpu is deleted immediately after sol_attn_chunked finishes.
    4. Model never holds 3.6GB gate layers as PyTorch modules.
    5. Optionally skips QKV & Gate computation on padded dummy rows for ~13% GEMM savings.
    """
    global PROFILER
    rec = PROFILER.get_current_block_record(block_index) if PROFILE_VSA else None
    if rec is None and PROFILE_VSA:
        rec = PROFILER.start_block(block_index)

    n_tokens = x.shape[0]
    heads, head_dim = attn.heads, attn.head_dim
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)

    # 1. gate_h2d_ms: Transfer gate weight to GPU ONCE per block
    if rec is not None:
        ev_h2d_s = torch.cuda.Event(enable_timing=True)
        ev_h2d_e = torch.cuda.Event(enable_timing=True)
        ev_h2d_s.record()

    if gate_mode == "int8":
        w_int8_cpu, w_scale_cpu = gate_cpu_tensor
        gate_gpu = w_int8_cpu.to(device=x.device, non_blocking=True)
        scale_gpu = w_scale_cpu.to(device=x.device, non_blocking=True)
    else:
        gate_gpu = gate_cpu_tensor.to(device=x.device, dtype=x.dtype, non_blocking=True)
        scale_gpu = None

    if rec is not None:
        ev_h2d_e.record()
        rec["gate_h2d"] = (ev_h2d_s, ev_h2d_e)

    plan = patch.vsa_plan(transformer_options["minimax_h3_layout"], x.device)
    n, freqs = plan["n"], patch.vsa_rope_freqs(rope_freqs, plan)
    sink = sink_q = (0, plan["n_prefix"])
    extra = {
        "tail": False,
        "block_len": plan["block_len"],
        "coarse_gate": x.new_empty(n, heads * head_dim).view(1, n, heads, head_dim)
    }

    def chunks():
        for i in range(0, n, PRODUCER_CHUNK):
            if rec is not None:
                ev_g_s = torch.cuda.Event(enable_timing=True)
                ev_g_e = torch.cuda.Event(enable_timing=True)
                ev_g_s.record()

            idx = plan["src"][i:i + PRODUCER_CHUNK]
            chunk_len = idx.shape[0]
            mask = (idx >= 0)
            all_valid = mask.all().item()

            if not skip_padding_qkv or all_valid:
                # All valid rows, or skip disabled (identical to original path)
                xc = x[idx.clamp_min(0)]
                if not all_valid:
                    xc = xc * mask.unsqueeze(1).to(x.dtype)

                if rec is not None:
                    ev_g_e.record()
                    ev_l_s = torch.cuda.Event(enable_timing=True)
                    ev_l_e = torch.cuda.Event(enable_timing=True)
                    ev_l_s.record()

                # Re-use gate_gpu for each chunk without re-transferring from CPU
                if gate_mode == "int8":
                    extra["coarse_gate"].view(n, heads * head_dim)[i:i + chunk_len] = ck.int8_linear(
                        xc, gate_gpu, scale_gpu, out_dtype=x.dtype
                    )
                else:
                    extra["coarse_gate"].view(n, heads * head_dim)[i:i + chunk_len] = F.linear(xc, gate_gpu)

                if rec is not None:
                    ev_l_e.record()
                    ev_q_s = torch.cuda.Event(enable_timing=True)
                    ev_q_e = torch.cuda.Event(enable_timing=True)
                    ev_q_s.record()

                proj = attn.qkv_proj(xc)

                if rec is not None:
                    ev_q_e.record()
                    rec["reorder_gather"].append((ev_g_s, ev_g_e))
                    rec["gate_linear"].append((ev_l_s, ev_l_e))
                    rec["qkv_proj"].append((ev_q_s, ev_q_e))

                yield proj
            else:
                # Padded rows exist and skip_padding_qkv is enabled
                valid_count = mask.sum().item()
                if valid_count == 0:
                    # Entire chunk is padding (no GEMM needed)
                    extra["coarse_gate"].view(n, heads * head_dim)[i:i + chunk_len].zero_()
                    if rec is not None:
                        ev_g_e.record()
                        ev_l_s = torch.cuda.Event(enable_timing=True)
                        ev_l_e = torch.cuda.Event(enable_timing=True)
                        ev_l_s.record()
                        ev_l_e.record()
                        ev_q_s = torch.cuda.Event(enable_timing=True)
                        ev_q_e = torch.cuda.Event(enable_timing=True)
                        ev_q_s.record()
                        ev_q_e.record()
                        rec["reorder_gather"].append((ev_g_s, ev_g_e))
                        rec["gate_linear"].append((ev_l_s, ev_l_e))
                        rec["qkv_proj"].append((ev_q_s, ev_q_e))
                    proj = torch.zeros((chunk_len, heads * head_dim * 3), dtype=x.dtype, device=x.device)
                    yield proj
                else:
                    valid_idx = idx[mask]
                    xc_valid = x[valid_idx]

                    if rec is not None:
                        ev_g_e.record()
                        ev_l_s = torch.cuda.Event(enable_timing=True)
                        ev_l_e = torch.cuda.Event(enable_timing=True)
                        ev_l_s.record()

                    if gate_mode == "int8":
                        gate_valid = ck.int8_linear(xc_valid, gate_gpu, scale_gpu, out_dtype=x.dtype)
                    else:
                        gate_valid = F.linear(xc_valid, gate_gpu)

                    gate_dest = extra["coarse_gate"].view(n, heads * head_dim)[i:i + chunk_len]
                    gate_dest.zero_()
                    gate_dest[mask] = gate_valid

                    if rec is not None:
                        ev_l_e.record()
                        ev_q_s = torch.cuda.Event(enable_timing=True)
                        ev_q_e = torch.cuda.Event(enable_timing=True)
                        ev_q_s.record()

                    proj_valid = attn.qkv_proj(xc_valid)
                    proj = torch.zeros((chunk_len, proj_valid.shape[-1]), dtype=proj_valid.dtype, device=proj_valid.device)
                    proj[mask] = proj_valid

                    if rec is not None:
                        ev_q_e.record()
                        rec["reorder_gather"].append((ev_g_s, ev_g_e))
                        rec["gate_linear"].append((ev_l_s, ev_l_e))
                        rec["qkv_proj"].append((ev_q_s, ev_q_e))

                    yield proj

    key = (block_index, n, tuple(transformer_options.get("uuids", ())))
    pooled = patch.pooled.get(key)

    if rec is not None:
        ev_sol_s = torch.cuda.Event(enable_timing=True)
        ev_sol_e = torch.cuda.Event(enable_timing=True)
        ev_sol_s.record()

    out, kmean, vscale = ck.sol_attn_chunked(
        chunks, n, heads, freqs, (qw, kw),
        kmean=None if pooled is None else pooled[0],
        vscale=None if pooled is None else pooled[1],
        tau=patch.tau, topk_ratio=patch.topk_ratio, token_aug=patch.extra_tokens,
        sink_blocks=list(sink), sink_q=list(sink_q),
        rope_eps=attn.q_norm.eps, **extra)

    if rec is not None:
        ev_sol_e.record()
        rec["sol_total"] = (ev_sol_s, ev_sol_e)

    patch.pooled[key] = (kmean, vscale)

    # Immediately release GPU gate weight for this block
    del gate_gpu
    if scale_gpu is not None:
        del scale_gpu

    mode = f"VSA tiles ({n} padded rows, {sink[1]} prefix tiles, skip_pad={skip_padding_qkv})"
    patch.log_once(("producer", n), f"sparse producer path: {n_tokens} tokens, {mode} [streaming gate block {block_index} mode={gate_mode}]")
    out = out.view(n, heads * head_dim)

    if rec is not None:
        ev_unperm_s = torch.cuda.Event(enable_timing=True)
        ev_unperm_e = torch.cuda.Event(enable_timing=True)
        ev_unperm_s.record()

    out = out[plan["inv"]]

    if rec is not None:
        ev_unperm_e.record()
        rec["output_unpermute"] = (ev_unperm_s, ev_unperm_e)

    if rec is not None:
        ev_out_s = torch.cuda.Event(enable_timing=True)
        ev_out_e = torch.cuda.Event(enable_timing=True)
        ev_out_s.record()

    res_out = attn.out_proj(out)

    if rec is not None:
        ev_out_e.record()
        rec["attn_out_proj"] = (ev_out_s, ev_out_e)

    return res_out


def make_h3_streaming_block_patch(block, block_index, patch: SparseAttnPatch, gate_cpu_tensor, gate_mode="int8", skip_padding_qkv=False):
    def attention(h, rope_freqs=None, transformer_options={}):
        return h3_streaming_vsa_attention(block.attn, h, rope_freqs, transformer_options, patch, block_index, gate_cpu_tensor, gate_mode=gate_mode, skip_padding_qkv=skip_padding_qkv)

    def block_patch(args, extra):
        global PROFILER
        if not h3_eligible(block.attn, args["img"], args["rope_freqs"], args["transformer_options"], patch, block_index):
            return extra["original_block"](args)

        rec = PROFILER.start_block(block_index) if PROFILE_VSA else None
        if rec is not None:
            ev_blk_s = torch.cuda.Event(enable_timing=True)
            ev_blk_e = torch.cuda.Event(enable_timing=True)
            ev_blk_s.record()

            from comfy.ldm.minimax.model import _mod_scale_shift, _mod_gate
            x = args["img"]
            t_emb = args["t_emb"]
            mod_segments = args["mod_segments"]
            rope_freqs = args["rope_freqs"]
            transformer_options = args["transformer_options"]

            ev_other1_s = torch.cuda.Event(enable_timing=True)
            ev_other1_e = torch.cuda.Event(enable_timing=True)
            ev_other1_s.record()
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
            h = _mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments)
            ev_other1_e.record()

            attn_out = attention(h, rope_freqs=rope_freqs, transformer_options=transformer_options)

            ev_other2_s = torch.cuda.Event(enable_timing=True)
            ev_other2_e = torch.cuda.Event(enable_timing=True)
            ev_other2_s.record()
            x = _mod_gate(x, gate_msa, attn_out, mod_segments)
            h = _mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_segments)
            ev_other2_e.record()

            ev_ffn_s = torch.cuda.Event(enable_timing=True)
            ev_ffn_e = torch.cuda.Event(enable_timing=True)
            ev_ffn_s.record()
            mlp_out = block.mlp(h)
            ev_ffn_e.record()

            ev_other3_s = torch.cuda.Event(enable_timing=True)
            ev_other3_e = torch.cuda.Event(enable_timing=True)
            ev_other3_s.record()
            x = _mod_gate(x, gate_mlp, mlp_out, mod_segments)
            ev_other3_e.record()

            ev_blk_e.record()
            rec["block_total"] = (ev_blk_s, ev_blk_e)
            rec["ffn_mlp"] = (ev_ffn_s, ev_ffn_e)
            rec["block_other"] = [(ev_other1_s, ev_other1_e), (ev_other2_s, ev_other2_e), (ev_other3_s, ev_other3_e)]
            return {"img": x}
        else:
            args = {**args, "attention": attention}
            return extra["original_block"](args)

    return block_patch


class Ref2VAStreamingVSAPatch:
    """FastH3 VSA Block Sparse Attention with Streaming Gate Weights.
    
    Loads the 50 FastH3 to_gate_compress matrices on CPU (pinned memory)
    and transfers each block's gate weight to GPU only during that block's attention
    computation, reusing it across 4096-token chunks and releasing it immediately.
    Avoids 3.6GB gate residency and dynamic offload thrashing on 12GB VRAM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "gate_file": (folder_paths.get_filename_list("loras"),),
                "gate_mode": (["int8", "bf16"], {"default": "int8"}),
                "keep_percent": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 95.0, "step": 0.5}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_tokens": ("INT", {"default": 12288, "min": 0, "max": 1000000}),
                "sink_conditioning": (["exact_kv_and_rows", "exact_kv", "off"], {"default": "exact_kv_and_rows"}),
                "skip_padding_qkv": ("BOOLEAN", {"default": False}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "FastH3/VSA"
    DESCRIPTION = (
        "Applies VSA (FastVideo) attention with streaming gate weights. "
        "Gates remain on CPU and stream to GPU block-by-block, eliminating 3.6GB GPU residency."
    )

    def patch(self, model, gate_file, gate_mode="int8", keep_percent=5.0, start_percent=0.0, end_percent=1.0,
              min_tokens=12288, sink_conditioning="exact_kv_and_rows", skip_padding_qkv=False, verbose=True):
        gate_path = folder_paths.get_full_path_or_raise("loras", gate_file)
        raw = comfy.utils.load_torch_file(gate_path, safe_load=True)
        gates_raw = _extract_gate_weights(raw)

        diffusion_model = model.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise ValueError("Ref2VA Streaming VSA requires a native MiniMaxH3Model")

        blocks = list(getattr(diffusion_model, "blocks", ()))
        if len(blocks) != 50:
            raise RuntimeError(f"expected 50 H3 DiT blocks, found {len(blocks)}")

        # Prepare CPU (pinned) gate weights
        gates_cpu = {}
        for idx in range(50):
            if idx not in gates_raw:
                raise ValueError(f"Missing gate for block {idx} in {gate_file}")
            w = gates_raw[idx].contiguous().to(dtype=torch.bfloat16, device="cpu")
            if gate_mode == "int8":
                w_scale = (w.abs().max() / 127.0).float()
                w_int8 = (w / w_scale).round().clamp(-128, 127).to(torch.int8)
                if torch.cuda.is_available():
                    try:
                        w_int8 = w_int8.pin_memory()
                        w_scale = w_scale.pin_memory()
                    except Exception:
                        pass
                gates_cpu[idx] = (w_int8, w_scale)
            else:
                if torch.cuda.is_available():
                    try:
                        w = w.pin_memory()
                    except Exception:
                        pass
                gates_cpu[idx] = w

        model_sampling = model.get_model_object("model_sampling")
        sigma_start = float(model_sampling.percent_to_sigma(start_percent))
        sigma_end = float(model_sampling.percent_to_sigma(end_percent))

        patch = SparseAttnPatch(
            tau=1.3,
            topk_ratio=keep_percent / 100.0,
            vsa=True,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            min_tokens=min_tokens,
            dense_blocks=set(),
            sink_conditioning=sink_conditioning,
            extra_tokens=0,
            verbose=verbose
        )

        def on_prepare_state(model_patcher, timestep, model_options):
            install_override(patch, model_options["transformer_options"])
            if PROFILE_VSA:
                PROFILER.new_step(timestep=timestep)

        def on_cleanup(model_patcher):
            patch.reset()
            if PROFILE_VSA:
                PROFILER.finish_and_report(output_json_path=os.environ.get("VSA_PROFILE_PATH", None))

        m = model.clone()
        if PROFILE_VSA:
            PROFILER.reset()
        install_override(patch, m.model_options["transformer_options"])
        m.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE,
            "ref2va_streaming_vsa",
            on_prepare_state
        )
        m.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
            "ref2va_streaming_vsa",
            on_cleanup
        )

        # Set block patches with streaming gate
        for i, block in enumerate(diffusion_model.blocks):
            m.set_model_patch_replace(
                make_h3_streaming_block_patch(block, i, patch, gates_cpu[i], gate_mode=gate_mode, skip_padding_qkv=skip_padding_qkv),
                "dit", "double_block", i
            )

        log.info(
            f"Ref2VA Streaming VSA installed: mode={gate_mode}, keep_percent={keep_percent}%, "
            f"skip_padding_qkv={skip_padding_qkv}, start={start_percent}, end={end_percent}, 50 gates pinned on CPU."
        )
        return (m,)


NODE_CLASS_MAPPINGS = {
    "Ref2VAStreamingVSAPatch": Ref2VAStreamingVSAPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ref2VAStreamingVSAPatch": "Ref2VA Streaming VSA Patch (12GB Low-VRAM)",
}

