# Streaming VSA INT8 v1 技術仕様 / Implementation

`ComfyUI-Ref2VA-VSA-StreamingGate` の実装仕様を、日本語・英語で記録します。  
Bilingual implementation notes for `ComfyUI-Ref2VA-VSA-StreamingGate`.

---

## 1. 技術系譜 / Technical Lineage

### 日本語

v1 は次の要素を組み合わせています。

- FastVideo / FastH3: 学習済み VSA gate の起源
- barelymining: `fasth3_vsa_gate.safetensors` の抽出・配布
- Kablex/ComfyUI-Ref2VA-VSA: Ref2VA に VSA を適用する先行実装
- ComfyUI core PR #16072: `SparseAttnPatch`、H3 layout/VSA plan、4096-token chunked sparse producer、`sol_attn_chunked`
- 本ノード: Gate を CPU INT8 cache として保持し、block単位で GPU へ Streaming

### English

v1 combines:

- FastVideo / FastH3: origin of the learned VSA gates
- barelymining: extraction and distribution of `fasth3_vsa_gate.safetensors`
- Kablex/ComfyUI-Ref2VA-VSA: earlier Ref2VA-specific VSA adaptation
- ComfyUI core PR #16072: `SparseAttnPatch`, H3 layout/VSA plan, 4096-token chunked sparse producer, and `sol_attn_chunked`
- this node: CPU INT8 gate cache with per-block GPU streaming

Links:

- https://github.com/FastVideo/FastVideo
- https://github.com/barelymining/ComfyUI-MiniMax-H3-FastVideo
- https://github.com/Kablex/ComfyUI-Ref2VA-VSA
- https://github.com/Comfy-Org/ComfyUI/pull/16072

---

## 2. パイプライン / Pipeline

```text
[MATLOW Fused INT8 Checkpoint]
            │
            ▼
[Ref2VAStreamingVSAPatch]
            │
            ├─ load fasth3_vsa_gate.safetensors
            │
            ├─ extract 50 to_gate_compress matrices
            │
            ├─ CPU BF16 → dynamic INT8 quantization
            │
            ├─ CPU cache (~1.79 GiB INT8 + scales)
            │      └─ pin_memory() attempted
            │
            ├─ each DiT block:
            │      ├─ copy this block's gate to GPU
            │      ├─ ck.int8_linear → coarse gate
            │      ├─ 4096-token qkv producer
            │      ├─ ck.sol_attn_chunked
            │      └─ delete GPU gate tensor
            │
            ▼
[MiniMaxChunkFeedForward chunks=4]
```

### 日本語

Gate 全体を PyTorch module として GPU 常駐させません。実行中には現在の block 分の Gate tensor は一時的に GPU に存在します。

### English

The full gate set is not kept resident on the GPU as model modules. During execution, the gate tensor for the current block temporarily exists on the GPU.

---

## 3. Gate 抽出 / Gate Extraction

`nodes.py` は以下の key 形式を受け付けます。

```text
transformer_blocks.{i}.attn.to_gate_compress.set_weight
transformer_blocks.{i}.attn.to_gate_compress.weight
diffusion_model.blocks.{i}.attn.to_gate_compress.weight
blocks.{i}.attn.to_gate_compress.weight
```

### 日本語

- 2D matrix 以外はエラー
- 同一 block の重複 gate はエラー
- H3 は 50 blocks を要求
- block 0〜49 のどれかが欠けていれば patch 時にエラー

### English

- Non-2D gate tensors are rejected
- Duplicate gates for the same block are rejected
- H3 is expected to contain 50 blocks
- Missing any gate from block 0 through 49 causes an error at patch time

---

## 4. INT8 量子化 / INT8 Quantization

実装 / Code:

```python
w = gates_raw[idx].contiguous().to(dtype=torch.bfloat16, device="cpu")
w_scale = (w.abs().max() / 127.0).float()
w_int8 = (w / w_scale).round().clamp(-128, 127).to(torch.int8)

if torch.cuda.is_available():
    try:
        w_int8 = w_int8.pin_memory()
        w_scale = w_scale.pin_memory()
    except Exception:
        pass
```

### 仕様 / Specification

- Gate shape: `[7168, 5376]`
- Elements / block: `38,535,168`
- Quantized dtype: signed `torch.int8`
- Scale: one FP32 scalar per block matrix (Tensor-wise)
- Quantization: symmetric uniform
- Original `.safetensors` file is not modified

### メモリ / Memory

```text
BF16:
38,535,168 × 2 bytes
≈ 73.50 MiB / block
≈ 3.59 GiB / 50 blocks

INT8 + FP32 scale:
≈ 36.75 MiB / block
≈ 1.79 GiB / 50 blocks
```

### 日本語: CPU RAM注意

patch 初回実行中は、元の BF16 gate state と新しい INT8 cache が一時的に同時存在します。そのため、最終 INT8 cache 約1.79GiB より大きい host RAM を一時的に使用します。

`pin_memory()` が失敗した場合は例外を無視し、通常の CPU tensor で処理を継続します。

### English: Host RAM note

During the first patch call, the original BF16 gate state and the new INT8 cache can temporarily coexist. Host RAM usage can therefore temporarily exceed the final ~1.79GiB INT8 cache size.

If `pin_memory()` fails, the exception is ignored and the node continues with normal CPU tensors.

---

## 5. Block単位 GPU Streaming / Per-Block GPU Streaming

```text
CPU INT8 gate cache
        ↓
current block gate (~36.75 MiB)
        ↓ non_blocking=True
GPU
        ↓
ck.int8_linear
        ↓
coarse_gate
        ↓
4096-token QKV chunks
        ↓
ck.sol_attn_chunked
        ↓
del gate_gpu / scale_gpu
```

### 日本語

Gate は block ごとに1回 GPU へ転送され、その block 内の複数の 4096-token chunk で再利用されます。

### English

The gate is transferred once per block and reused across all 4096-token chunks within that block.

---

## 6. VSA Layout と Prefix 保護 / VSA Layout and Prefix Protection

### 日本語

ComfyUI core の VSA plan は:

- 非video segment を 64-row の segment-pure tile に分割
- video segment を 4×4×4 = 64-token cube に分割
- padding row は `-1` として配置
- `src` / `inv` で padded tile order と native packed order を相互変換

Streaming v1 では:

```python
sink = sink_q = (0, plan["n_prefix"])
```

として、すべての非video prefix tiles を sparse top-k pruning から保護します。

### English

The ComfyUI core VSA plan:

- splits non-video segments into segment-pure 64-row tiles
- splits the video segment into 4×4×4 = 64-token cubes
- represents padding rows as `-1`
- uses `src` / `inv` to map between padded tile order and native packed order

Streaming v1 uses:

```python
sink = sink_q = (0, plan["n_prefix"])
```

so all non-video prefix tiles are protected from sparse top-k pruning.

---

## 7. `sink_conditioning` の実際の挙動 / Actual `sink_conditioning` Behavior

### 日本語

v1 の UI には `sink_conditioning` が存在しますが、Streaming H3 VSA path では `patch.sinks()` を呼ばず、上記の `sink = sink_q = (0, plan["n_prefix"])` を直接使用します。

したがって **`exact_kv_and_rows` / `exact_kv` / `off` を変更しても、現在の Streaming H3 VSA path の sink 範囲は変わりません。**

これは v1 の互換性用 widget です。将来コードで `patch.sinks()` を接続する場合、このドキュメントも更新してください。

### English

The v1 UI exposes `sink_conditioning`, but the Streaming H3 VSA path does not call `patch.sinks()`. It directly uses `sink = sink_q = (0, plan["n_prefix"])`.

Therefore, changing **`exact_kv_and_rows` / `exact_kv` / `off` does not currently change the sink range of the Streaming H3 VSA path.**

Treat it as a compatibility widget in v1. If a future code revision wires `patch.sinks()` into this path, update this documentation.

---

## 8. Sparse Eligibility と Dense Fallback / Sparse Eligibility and Dense Fallback

### 日本語

各 block は `h3_eligible()` で sparse path を使えるか判定します。

主な条件:

- CUDA
- activation dtype = BF16
- `head_dim == 128`
- `ck.sol_attn_is_available(device)` が True
- token数 >= `min_tokens`
- sigma が start/end window 内
- VSA 用 MiniMax H3 layout が存在

条件を満たさない場合:

```python
return extra["original_block"](args)
```

で original dense block に戻ります。

**したがって「動画が完成した = Sparse VSA が動作した」とは限りません。**

`verbose=True` で `sparse producer path` ログを確認してください。

### English

Each block checks `h3_eligible()` before using the sparse path.

Main requirements:

- CUDA
- BF16 activations
- `head_dim == 128`
- `ck.sol_attn_is_available(device)` is True
- token count >= `min_tokens`
- sigma is inside the configured start/end window
- valid MiniMax H3 VSA layout metadata

If the requirements are not met, the patch returns to:

```python
return extra["original_block"](args)
```

which runs the original dense block.

**A successfully generated video therefore does not prove that Sparse VSA was active.**

With `verbose=True`, verify the `sparse producer path` logs.

---

## 9. QKV / FFN

### 日本語

付属 workflow の MATLOW Fused checkpoint は QKV / out_proj / fc1 / fc2 の heavy Linear を INT8 ConvRot で持ちます。本ノードはこれらを再量子化しません。

QKV は 4096-token chunk 単位で `attn.qkv_proj()` を実行して `sol_attn_chunked` へ渡します。

FFN:

```text
MiniMaxChunkFeedForward
chunks = 4
seq_threshold = 4096
```

### English

The MATLOW Fused checkpoint used by the included workflow stores the heavy QKV / out_proj / fc1 / fc2 Linear weights in INT8 ConvRot format. This node does not re-quantize them.

QKV projection is executed in 4096-token chunks and fed into `sol_attn_chunked`.

FFN:

```text
MiniMaxChunkFeedForward
chunks = 4
seq_threshold = 4096
```

---

## 10. ComfyUI Model Management

### 日本語

- `--disable-comfy-compiler`
  - H3 Sparse Attention との組み合わせで大幅な速度低下を実測したため、v1 の再現性重視設定では無効化を推奨
  - Compiler自体が一般に悪いという主張ではありません
- `--disable-pinned-memory`
  - 正式ベンチマークでは使用
  - 本ノードの必須要件ではない
  - 本ノードの Gate CPU cache は独自に `pin_memory()` を試行する

### English

- `--disable-comfy-compiler`
  - Recommended for reproducibility in v1 because a large slowdown was measured with this H3 sparse-attention stack
  - This is not a claim that Comfy Compiler is generally bad
- `--disable-pinned-memory`
  - Used in the formal benchmark
  - Not a hard requirement of this node
  - The node separately attempts `pin_memory()` for its own gate CPU cache

---

## 11. Profiler

Default:

```text
VSA_PROFILE=0
```

Enable:

```text
VSA_PROFILE=1
VSA_PROFILE_PATH=/path/to/result.json
```

### 日本語

通常生成では OFF 推奨です。Profiler は CUDA Event を多数追加するため、通常速度測定と同時に使用しないでください。

### English

Keep it OFF for normal generation. The profiler adds many CUDA events and should not be enabled during ordinary wall-time benchmarking.

---

## 12. ライセンス・帰属 / License and Attribution

### 日本語

**本リポジトリのソースコードは GPL-3.0 で公開します。**

この実装は ComfyUI core の H3 sparse-attention / chunked-producer 構造を適応しています。そのため、配布コードは GPL-3.0 として扱います。

第三者コンポーネント:

- ComfyUI core: GPL-3.0
- Kablex/ComfyUI-Ref2VA-VSA: Apache License 2.0
- barelymining のカスタムノードコード: MIT License
- comfy-kitchen: Apache License 2.0
- FastH3 / VSA Gate 重み: MiniMax H3 Community License

Apache-2.0 / MIT 由来部分の元の著作権表示・帰属は維持してください。  
重みファイルは本リポジトリに同梱せず、各配布元のライセンスに従います。

詳細は root の `LICENSE`、`LICENSE_NOTES.md`、`THIRD_PARTY_NOTICES.md` を参照してください。

### English

**The source code in this repository is distributed under GPL-3.0.**

This implementation adapts the H3 sparse-attention / chunked-producer structure from ComfyUI core, so the distributed source code is treated as GPL-3.0.

Third-party components:

- ComfyUI core: GPL-3.0
- Kablex/ComfyUI-Ref2VA-VSA: Apache License 2.0
- barelymining custom-node code: MIT License
- comfy-kitchen: Apache License 2.0
- FastH3 / VSA gate weights: MiniMax H3 Community License

Retain the original copyright and attribution notices for Apache-2.0 / MIT-derived material.  
Weight files are not bundled and remain governed by their respective upstream licenses.

See the root `LICENSE`, `LICENSE_NOTES.md`, and `THIRD_PARTY_NOTICES.md` for details.
