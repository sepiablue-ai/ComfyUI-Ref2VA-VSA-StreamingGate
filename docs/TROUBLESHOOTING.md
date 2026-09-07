# Streaming VSA INT8 トラブルシューティング / Troubleshooting

---

## 1. 必須ノードが見つからない / Missing Required Nodes

### `Ref2VAStreamingVSAPatch`

**日本語**

本リポジトリが以下へ配置されているか確認:

```text
ComfyUI/custom_nodes/ComfyUI-Ref2VA-VSA-StreamingGate/
```

ComfyUIを完全再起動してください。

**English**

Verify that this repository is installed at:

```text
ComfyUI/custom_nodes/ComfyUI-Ref2VA-VSA-StreamingGate/
```

Then fully restart ComfyUI.

### `MiniMaxChunkFeedForward`

Repository:

```text
https://github.com/kijai/ComfyUI-KJNodes
```

### `MiniMaxH3FastVAEDecode`

Tested dependency:

```text
https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE
```

**日本語**

古い検証用 ComfyUI への junction / symlink を使っていると、その元環境を削除した時にノードも消えます。公開・本番環境では `custom_nodes` 配下へ実体を置くことを推奨します。

**English**

If the node directory is a junction/symlink into an old test ComfyUI environment, deleting that source environment will also break the node. For a release/production install, keep a real copy/clone under the active ComfyUI `custom_nodes` directory.

---

## 2. `comfy_extras.nodes_sparse_attention` が無い / Missing Core Sparse Attention

**日本語**

この import が失敗する場合、ComfyUI core が古い可能性があります。

必要なのは `comfy_extras/nodes_sparse_attention.py` を含む ComfyUI です。PR #16072 は master に merge 済みです。

- https://github.com/Comfy-Org/ComfyUI/pull/16072

**English**

If this import fails, your ComfyUI core is probably too old.

You need a ComfyUI build containing `comfy_extras/nodes_sparse_attention.py`. PR #16072 has been merged into master.

- https://github.com/Comfy-Org/ComfyUI/pull/16072

---

## 3. Gate重みが無い / Missing Gate Weight

Example:

```text
ValueError: Missing gate for block 0 in fasth3_vsa_gate.safetensors
```

Required file:

```text
fasth3_vsa_gate.safetensors
```

Destination:

```text
ComfyUI/models/loras/
```

Source:

```text
https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo
https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo/resolve/main/fasth3_vsa_gate.safetensors
```

**日本語**

このファイルは通常の LoRA としてロードしません。`Ref2VAStreamingVSAPatch` の `gate_file` へ指定します。

ノードは block 0〜49 の50個の `to_gate_compress` matrix を要求します。ファイルサイズはおおよそ 3.85 GB / 3.59 GiB です。特定の byte 数だけを正常性判定には使わず、配布元の正常なファイルを使用してください。

**English**

Do not load this file as a normal LoRA in the included workflow. Select it in the `gate_file` input of `Ref2VAStreamingVSAPatch`.

The node expects 50 `to_gate_compress` matrices for blocks 0–49. The file is roughly 3.85 GB / 3.59 GiB. Do not use one exact byte count as the only integrity check; use the intact file from the upstream distribution.

---

## 4. CUDA OOM / Out of Memory

### 日本語

確認順:

1. `gate_mode = int8`
2. `MiniMaxChunkFeedForward chunks = 4`
3. `MiniMaxH3FastVAEDecode tile_batch_size = 2`
4. 他の GPU 利用プロセスを停止
5. 必要に応じて `--disable-pinned-memory` を試す

`chunks=2` / `1` は1チャンクの中間 tensor が大きくなり、12GBではピークVRAMが上がります。ただし「必ずOOMする」という意味ではありません。

初回 patch 時は BF16 gate state と INT8 cache が一時的に共存するため、VRAMだけでなく system RAM にも余裕が必要です。

### English

Check in this order:

1. `gate_mode = int8`
2. `MiniMaxChunkFeedForward chunks = 4`
3. `MiniMaxH3FastVAEDecode tile_batch_size = 2`
4. stop other GPU-heavy processes
5. if necessary, try `--disable-pinned-memory`

Using `chunks=2` or `1` increases the size of each FFN intermediate chunk and raises peak VRAM on a 12GB card. This does not mean it always OOMs.

During the initial patch call, the BF16 gate state and INT8 cache can temporarily coexist, so sufficient system RAM is also required.

---

## 5. 生成が遅い / Unexpectedly Slow Generation

### Gate mode

```text
gate_mode = int8
```

Development measurement:

```text
BF16 gate processing ≈ 7.34 s/step
INT8 gate processing ≈ 1.41 s/step
```

### `skip_padding_qkv`

```text
skip_padding_qkv = False
```

Measured at 832×1408:

```text
QKV: 3.52s → 3.91s (~11% slower with True)
Sampler: 108s → 109s (~1% slower with True)
```

### Comfy Compiler

Recommended for the tested v1 stack:

```text
--disable-comfy-compiler
```

**日本語**

これは Compiler 自体が一般に悪いという意味ではありません。MiniMax H3 Sparse Attention との組み合わせで速度低下を実測したため、切り分け・再現性のため無効化しています。

**English**

This does not mean Comfy Compiler is generally bad. A large slowdown was measured with this MiniMax H3 sparse-attention stack, so disabling it is recommended for isolation and reproducibility.

---

## 6. 動画は生成できるがSparseか分からない / Generation Works but Sparse Path May Not Be Active

**日本語**

これは重要です。`h3_eligible()` が False の場合、該当 block は original dense block へフォールバックします。そのため **生成成功だけでは sparse path の動作確認になりません。**

`verbose=True` にして console で:

```text
sparse producer path: ...
[streaming gate block ... mode=int8]
```

を確認してください。

Sparse path 条件:

- CUDA
- BF16 activation
- `head_dim == 128`
- `comfy_kitchen.sol_attn` kernel 使用可能
- token数 >= `min_tokens`
- sigma window内
- MiniMax H3 layout あり

**English**

This is important. If `h3_eligible()` returns False, the block falls back to the original dense block. Therefore, **successful generation alone does not prove that the sparse path was active.**

Enable `verbose=True` and check the console for:

```text
sparse producer path: ...
[streaming gate block ... mode=int8]
```

Sparse-path requirements include:

- CUDA
- BF16 activations
- `head_dim == 128`
- usable `comfy_kitchen.sol_attn` kernel
- token count >= `min_tokens`
- sigma inside the configured window
- valid MiniMax H3 layout

---

## 7. `sink_conditioning` を変えても結果が変わらない / `sink_conditioning` Has No Effect

**日本語**

v1 の Streaming H3 VSA path は:

```python
sink = sink_q = (0, plan["n_prefix"])
```

を直接使用しており、`patch.sinks()` を呼びません。

そのため現在の v1 では `exact_kv_and_rows` / `exact_kv` / `off` を変更しても Streaming H3 VSA path の sink 範囲は変化しません。互換性用 widget として扱ってください。

**English**

The v1 Streaming H3 VSA path directly uses:

```python
sink = sink_q = (0, plan["n_prefix"])
```

and does not call `patch.sinks()`.

Therefore, changing `exact_kv_and_rows` / `exact_kv` / `off` does not currently change the sink range of the Streaming H3 VSA path. Treat this widget as a compatibility field in v1.

---

## 8. `pin_memory()` 関連 / Pinned Memory Behavior

### 日本語

本ノードは Gate INT8 cache に対して `pin_memory()` を試行します。失敗した場合は例外を無視して通常 CPU tensor のまま続行します。

ComfyUI CLI の:

```text
--disable-pinned-memory
```

は本ノードの Gate cache 実装そのものと同じ機構ではありません。正式ベンチマークでは併用しましたが、本ノードの必須要件ではありません。

### English

The node attempts `pin_memory()` on the INT8 gate cache. If it fails, the exception is ignored and normal CPU tensors are used.

ComfyUI's CLI option:

```text
--disable-pinned-memory
```

is not the same mechanism as this node's own gate-cache pinning. It was used in the formal benchmark but is not a hard requirement.

---

## 9. Profiler

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

通常生成・通常ベンチマークでは OFF 推奨です。

### English

Keep the profiler OFF for normal generation and ordinary wall-time benchmarking.

---

## 10. 関連リンク / Related Links

- Kablex Ref2VA VSA: https://github.com/Kablex/ComfyUI-Ref2VA-VSA
- Gate / FastH3 conversion: https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo
- ComfyUI Sparse Attention PR #16072: https://github.com/Comfy-Org/ComfyUI/pull/16072
- KJNodes: https://github.com/kijai/ComfyUI-KJNodes
- FastVAE dependency: https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE
