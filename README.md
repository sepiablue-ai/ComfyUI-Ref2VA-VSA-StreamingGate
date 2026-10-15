# ComfyUI-Ref2VA-VSA-StreamingGate v1

MiniMax H3 Reference-to-Video (Ref2VA) 向けの **Streaming VSA INT8** 低VRAMカスタムノードです。  
A low-VRAM **Streaming VSA INT8** custom node for MiniMax H3 Reference-to-Video (Ref2VA).

> 作者 / Author：かみもと（Kamimoto）｜検証速報 / Quick results：**[X（@sep_is_heim）](https://x.com/sep_is_heim)**｜解説記事 / Write-ups：**[note](https://note.com/sepiablue)**（[記事INDEX / Article index](https://note.com/sepiablue/n/n8e612c619a3e)）

> **より新しい構成 / Newer configuration:** 1088×1920出力でより速いRef2VAとして、[ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream)のノードを使う[さやかベンチマーク](https://sayakabenchmark.pages.dev/) Ref2VA [020](https://sayakabenchmark.pages.dev/benchmarks/ref2va/020-matlow-fused4-sla5-x2vae-int8-stream/workflow.json)があります（544×960で生成、68.9秒）。 / For faster Ref2VA with 1088×1920 output, see Sayaka Benchmark Ref2VA 020 (544×960 sampling, 68.9 s), which uses the ComfyUI-H3-X2-Stream nodes.

---

## 1. 概要 / Overview

### 日本語

このノードは、FastH3 由来の学習済み Visual Sparse Attention (VSA) Gate を **GPUへ全体常駐させず**、CPU上で INT8 量子化して保持し、各 DiT ブロックの Attention 実行時にそのブロック分だけ GPU へ転送します。

RTX 4070 12GB 環境で、MATLOW Fused checkpoint を用いた 1024×1792 / 124 frames / Ref 3枚の MiniMax H3 Ref2VA を完走確認しています。

主な特徴:

- `fasth3_vsa_gate.safetensors` を CPU 上で動的 INT8 化
- Gate 転送量を BF16 約 73.5 MiB/block → INT8 約 36.8 MiB/block に削減
- Gate は DiT block ごとに GPU へ転送し、Attention 終了後に GPU tensor を解放
- QKV は 4096-token chunk producer と `comfy_kitchen.sol_attn_chunked` を使用
- 12GB VRAM の実機環境で Ref2VA + VSA を動作確認
- 画質・音質の優位性は主張しません。正式追試では方式間の明確な品質差は再現しませんでした

正式比較では、Ref 3枚の入力負荷で:

- MATLOW Fused SLA95: **142s**
- Streaming VSA INT8: **165s**
- FastH3 VSA95 + Ref3: **167s**

でした。ただし、MATLOW系と FastH3 系では checkpoint / Sampler / Sigma / ChunkFFN 設定が異なるため、Attention 実装だけを分離した純粋な A/B 比較ではありません。

### English

This node keeps the learned FastH3 Visual Sparse Attention (VSA) gate **off the GPU as a full resident model component**. The gate is dynamically quantized to INT8 on CPU, and only the gate matrix for the currently executing DiT block is transferred to the GPU.

The workflow has been tested on an RTX 4070 12GB at 1024×1792 / 124 frames with three Ref2VA reference images using the MATLOW Fused checkpoint.

Key features:

- Dynamically quantizes `fasth3_vsa_gate.safetensors` to INT8 on CPU
- Reduces gate transfer size from about 73.5 MiB/block in BF16 to about 36.8 MiB/block in INT8
- Streams one DiT block's gate to GPU at a time and releases the GPU tensor after attention finishes
- Uses a 4096-token QKV producer with `comfy_kitchen.sol_attn_chunked`
- Verified to run Ref2VA + VSA on a 12GB GPU
- Does **not** claim general image-quality or audio-quality superiority; later controlled checks did not reproduce a clear quality advantage

In the formal comparison with three reference images:

- MATLOW Fused SLA95: **142s**
- Streaming VSA INT8: **165s**
- FastH3 VSA95 + Ref3: **167s**

The MATLOW and FastH3 pipelines use different checkpoints, samplers, sigma schedules, and ChunkFFN settings, so this is a practical workflow comparison rather than an attention-only A/B test.

See [BENCHMARK.md](docs/BENCHMARK.md) for details.

---

## 2. 派生元・技術系譜 / Upstream and Technical Lineage

### 日本語

この実装の技術系譜は以下です。

1. **FastVideo / FastH3 VSA**
   - VSA の学習済み `to_gate_compress` Gate の起源
   - https://github.com/FastVideo/FastVideo
2. **barelymining/ComfyUI-MiniMax-H3-FastVideo**
   - FastVideo の VSA-DataFree checkpoint から抽出した `fasth3_vsa_gate.safetensors` の配布元
   - https://github.com/barelymining/ComfyUI-MiniMax-H3-FastVideo
   - https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo
3. **Kablex/ComfyUI-Ref2VA-VSA**
   - Ref2VA へ VSA を適用する発想、prefix/video の分離、Gate transplant の先行実装
   - https://github.com/Kablex/ComfyUI-Ref2VA-VSA
4. **ComfyUI Core Sparse Attention / PR #16072**
   - 現在の v1 が利用する `SparseAttnPatch`、H3 VSA plan、4096-token chunked producer、`sol_attn_chunked` 経路の基盤
   - https://github.com/Comfy-Org/ComfyUI/pull/16072
   - `comfy_extras/nodes_sparse_attention.py`

**Kablex版カスタムノードや barelymining版カスタムノードを別途インストールする必要はありません。**  
本ノードが必要とするのは、それらのうち **Gate重み `fasth3_vsa_gate.safetensors`** と、下記の実行依存関係です。

### English

The implementation lineage is:

1. **FastVideo / FastH3 VSA**
   - Origin of the trained VSA `to_gate_compress` gates
   - https://github.com/FastVideo/FastVideo
2. **barelymining/ComfyUI-MiniMax-H3-FastVideo**
   - Distribution source for `fasth3_vsa_gate.safetensors`, extracted from FastVideo's VSA-DataFree checkpoint
   - https://github.com/barelymining/ComfyUI-MiniMax-H3-FastVideo
   - https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo
3. **Kablex/ComfyUI-Ref2VA-VSA**
   - Earlier Ref2VA-specific VSA adaptation, including prefix/video separation and gate-transplant ideas
   - https://github.com/Kablex/ComfyUI-Ref2VA-VSA
4. **ComfyUI Core Sparse Attention / PR #16072**
   - Basis for the v1 `SparseAttnPatch`, H3 VSA plan, 4096-token chunked producer, and `sol_attn_chunked` path
   - https://github.com/Comfy-Org/ComfyUI/pull/16072
   - `comfy_extras/nodes_sparse_attention.py`

You do **not** need to install the Kablex or barelymining custom-node repositories separately.  
This node only requires the **gate weight file `fasth3_vsa_gate.safetensors`** from that lineage, plus the runtime dependencies listed below.

---

## 3. 必須モデル・重み / Required Models and Weights

### 日本語

付属 v1 workflow では以下を使用します。

| 種類 | ファイル | 配置先 |
|---|---|---|
| Diffusion Model | `minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models/` |
| **VSA Gate** | **`fasth3_vsa_gate.safetensors`** | **`ComfyUI/models/loras/`** |
| Text Encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `ComfyUI/models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | `ComfyUI/models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `ComfyUI/models/vae/` |

配布元 / Direct sources:

- MATLOW Fused:
  - https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot
  - https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/resolve/main/diffusion_models/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors
- **VSA Gate（必須）**:
  - https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo
  - https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo/resolve/main/fasth3_vsa_gate.safetensors
- Text Encoder:
  - https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
- Video VAE:
  - https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_video_vae_int8_convrot.safetensors
- Audio VAE:
  - https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors

**重要:** `fasth3_vsa_gate.safetensors` は通常の LoRA ではありません。付属 workflow では `LoraLoader` に入れず、`Ref2VAStreamingVSAPatch` の `gate_file` へ指定します。

MATLOW Fused checkpoint には Turbo / motion-related merge が既に含まれているため、付属 v1 workflow では Kablex版READMEにある Ref2VA Turbo LoRA や barelymining版 FastH3 LoRA を追加ロードしません。

### English

The included v1 workflow uses:

| Type | File | Destination |
|---|---|---|
| Diffusion Model | `minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models/` |
| **VSA Gate** | **`fasth3_vsa_gate.safetensors`** | **`ComfyUI/models/loras/`** |
| Text Encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `ComfyUI/models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | `ComfyUI/models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `ComfyUI/models/vae/` |

Direct sources:

- MATLOW Fused:
  - https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot
  - https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/resolve/main/diffusion_models/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors
- **VSA Gate (required)**:
  - https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo
  - https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo/resolve/main/fasth3_vsa_gate.safetensors
- Text Encoder:
  - https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
- Video VAE:
  - https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_video_vae_int8_convrot.safetensors
- Audio VAE:
  - https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors

**Important:** `fasth3_vsa_gate.safetensors` is not a normal LoRA in this workflow. Do not load it through `LoraLoader`; select it in the `gate_file` input of `Ref2VAStreamingVSAPatch`.

The MATLOW Fused checkpoint already contains the Turbo / motion-related merge used by this workflow, so the included v1 graph does not separately load the Ref2VA Turbo LoRA from the Kablex example or the FastH3 LoRA from the barelymining stack.

---

## 4. 実行依存関係 / Runtime Dependencies

### 日本語

必須:

1. **ComfyUI**
   - `comfy_extras.nodes_sparse_attention` を含む版
   - PR #16072 は 2026-09-06 に master へ merge 済み
2. **ComfyUI-KJNodes**
   - https://github.com/kijai/ComfyUI-KJNodes
   - `MiniMaxChunkFeedForward` を使用
3. **ComfyUI-MiniMax-H3-MotionCache-FastVAE**
   - https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE
   - `MiniMaxH3FastVAEDecode` を使用
4. **comfy-kitchen**
   - `int8_linear` と `sol_attn_chunked` を使用
   - v1 正式環境では `0.2.33` を確認

追加の `vsa` Python package は、この v1 ノードの実行には不要です。VSA kernel は ComfyUI core + comfy-kitchen 経路を使用します。

### English

Required:

1. **ComfyUI**
   - A build containing `comfy_extras.nodes_sparse_attention`
   - PR #16072 was merged into master on 2026-09-06
2. **ComfyUI-KJNodes**
   - https://github.com/kijai/ComfyUI-KJNodes
   - Provides `MiniMaxChunkFeedForward`
3. **ComfyUI-MiniMax-H3-MotionCache-FastVAE**
   - https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE
   - Provides `MiniMaxH3FastVAEDecode`
4. **comfy-kitchen**
   - Provides `int8_linear` and `sol_attn_chunked`
   - v1 was formally tested with `0.2.33`

The separate FastVideo `vsa` Python package is **not required** for this v1 node. The sparse kernel path comes from ComfyUI core + comfy-kitchen.

---

## 5. 実機確認環境 / Tested Environment

### 日本語

- OS: Windows 11 64-bit
- GPU: NVIDIA GeForce RTX 4070 12GB
- Python: 3.13.14
- PyTorch: 2.13.0+cu130
- comfy-kitchen: 0.2.33
- ComfyUI: master 系（2026年9月時点）
- 正式ベンチマーク時:
  - `--disable-comfy-compiler`
  - `--disable-pinned-memory` も併用

`--disable-pinned-memory` は本ノードの必須要件ではありません。本ノード自身は Gate CPU tensor に対して明示的に `pin_memory()` を試行します。

### English

- OS: Windows 11 64-bit
- GPU: NVIDIA GeForce RTX 4070 12GB
- Python: 3.13.14
- PyTorch: 2.13.0+cu130
- comfy-kitchen: 0.2.33
- ComfyUI: master branch, September 2026
- Formal benchmark launch options:
  - `--disable-comfy-compiler`
  - `--disable-pinned-memory` was also used

`--disable-pinned-memory` is **not a hard requirement** of this node. The node explicitly attempts `pin_memory()` on its own CPU gate tensors.

---

## 6. 推奨設定 / Recommended Settings

| Setting | v1 recommended | 日本語 / English |
|---|---:|---|
| `gate_mode` | `int8` | BF16 Streaming より Gate 転送・Gate Linear が高速 / Faster gate transfer and gate linear than BF16 streaming |
| `keep_percent` | `5.0` | 5/7.5/10% sweep では5%が最速 / Fastest in the tested 5/7.5/10% sweep |
| `start_percent` | `0.0` | 全stepでVSA適用 / Apply VSA from the start |
| `end_percent` | `1.0` | 全stepでVSA適用 / Apply VSA through the end |
| `min_tokens` | `12288` | これ未満は sparse path の対象外 / Below this threshold the sparse path is not used |
| `skip_padding_qkv` | `False` | True は QKV 約11%、Sampler全体約1%遅化 / True was ~11% slower for QKV and ~1% slower overall |
| `ChunkFFN chunks` | `4` | MATLOW + Streaming v1 の採用値 / Tested v1 setting |
| `FastVAE tile_batch_size` | `2` | 12GB正式workflowの採用値 / Used in the formal 12GB workflow |
| `--disable-comfy-compiler` | recommended for tested stack | H3 Sparse Attention で大幅な速度低下を実測したため / Recommended because a large slowdown was observed with this H3 sparse stack |

### `sink_conditioning` について / About `sink_conditioning`

**v1 の重要な実装上の注意です。**

現在の Streaming H3 VSA path は、`h3_streaming_vsa_attention()` 内で

```python
sink = sink_q = (0, plan["n_prefix"])
```

を直接使用します。

そのため、UI の `sink_conditioning` 値は `SparseAttnPatch` に保存されますが、**現在の Streaming VSA H3 path の sink 範囲を変更しません**。v1 では互換性用の入力として扱ってください。非video prefix tiles は常に exact sink / dense-query sink として保護されます。

**Important v1 implementation note:**

The current Streaming H3 VSA path directly uses:

```python
sink = sink_q = (0, plan["n_prefix"])
```

Therefore, although the `sink_conditioning` widget is stored in `SparseAttnPatch`, it **does not change the sink range of the current Streaming VSA H3 path**. Treat it as a compatibility field in v1. All non-video prefix tiles are kept as exact sink / dense-query sink tiles.

---

## 7. インストール / Installation

### 日本語

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-KJNodes.git
git clone https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE.git
```

本リポジトリを:

```text
ComfyUI/custom_nodes/ComfyUI-Ref2VA-VSA-StreamingGate/
```

へ配置してください。

推奨起動例:

```bash
python main.py --disable-comfy-compiler
```

正式ベンチマークでは `--disable-pinned-memory` も併用しましたが必須ではありません。

Workflow:

- GUI: `workflows/Streaming_VSA_INT8_example.json`
- API: `workflows/Streaming_VSA_INT8_example.api.json`

3つの Reference Image を自分の画像へ差し替え、`gate_file` に `fasth3_vsa_gate.safetensors` を選択してください。

### English

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-KJNodes.git
git clone https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE.git
```

Place this repository at:

```text
ComfyUI/custom_nodes/ComfyUI-Ref2VA-VSA-StreamingGate/
```

Recommended launch example:

```bash
python main.py --disable-comfy-compiler
```

The formal benchmark also used `--disable-pinned-memory`, but it is not required by the node.

Workflow files:

- GUI: `workflows/Streaming_VSA_INT8_example.json`
- API: `workflows/Streaming_VSA_INT8_example.api.json`

Replace the three reference images with your own and select `fasth3_vsa_gate.safetensors` in `gate_file`.

---

## 8. Sparse動作の確認 / Verifying the Sparse Path

### 日本語

生成が完走しただけでは、すべての block が sparse path を通った証明にはなりません。

`h3_eligible()` が False の場合、該当 block は original dense block へフォールバックします。`verbose=True` で ComfyUI console に以下のようなログが出ることを確認してください。

```text
sparse producer path: ...
[streaming gate block ... mode=int8]
```

Sparse path の主な条件:

- CUDA
- BF16 H3 activation
- `head_dim == 128`
- `comfy_kitchen.sol_attn` kernel が利用可能
- token数が `min_tokens` 以上
- `start_percent` / `end_percent` の範囲内
- MiniMax H3 layout が取得できる

### English

A completed generation does not by itself prove that every block used the sparse path.

If `h3_eligible()` returns False, that block falls back to the original dense block. With `verbose=True`, verify console messages such as:

```text
sparse producer path: ...
[streaming gate block ... mode=int8]
```

Main sparse-path requirements:

- CUDA
- BF16 H3 activations
- `head_dim == 128`
- a usable `comfy_kitchen.sol_attn` kernel
- token count at or above `min_tokens`
- sigma inside the `start_percent` / `end_percent` window
- valid MiniMax H3 layout metadata

---

## 作者・関連リポジトリ / Author and related repositories

**日本語:** かみもと（Kamimoto）です。RTX 4070 12GBで、MiniMax H3とComfyUIの高速化・省VRAM化を実測しながら公開しています。

**English:** I'm Kamimoto (かみもと). I measure and publish MiniMax H3 and ComfyUI speed-ups and VRAM reductions on an RTX 4070 12 GB.

- X：[@sep_is_heim](https://x.com/sep_is_heim) — 検証速報 / quick results with step counts, seconds and speed-ups
- note：[かみもと｜ローカルAI生成](https://note.com/sepiablue) — 解説記事 / write-ups in Japanese（[記事INDEX / Article index](https://note.com/sepiablue/n/n8e612c619a3e)）
- [さやかベンチマーク / Sayaka Benchmark](https://sayakabenchmark.pages.dev/) — H3ワークフローの速度・画質比較 / speed and quality comparison of H3 workflows

| リポジトリ / Repository | 内容 / What it does |
|---|---|
| [ComfyUI-MiniMax-H3-W4A4-VSA](https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-W4A4-VSA) | Ref2VAの高速化（FC1 W4A4、Streaming VSA） / Ref2VA acceleration with FC1 W4A4 and Streaming VSA |
| [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) | X2 VAEのINT8化と非同期NVENC保存。1088×1920の音声付き5秒動画を約60秒で生成 / INT8 X2 VAE and asynchronous NVENC saving; 5 s 1088×1920 video with audio in about 60 s |
| [ComfyUI-H3-LBH-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-LBH-X2-Stream) | 低解像度3step＋高解像度1step。14秒動画で約25%短縮 / 3 low-resolution steps + 1 high-resolution step; about 25% faster for 14 s video |
| [ComfyUI-H3-AudioFirst-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-AudioFirst-X2-Stream) | 音声を先に作って固定し、金属的な響きを抑える2パス生成 / Audio-first two-pass generation that reduces the metallic voice |

---

## 9. ライセンス・帰属 / License and Attribution

### 日本語

**本リポジトリのソースコードは GPL-3.0 で公開します。**

この実装は ComfyUI core の H3 sparse-attention / chunked producer 構造を適応しているため、配布コード全体は GPL-3.0 として扱います。

主な第三者コンポーネント:

- ComfyUI core: GPL-3.0
- Kablex/ComfyUI-Ref2VA-VSA: Apache License 2.0
- barelymining のカスタムノードコード: MIT License
- comfy-kitchen: Apache License 2.0
- FastH3 / VSA Gate 重み: MiniMax H3 Community License

Apache-2.0 / MIT の第三者コード・アイデアに関する元の著作権表示と帰属は維持します。  
モデル・VSA Gate・VAE・Text Encoder などの重みは本リポジトリには同梱せず、各配布元のライセンスに従って利用してください。

詳細は `LICENSE`、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[LICENSE_NOTES.md](LICENSE_NOTES.md) を参照してください。

### English

**The source code in this repository is distributed under GPL-3.0.**

Because this implementation adapts the H3 sparse-attention / chunked-producer structure from ComfyUI core, the distributed source code is treated as GPL-3.0.

Main third-party components:

- ComfyUI core: GPL-3.0
- Kablex/ComfyUI-Ref2VA-VSA: Apache License 2.0
- barelymining custom-node code: MIT License
- comfy-kitchen: Apache License 2.0
- FastH3 / VSA gate weights: MiniMax H3 Community License

Original copyright and attribution notices for Apache-2.0 / MIT third-party material must be retained.  
Model weights, VSA gate weights, VAEs, and text encoders are not bundled in this repository and remain subject to their respective upstream licenses.

See `LICENSE`, [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [LICENSE_NOTES.md](LICENSE_NOTES.md) for details.

---

## 10. ドキュメント / Documentation

- [BENCHMARK.md](docs/BENCHMARK.md)
- [IMPLEMENTATION.md](docs/IMPLEMENTATION.md)
- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- [LICENSE_NOTES.md](LICENSE_NOTES.md)
