# Streaming VSA INT8 v1 ベンチマーク / Benchmark

MiniMax H3 Ref2VA における Streaming VSA INT8 の正式比較と開発時プロファイルを記録します。  
Formal comparison and development profiling notes for Streaming VSA INT8 on MiniMax H3 Ref2VA.

---

## 1. 正式4手法比較 / Formal Four-Workflow Comparison

### 日本語: 共通条件

- ComfyUI 環境: `sage_clean_setup`
- GPU: NVIDIA GeForce RTX 4070 12GB
- OS: Windows 11
- Python: 3.13.14
- PyTorch: 2.13.0+cu130
- comfy-kitchen: 0.2.33
- 解像度: 1024×1792
- フレーム数: 124
- seed: 43
- Prompt: PROMPT-08
- Sigma Shift: video 12 / audio 3
- FastVAE: `tile_batch_size = 2`
- 測定時起動引数: `--disable-pinned-memory --disable-comfy-compiler`
- 同一 ComfyUI 環境内で正式 workflow を実行

### English: Common conditions

- ComfyUI environment: `sage_clean_setup`
- GPU: NVIDIA GeForce RTX 4070 12GB
- OS: Windows 11
- Python: 3.13.14
- PyTorch: 2.13.0+cu130
- comfy-kitchen: 0.2.33
- Resolution: 1024×1792
- Frames: 124
- Seed: 43
- Prompt: PROMPT-08
- Sigma Shift: video 12 / audio 3
- FastVAE: `tile_batch_size = 2`
- Launch options: `--disable-pinned-memory --disable-comfy-compiler`
- All formal workflows were run in the same ComfyUI environment

### 方式ごとの条件 / Per-workflow configuration

| No. | Method | Checkpoint / Input | Sparse configuration | Sampler / Sigma | FFN |
|---:|---|---|---|---|---:|
| 01 | MATLOW Fused SLA95 | `minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors`, Ref3 | Core `BlockSparseAttention`, `top-k (SLA)`, keep 5%, start 0, end 1 | `res_multistep` + `simple`, 4 steps | 4 |
| 02 | Streaming VSA INT8 | Same MATLOW checkpoint, Ref3 | `Ref2VAStreamingVSAPatch`, gate INT8, keep 5%, start 0, end 1 | `res_multistep` + `simple`, 4 steps | 4 |
| 03 | FastH3 VSA95, no Ref | `minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors`, no reference image | Core `BlockSparseAttention`, `VSA (FastVideo)`, keep 5%, start 0, end 1 | Euler + ManualSigmas | 2 |
| 04 | FastH3 VSA95 + Ref3 | Same FastH3 checkpoint, Ref3 | Core `BlockSparseAttention`, `VSA (FastVideo)`, keep 5%, start 0, end 1 | Euler + ManualSigmas | 2 |

FastH3 ManualSigmas:

```text
0.9999166, 0.9728326, 0.9230769, 0.8, 0.0
```

### 日本語: 比較上の注意

01 / 02 / 04 は Ref 3枚を投入していますが、**01/02 と 04 は checkpoint、Sampler/Sigma、ChunkFFN が同一ではありません**。したがって、これは各方式の実用 workflow 比較であり、Attention backend だけの純粋な A/B テストではありません。

04 は FastH3 checkpoint に Ref3 入力を与えた **負荷比較用の参考 workflow** です。FastH3 の Ref2VA 品質を正式サポートするもの、または推奨生成経路として扱いません。

03 は参照画像なしの FastH3 基準です。Ref2VA と直接速度比較すると token 負荷が揃わないため、03 と 04 の差を合わせて確認します。

### English: Comparison caveats

01 / 02 / 04 all use three reference images, but **01/02 and 04 do not use the same checkpoint, sampler/sigma schedule, or ChunkFFN configuration**. This is therefore a practical workflow comparison, not a pure attention-backend A/B test.

04 is a **reference load-comparison workflow** that feeds Ref3 into the FastH3 checkpoint. It should not be presented as an officially supported or recommended FastH3 Ref2VA quality path.

03 is the no-reference FastH3 baseline. Because its token load is lower than Ref2VA, compare 03 and 04 together before interpreting speed differences against Ref2VA workflows.

---

## 2. 正式結果 / Formal Results

| No. | Method | Input | Native / padded tokens | Sampler | Step avg. | FastVAE | Prompt total |
|---:|---|---|---:|---:|---:|---:|---:|
| 01 | MATLOW Fused SLA95 | Ref3 | 69,560 / 75,008 | **142s** | **35.54 s/it** | 39.58s | 214.71s |
| 02 | Streaming VSA INT8 | Ref3 | 69,560 / 75,008 | **165s** | **41.47 s/it** | 40.75s | 231.88s |
| 03 | FastH3 VSA95 | no Ref | 67,232 / 72,704 | **142s** | **35.63 s/it** | 42.12s | 212.75s |
| 04 | FastH3 VSA95 + Ref3 | Ref3 | 69,560 / 75,008 | **167s** | **41.94 s/it** | 41.54s | 242.64s |

### 日本語: 解釈

- Ref3 の実用 workflow 比較では **MATLOW SLA95 が最速**でした。
- Streaming VSA INT8 と FastH3 + Ref3 は **165s vs 167s** でほぼ同じ速度帯です。2秒の単発差を方式の本質的な優位差とは扱いません。
- FastH3 の参照画像なし→Ref3 では Sampler が **142s → 167s（+25s）**。
- token 数は native **67,232 → 69,560（+2,328）**、padded **72,704 → 75,008（+2,304）**。
- したがって、FastH3 no-Ref だけを Ref2VA の速度と直接比較するのは不公平です。
- 正式結果表には、同じ測定ログで確認できなかった Peak VRAM 値を載せていません。VRAM値を掲載する場合は、同じ正式測定条件で別途採取したログを添えてください。

### English: Interpretation

- In the practical Ref3 workflow comparison, **MATLOW SLA95 was the fastest**.
- Streaming VSA INT8 and FastH3 + Ref3 were in almost the same speed range at **165s vs 167s**. A one-run two-second difference should not be treated as an intrinsic method advantage.
- For FastH3, adding Ref3 changed sampler time from **142s to 167s (+25s)**.
- Native tokens increased from **67,232 to 69,560 (+2,328)**, and padded tokens from **72,704 to 75,008 (+2,304)**.
- Therefore, comparing the FastH3 no-reference result directly against Ref2VA speed is not input-balanced.
- Peak-VRAM values are intentionally omitted because they were not part of the same formal measurement record used for the timing table. Add them only if captured under the same formal conditions.

---

## 3. 追試の速度帯 / Follow-up Timing Range

### 日本語

複数 Prompt の追試では、Ref3 を使用する 01 / 02 / 04 の Sampler は概ね:

```text
01 MATLOW SLA95          : 142–146s
02 Streaming VSA INT8   : 165–169s
04 FastH3 VSA95 + Ref3  : 167–169s
```

の範囲でした。

### English

Across additional prompts, the Ref3 workflows were approximately:

```text
01 MATLOW SLA95          : 142–146s
02 Streaming VSA INT8   : 165–169s
04 FastH3 VSA95 + Ref3  : 167–169s
```

These are follow-up ranges, not replacements for the formal PROMPT-08 result table above.

---

## 4. 生成品質・音声 / Visual and Audio Quality

### 日本語

初期の PROMPT-C / 832×1408 では、VSA系の音声が SLA より聞き取りやすいと感じた主観評価がありました。

その後の正式比較および追試では、**方式間の明確な音質差・画質差は再現しませんでした**。

v1 では以下を一般的な利点として主張しません。

- Streaming VSA の音質優位
- Streaming VSA の画質優位

### English

In an early PROMPT-C / 832×1408 test, the VSA outputs were subjectively perceived as easier to hear than SLA.

That difference was **not reproduced clearly** in the later formal comparison and follow-up tests.

v1 therefore does not claim:

- superior audio quality from Streaming VSA
- superior visual quality from Streaming VSA

---

## 5. Gate BF16 → INT8 内部実測 / Internal Gate BF16 → INT8 Measurement

| Measurement | BF16 Streaming | INT8 Streaming |
|---|---:|---:|
| Gate processing / step | ~7.34s | ~1.41s |
| Gate transfer size / block | ~73.5 MiB | ~36.8 MiB |

### 日本語

Gate処理部分では約 **5.2倍**の高速化です。Sampler全体が5.2倍になるという意味ではありません。

### English

This is about a **5.2× speedup in the gate-processing portion only**. It does not mean the full sampler becomes 5.2× faster.

---

## 6. 832×1408 開発時プロファイル / 832×1408 Development Profile

| Component | Time / step |
|---|---:|
| FFN / MLP | 9.60s |
| SOL Attention kernel + coarse residual | 7.89s |
| QKV INT8 GEMM | 3.52s |
| Gate INT8 Linear | 1.41s |
| Attention Out Projection | 1.30s |
| AdaLN / Norm / Modulate | 1.11s |
| Gate H2D + Unpermute etc. | 0.74s |
| Sampler outer loop | ~1.5s |

### 日本語

このプロファイルは 832×1408 の開発計測で、1024×1792 の正式比較とは条件が異なります。

### English

This profile was captured during 832×1408 development testing and is not directly interchangeable with the 1024×1792 formal benchmark.

---

## 7. `skip_padding_qkv` 実験 / `skip_padding_qkv` Experiment

### 日本語

832×1408 にて:

- Latent Max Absolute Difference: `0.00000000e+00`（Bit-Exact）
- QKV: 3.52s → 3.91s（約11%悪化）
- Sampler: 108s → 109s（約1%悪化）

4096整列が崩れること、および gather/scatter/buffer overhead が削減演算量を上回ったため:

```text
skip_padding_qkv = False
```

を v1 推奨値とします。

### English

At 832×1408:

- Latent Max Absolute Difference: `0.00000000e+00` (bit-exact)
- QKV: 3.52s → 3.91s (~11% slower)
- Sampler: 108s → 109s (~1% slower)

Breaking the 4096-aligned workload and adding gather/scatter/buffer overhead outweighed the skipped math, so v1 recommends:

```text
skip_padding_qkv = False
```

---

## 8. 参照 / References

- Kablex/ComfyUI-Ref2VA-VSA: https://github.com/Kablex/ComfyUI-Ref2VA-VSA
- ComfyUI PR #16072: https://github.com/Comfy-Org/ComfyUI/pull/16072
- Gate distribution: https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo
- MATLOW Fused: https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot
