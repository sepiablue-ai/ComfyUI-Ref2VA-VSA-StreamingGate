# Third-Party Notices and Attribution

本プロジェクト（`ComfyUI-Ref2VA-VSA-StreamingGate`）は、以下のオープンソースプロジェクトおよび研究成果を参考・利用・派生して開発されています。

---

## 1. Kablex/ComfyUI-Ref2VA-VSA

- **Repository**: [https://github.com/Kablex/ComfyUI-Ref2VA-VSA](https://github.com/Kablex/ComfyUI-Ref2VA-VSA)
- **Author**: Kablex
- **License**: Apache License 2.0
- **利用・参考部分**:
  - `SparseAttnPatch`、VSA のレイアウト計画（`vsa_plan`）、4096 トークン QKV producer パス、および ComfyUI モデルパッチの基本構造。
- **改変点**:
  - 3.59GiB (約3.85GB) の Gate 重みを GPU に常駐させず、起動時に CPU 上で動的 INT8 量子化して Pinned Host Memory に保持。
  - 各 DiT ブロックの Attention 実行時にブロック単位でストリーミング転送し、直後に GPU メモリを解放する機構（Streaming Gate）を新規設計・実装。
  - `comfy_kitchen.int8_linear` による INT8 Gate 線形計算の統合。
- **再配布条件**:
  - Apache 2.0 の条項に基づき、ライセンス表示および改変事実の告知を記載。

---

## 2. FastVideo/FastVideo

- **Repository**: [https://github.com/FastVideo/FastVideo](https://github.com/FastVideo/FastVideo)
- **License**: Apache License 2.0
- **関係性**:
  - Visual Sparse Attention (VSA) の基本概念および実装リファレンス。

---

## 3. barelymining/ComfyUI-MiniMax-H3-FastVideo

- **Repository**: [https://github.com/barelymining/ComfyUI-MiniMax-H3-FastVideo](https://github.com/barelymining/ComfyUI-MiniMax-H3-FastVideo)
- **Node / Code License**: MIT License
- **関係性**:
  - FastH3 ノード実装および Gate 重みファイルの配布元。

---

## 4. fasth3_vsa_gate.safetensors (Model Weight)

- **Distribution**: HuggingFace [barelymining/ComfyUI-MiniMax-H3-FastVideo](https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo)
- **License**: **MiniMax H3 Community License**
- **Weight 同梱方針**:
  - 本リポジトリにはモデル重みを一切同梱しません。利用者が元配布元より直接取得してください。

---

## 5. comfy-kitchen

- **Repository**: [https://github.com/Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)
- **License**: Apache License 2.0
- **利用部分**:
  - `sol_attn_chunked` (CUDA スパース Attention カーネル)
  - `int8_linear` (CUDA CUTLASS INT8 GEMM & Hadamard ConvRot 量子化カーネル)

---

## 6. ComfyUI

- **Repository**: [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- **License**: GNU General Public License v3.0 (GPL-3.0)
- **関係性**:
  - ComfyUI 本体は GPL-3.0 です。
  - 本カスタムノード（`ComfyUI-Ref2VA-VSA-StreamingGate`）は Apache-2.0 で配布されています。
  - 本ソフトウェアは ComfyUI 上で動作するサードパーティ製カスタムノードです。
