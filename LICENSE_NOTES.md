# ライセンスおよびモデル重みの取り扱い方針 (License Notes)

## 1. ソフトウェアライセンス構成

- **本カスタムノード**:
  - 本リポジトリのソースコード（`nodes.py`、`__init__.py`、ワークフロー JSON、ドキュメント類）は、派生元である [ComfyUI-Ref2VA-VSA](https://github.com/Kablex/ComfyUI-Ref2VA-VSA) と同様に **Apache License 2.0** で配布されます。
- **ComfyUI 本体との関係**:
  - ComfyUI 本体は **GNU General Public License v3.0 (GPL-3.0)** です。
  - 本カスタムノード（`ComfyUI-Ref2VA-VSA-StreamingGate`）は **Apache-2.0** で配布されています。
  - 本ソフトウェアは ComfyUI 上で動作するサードパーティ製カスタムノードです。
- **comfy-kitchen**:
  - 本ノードが依存する `comfy-kitchen` ライブラリは **Apache License 2.0** です。
- 詳細な著作権・ライセンス表示については [LICENSE](LICENSE) および [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) をご参照ください。

---

## 2. モデル重み・チェックポイントの取り扱い（同梱方針）

### (1) モデル重みの非同梱
本配布パッケージには、以下のモデル重み・チェックポイント・ファイルは意図的に同梱していません。
- MiniMax H3 ベースチェックポイント / MATLOW Fused モデル本体
- VSA Gate 重みファイル (`fasth3_vsa_gate.safetensors`)
- VAE モデル (`minimax_h3_video_vae_int8_convrot.safetensors` 等)
- テキストエンコーダー (`qwen3vl_32b_...`)

### (2) 非同梱方針と適用ライセンス
- FastH3 Gate / LoRA 重み (`fasth3_vsa_gate.safetensors`) は **MiniMax H3 Community License** の対象です。
- 本リポジトリでは、大容量モデル重みとソースコードの管理を明確に分離するため、weight を意図的に同梱していません。

### (3) 利用者による取得
- 各モデル重みは、元配布先（HuggingFace 等）の利用規約（MiniMax H3 Community License 等）をご確認の上、利用者ご自身の手で直接取得して所定のフォルダへ配置してください。
- 元配布元 URL および配置先は [README.md](README.md) に記載されています。
