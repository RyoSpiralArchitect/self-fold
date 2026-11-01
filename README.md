# SELF-FOLD

**Self Without a Transcript: Semantic Recurrence Across Text and Latent States**

自己記述の意味を折り返したとき、言語化は有用な区別を作るのか、意味を固定するのか。その変化は、その後の未見課題にも作用するのか。

学習済みの **Mistral-7B-Instruct-v0.3** で、同じ2つの初期記述を使い、EOSまで待つ条件と8トークンで区切る条件を比較しました。前者は18生成すべてが自然に終了しました。Tは初期記述に応じて異なる文を出し、Bはどちらの記述からも同じ13トークンの文を出しました。言語化後のBの保存状態は、各条件内で4更新すべてが一致しました。未校正の潜在経路で行った4試行の記録です。[実験条件・生成例・生データ](provenance/mistral_eos_20261101/RESULTS.md)

最初の公開実験では、2つの初期記述と2通りの更新手順を組み合わせ、CPUで36回の状態遷移を記録しました。LとBは4回とも同じ潜在状態から始まり、最初の更新でも一致しました。その後、Lの12個の更新後状態はすべて異なり、言語化と再符号化を挟んだBの12個はすべて同一でした。ランダム初期化した小型デコーダーで、Bの生成は全回8トークンの上限に達しています。学習済みモデルにおける意味の保持や推論への影響は未検証です。[初回の結果と生データ](provenance/launch_cpu_20261101/RESULTS.md) · [実験条件と再現コマンド](provenance/launch_cpu_20261101/PLAN.md)

## 最初に読む

- [Codex向け framing-handoff](docs/FRAMING_HANDOFF.md): 研究の芯、変更してはいけない比較、次の手順。
- [実装状況](docs/IMPLEMENTATION_STATUS.md) と [実験プロトコル](docs/PROTOCOL.md)。
- [Furnace runbook](docs/FURNACE_RUNBOOK.md) と [意味ゲート](docs/QUALIFICATION.md)。
- [薄い論文草稿](paper/main.md)、[LaTeX原稿](paper/main.tex)、[BibTeX](paper/references.bib)、[引用監査](paper/REFERENCE_AUDIT.md)。

## 三条件

| 条件 | 継承するもの | 更新 |
|---|---|---|
| T | 直前の自己記述だけ | `T(current_text)` |
| L | 固定長のlatent slotsだけ | `G(current_slots)` |
| B | 同じlatent slots | `E(D(G(current_slots)))` |

L/Bは同じ `G` と同じ初期 `E(s0)` を使います。Bだけが更新ごとに言語化と再符号化を挟みます。主比較はL対Bです。T対Lは異なる系全体の比較です。普通の更新と `decompose → boundary → reintegrate` の固定演算順序を全条件に適用できます。

ログは実験者側だけに残ります。KV cache、過去の発話、観測用の文章、評価課題の答えを再帰の本流に戻しません。意味の新奇さは強制せず、妥当な収束は許容します。

## すぐ動くCPU確認

既存のPython環境を無断で変更しないでください。PyTorchを使える環境で、プロジェクトrootから:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
bash scripts/run_smoke.sh runs/my-first-tiny-smoke
python -m selffold plan
```

`tiny` は本物のforward/hookを通るランダム初期化デコーダーです。事前学習モデルでも、意味理解の模擬証拠でもありません。表示される `ENGINEERING_PASS_NOT_SEMANTICS` を品質の成功に読み替えないでください。

ネットワークなしの既存環境では `PYTHONPATH=src python -m pytest -q` でも実行できます。Transformersがなければ4系統の任意試験はskipになります。CPU基本試験と別に、Furnaceでは `python -m pytest -q -m hf` が**4件すべて実行されること**を確認してください。

## 学習済みモデルへの入口

```bash
python -m selffold inventory --out runs/inventory-new.json
# 既存のローカルcheckpointだけを指定する。パスは環境から確認する。
bash scripts/furnace_smoke.sh configs/models/mistral7b.yaml \
  "$MISTRAL_LOCAL_CHECKPOINT" runs/mistral-canary-001
```

各候補を一プロセスずつ検証します。モデルの自動取得、remote codeの自動許可、量子化の自動切替、環境の自動アップグレードはありません。

学習済みモデルの生成は、既定で各モデルのEOSまで続きます。`--until-eos` では全条件に同じコンテキスト上限を適用し、入力を除いた残りの枠を使います。固定長との比較には `--max-new-tokens N` を使えます。上限に達した出力も、空でなければ打ち切りの印を付けて次の更新に使います。小型CPUデコーダーは従来の8トークン設定を保ち、実装上EOSを抑制しています。

| 候補profile | 学習済みcheckpoint候補 | 位置づけ |
|---|---|---|
| `mistral7b` | `mistralai/Mistral-7B-Instruct-v0.3` | 既存研究との接点 |
| `llama3b` | `meta-llama/Llama-3.2-3B-Instruct` | 小型dense比較 |
| `gemma2b` | `google/gemma-2-2b-it` | 小型text-only比較 |
| `phi_mini_moe` | `microsoft/Phi-mini-MoE-instruct` | MoE比較候補 |
| `phi_tiny_moe` | `microsoft/Phi-tiny-MoE-instruct` | MoEの軽量な代替候補 |

Mistralのプロファイルには今回使用したチェックポイントのrevisionと内容ハッシュを記載しています。各試行でも、実際に読み込むローカルファイルをハッシュで同定します。各モデルの出典とメモリ条件はrunbookにあります。Phi-tinyでは標準Phimoeとのattention幅の不一致を検査します。レビュー済みのローカル独自実装を指定する経路は、実モデルでは未検証です。

**latent本実験には追加の意味ゲートが必要です。** `--allow-uncalibrated` が許可するのはsmokeの配線確認であって、意味再帰の成立ではありません。writer出力を次の入力境界へ運ぶ学習・校正や読出し調整が必要になる可能性があります。この作業はまだ完了していません。

## 出力と再現

各runは新規directoryを要求します。source/model/template/transport hash、環境、未加工の遷移、state、活性、課題得点、停止理由が残ります。`load_state` はpickleを使わずNPZ+JSONからdtypeとhashを検証して復元します。読出しの乱数を本流へ持ち越しません。失敗runは保存し、自動retryしません。

`data/*.dev.jsonl` は48個の初期記述と20個の人工課題です。すべて開発用で、確認用holdoutはまだ作っていません。実装確認に使った課題を後から未知評価と呼ばないでください。

## リポジトリとライセンス

```bash
git clone https://github.com/RyoSpiralArchitect/self-fold.git
cd self-fold
```

このリポジトリのコード、文書、同梱の合成開発データにはApache License 2.0を適用します。モデルの重みとトークナイザーには各配布元の条件が適用されます。[LICENSE](LICENSE) · [出典と適用範囲](docs/LICENSING.md)
