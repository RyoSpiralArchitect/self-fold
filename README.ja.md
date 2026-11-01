# SELF-FOLD

[English](README.md) · 日本語

**モデルの内部状態をいったん言葉にし、その言葉を読み直すと、何が残るのか。** SELF-FOLDでは、重みを固定した言語モデルが短い自己記述を更新する過程を追います。次の更新に渡すのは、その時点の文章か、決まった数の潜在スロットだけです。過去の発話は実験者側に記録します。

| 条件 | 次のステップに渡すもの |
| --- | --- |
| T・文章 | モデルが新しく書いた自己記述。 |
| L・潜在状態 | 内部で1回更新した8個のスロット `G(Z)`。 |
| B・言語化往復 | Lと同じ更新を行い、その結果を文章にしてもう一度符号化したスロット `E(D(G(Z)))`。 |

LとBは同じ初期状態、更新演算、層の境界、生成の停止規則を使います。この特定の言語化往復で何が変わり、その変化が後の課題にどう響くかを調べています。[論文草稿](paper/main.md) · [実験プロトコル](docs/PROTOCOL.md)

## 実験

- **Bが同じ状態に戻るのは、生成文が短いためか。** Mistral-7B-Instruct-v0.3で、対になる2つの初期記述を2回ずつ更新しました。EOSを待つ条件では全18生成が自然に終了しましたが、Bはどちらの記述からも同じ13トークンの文を返しました。Bの更新後の状態は停止規則ごとに1種類、Lはそれぞれ4種類でした。[条件・生成例・状態ハッシュ](provenance/mistral_eos_20261101/RESULTS.md)
- **CPUでの動作確認。** ランダム重みの小型デコーダーで4試行を行い、初期状態の一致、保存と復元、計算量、T/L/Bの更新経路を確認しました。[CPU実験の記録](provenance/launch_cpu_20261101/RESULTS.md)

## 再現手順

リポジトリをクローンし、Python 3.11以降で以下を実行します。

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hf]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
bash scripts/run_smoke.sh runs/tiny-001
~~~

試行ごとに新しい出力先を指定します。実験記録にはソースコミット、モデルの識別情報、採点前の生スコア、生成の停止理由を残しています。[学習済みモデルでの再現とファイルの照合](docs/REPRODUCING.md) · [モデルと状態の実装](docs/ARCHITECTURE.md) · [意味を検証する基準](docs/QUALIFICATION.md)

このプロジェクトのコード、文書、合成データにはApache License 2.0を適用します。モデルの重みとトークナイザーには配布元の条件が適用されます。[ライセンス](LICENSE) · [出典](docs/LICENSING.md)
