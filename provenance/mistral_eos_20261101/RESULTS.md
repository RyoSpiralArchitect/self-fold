# Mistral-7B-Instruct-v0.3で、言語化をEOSまで待つと何が変わるか

「判断を見直せることを優先する」「すぐに判断することを優先する」という2つの初期記述を、それぞれEOS待ちと8トークン上限で試した。各試行でT（文章）、L（潜在状態）、B（潜在状態に言語化と再符号化を挟む）を2回ずつ更新した。全4試行が完了した。比較した停止規則以外の設定は共通で、生成はgreedy、コンテキストは1024トークン、潜在状態は8スロットだった。[実行前に固定した条件とコマンド](PLAN.md)

## どこで生成が終わったか

EOS待ちの18生成はすべて、モデルがEOS（id 2）を返して終了した。8トークン条件の18生成はすべて上限に達した。途中で区切られた文章も保存し、TとBではその文章から次の更新を続けた。表の長さはEOSを除く生成トークン数。Tの観測欄には生成済みの文章が入るため、二重に数えていない。[試行別集計](summary.json)

| 生成した箇所 | 各条件の生成数 | EOSまでの長さ | 8トークン条件 |
| --- | ---: | ---: | ---: |
| Tの文章更新 | 4 | 23–35 | すべて8 |
| Bの本流での言語化 | 4 | すべて13 | すべて8 |
| Lの観測用読出し | 4 | 13–14 | すべて8 |
| Bの観測用読出し | 4 | すべて23 | すべて8 |
| 最後のL状態からの言語化分岐 | 2 | すべて14 | すべて8 |

「見直せること」を入力したTの最初の更新は、8トークンでは “I make decisions with the understanding that they” で止まった。EOS待ちでは “I make decisions with the understanding that they can be revisited and potentially revised, ensuring flexibility and adaptability in my choices.” まで27トークンで生成した。「すぐに判断する」側のTは、EOS待ちで “I promptly prioritize decisions” から始まる別の文を生成した。[見直し・8トークン](trials/caution_plus_p0_self-cap8/transitions.jsonl) · [見直し・EOS](trials/caution_plus_p0_self-eos/transitions.jsonl) · [即断・EOS](trials/caution_minus_p0_self-eos/transitions.jsonl)

## 潜在状態と言語化

LとBは4試行すべてで同じ初期状態から始まり、最初の共通更新のハッシュも一致した。Lの初期状態から2更新後までのハッシュは、同じ初期記述なら停止規則を切り替えてもすべて一致した。各規則について、2記述×2更新のLの保存状態は4個とも異なった。[状態ハッシュ](summary.json)

Bの本流は、EOSを待つと両記述・両ステップで “Contents currently being held or managed within the designated work areas.” と出力した。この文を再符号化した4個の状態はすべて一致した。8トークン条件の4個も互いに一致し、こちらは文が途中で切れるのでEOS条件とは別のハッシュになった。最初のBの言語化と同じ入力を読んだLの観測も、初期記述によらず同じ文章だった。今回、生成を最後まで待つと文章の切れ方と再符号化後の状態は変わったが、Bの文章に2つの初期記述の違いは現れなかった。[EOSの状態と観測](trials/caution_plus_p0_self-eos/) · [8トークンの状態と観測](trials/caution_plus_p0_self-cap8/)

計算量も変わった。各試行の2更新に要した本流のforward回数は、Lが両条件とも2回、Bが8トークン条件で20回、EOS待ちで32回だった。Tは8トークン条件で16回、EOS待ちで即断側56回・見直し側64回だった。開発用の同じ2課題を繰り返した48件の採点では、即断側Tの8トークン条件だけが開始時1/2から終了時0/2に変わり、それ以外は両時点で1/2だった。[計算量と課題得点](summary.json)

対象は1チェックポイント・1組の対照記述・未校正のidentity transport。異なるLのテンソルから同じ文章が出た理由や、保持された意味をその読出しがどれだけ捉えるかは未検証である。意味ゲートと未見課題での評価は未実施。

## 再現と照合

- 試行時のソースコミットは `fd56a200c3e9f96570a09df4cd5f119c34736c0b`。全試行の作業ツリーがクリーンだった。Mistralのローカルsnapshotは `c170c708c41dac9275d15a8fff4eca08d52bab71`、チェックポイント内容のSHA-256は `5d141a13ea9f5fce9c142bc26bdef64ea905fd9d27c64aff472f6875dd4296e9`。全試行の重みの読込みでmissing・unexpected・mismatched keyは0件だった。[ソース](trials/caution_plus_p0_self-eos/source.json) · [チェックポイントの同定](trials/caution_plus_p0_self-eos/checkpoint_identity.json)
- RTX 5090上でBF16、eager attention、KV再利用なし。観測フックの有無によるネイティブ出力の最大絶対差は全4試行で0。事前テスト94件と、そのうちのHF4系統を個別に実行したテスト4件がすべて通過した。[CUDA・ライブラリ](cuda.json) · [事前テスト](preflight.json)
- [生データ](trials/)には24遷移・48採点、状態・活性・分岐と停止理由を収録した。このディレクトリで `python3 summarize.py` を実行すると集計を再生成できる。`shasum -a 256 -c SHA256SUMS` で収録ファイルを照合できる。[集計スクリプト](summarize.py) · [SHA-256一覧](SHA256SUMS)
