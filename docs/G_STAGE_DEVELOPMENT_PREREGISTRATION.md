# G の更新・書き込み・読み出しを分離する開発実験仕様

**状態：`DESIGN_ONLY_NOT_EXECUTED`。実行マニフェストの凍結前。**

この文書は、PR #27 の結果を見た後に設計した追加測定の仕様である。既存結果の再集計は事後的な診断として扱い、この仕様を過去の結果の事前登録とは呼ばない。次の測定は、実装、入力の展開済み一覧、採点器、モデル・部品、予算を一つのマニフェストに束ね、その SHA256 と Git commit を**測定前に**記録してから開始する。この文書だけでは、その凍結は完了していない。

対象の基点は `ba20b3125ac498ef1f2fa21730eef3557e3dd953`。既存の [結果](../provenance/supported_g_20261104/RESULTS.md)、[一次計画](../provenance/supported_g_20261104/PLAN.md)、[読み手の継続学習計画](../provenance/supported_g_20261104/READOUT_PLAN.md) と独立した開発記録を追加する。公開済み preprint v1、既存の結果・判定・チェックサムは変更しない。独立資格試験の 24 群は読み込み、生成、照合、採点のいずれにも使わない。人による遷移評定は未実施のまま `pending`、正式判定は `NOT_QUALIFIED` を維持する。

## 1. この実験で区別するもの

調べたいのは、固定された読み手の正答率がどこで変わるかと、その変化をどの処理に帰属できるかである。次の三つの証拠を分ける。

| 証拠 | 言えること | それだけでは言えないこと |
| --- | --- | --- |
| バイト列・テンソルが変わる | 数値的な更新が行われた | 正しい意味的区別が新たに使えるようになった |
| 独立に定義した正しい行動表が得られる | この有限課題の帰結が正しく構成された | 任意の読み手、言語化、次の更新でも利用できる |
| 同じ問い・読み手で、対立する方策の両側が正しく切り替わる | その読み手が必要な区別を行動に利用した | 元の状態にその区別が存在しなかった、または G が新しい外部情報を獲得した |

native G の E と更新後状態のハッシュが違うことと、固定読み手の回答がすべて A に崩れていることは両立する。後者から「G は状態を更新していない」とは結論しない。逆に、異なるハッシュから意味の更新を認定しない。

構造化経路は、さらに次の境界を別々に扱う。

`source → 推定前提 → G の soft table → global argmax → 4 行の文章`

この同じ文章から、literal reader による直接回答と、文章の再符号化 E(D(G)) を介した回答へ**並列に分岐**する。literal reader の A/B 回答を再符号化するのではない。

この順序に対し native 経路は次のようになる。

`source → E の 8 slots → post_writer の W → transport T(W) → 実際の注入状態 → reader`

native D は更新後状態から別に分岐する。W を、そのまま pre-cut 状態として注入する対照は設けない。異なる層の座標系を同一視すると、書き込みの失敗と無効な注入の失敗が混ざるためである。[現行の更新処理](../src/selffold/host.py) と [raw 測定](../scripts/raw_supported_boundary.py) がこの分解の基準になる。

## 2. 対応付ける単位と、同じ条件の意味

一つの観測キーを次のように持つ。

```text
panel_id, source_bytes_sha256,
topic, trigger, veto, action_polarity, source_style, source_order,
world, answer_order, question_sha256,
route, reader_sha256, geometry_id, scoring_rule_sha256
```

`source_style=0` という整数だけで異なる実験の文面を結合しない。最初の文面と読み手継続後の文面では、同じ style 番号の文字列が違う。意味上の対応キーと、実際に同一のニューラル入力だったことを示すキーを別に保存する。

比較する六つの経路族は以下とする。identity、D、再符号化には carrier ごとの枝があり、一つの値に潰さない。

| 経路族 | 入力・出力 | 主な対照 |
| --- | --- | --- |
| identity | 構造化 E の未解決表／native E の 8 slots | 同じ source の G と比較 |
| 形式的な直接解釈 `F_inferred` | 同じ推定済み表・役割 bit・凍結 pointer を用い、問いの時点で形式的な veto を適用 | G と物理的に同じ rows を得るか |
| structured G | 固定された learned gates、pointer、bit による更新 | `F_inferred`、identity、gold-premise 介入 |
| native G | 実際の boundary writer と固定 transport | native identity、独立に書いた正しい successor の符号化 |
| D(G) | 構造化 renderer の文章／native 自由生成の文章 | D(E)、独立に作った正しい表の文章 |
| 再符号化後の読み出し | 実際に出力した D(G) だけから得る状態 | 元の G、D(G) 本文、対応する D(E) の往復 |

`F_gold` は正しい authored premises に形式規則を適用した**採点用・介入用の oracle**であり、同じ推定前提を使う `F_inferred` と分ける。oracle を G に与えた主結果として扱わない。

既存の analytic route は `resolve(policy, GUARD_TRUTH, actor)` であり、learned gate だけを Boolean veto に置き換える。source fields や通常行を取り出す pointer の誤りを、その baseline が自動的に修復するとは仮定しない。

### 既存の structured と native は、同じ読み出し課題ではない

現在の structured route は外部の world address で行を選び、[NativeChoice](../scripts/state_only_native.py) が「保存された行動はどちらの選択肢か」を読む。native route は世界条件を含む方策質問と 8 slots を受け取る。読み手の重みと注入位置も違う。Case が同じでも、この差を消した比較にはならない。

そこで、既存の条件を忠実に再集計する比較と、後述の **C：共通質問・交差読み手** を区別する。新しい共通条件の成績を、旧条件の再現値として上書きしない。共通条件でも、structured の外部行選択と embedding bank、native の cut-13 注入は経路の構成要素として残る。経路差全体と、G 単体の効果を混同しない。

## 3. 追加測定に先立つ固定事項

既存の saved states、dispatch、native scores、literal scores、free text とその hash から計算できるものは再集計する。既に測られた値を得るためだけの新しい学習・推論は行わない。追加実行は次の不足を埋める。

| 不足している観測 | 追加する測定 |
| --- | --- |
| soft row を global argmax の one-hot にした時の、同一読み手での応答 | S：soft／hard／oracle one-hot の対比較 |
| native writer と transport の中間テンソル、実際の注入テンソル | N：同じ source の E／W／T(W)／injected state の捕捉 |
| 同一質問・同一読み手での六経路の比較 | C：既存自由生成 subset での交差読み手試験 |

S・N・C の入力は既に公開・観察された開発例である。これは診断用の再測定であり、新規文面や未使用意味系列への一般化試験ではない。新しい source panel、資格試験、モデル fitting は今回の実行に含めない。

全実行に共通して、Gemma checkpoint、tokenizer、BF16 eager、slots=8、cut=13、writer=25 を既存 binding に従って固定する。source capture は 256 tokens、G／自由 D の host は 512-token ceiling、task host は 1,024-token ceiling とする。別の host を使った部品の loader check を無効にしない。source、G、renderer、既存 readers、transport、role writer と pointers は再学習しない。

測定前に展開して保存するマニフェストには、全 source、全 question、全 route、全 gold ledger、文字列の hash、部品の hash、順序、重複計算をまとめる規則、予算、停止規則を含める。必須項目は `base_commit`、実行コードの commit／SHA256、`source_ledger_sha256`、`question_ledger_sha256`、`route_ledger_sha256`、`scorer_sha256`、checkpoint／tokenizer／全部品の SHA256、`reader_bindings`、`geometry_bindings`、`resource_caps`、`replay_cache_bindings` とする。未設定項目、未知の route、重複して異なる source を指すキーを検出したら実行前に拒否する。freeze receipt にこのマニフェストの SHA256 と時刻を残し、測定後の変更は新しい計画として扱う。gold は採点専用の別オブジェクトに置き、G は source state と固定演算名だけを受け取る。後の world、A/B 順序、正解は G に渡らない。qualification データへのアクセスを含む generator・loader は起動しない。

## 4. S：構造化された更新と hard 化の切り分け

### S1：計算だけで確定する部分

二つの公開済み structured panel を別々に扱う。推定 trigger、veto、役割、action fields と authored premises の一致、G と `F_inferred` の rows のバイト一致、global 16-action argmax の正答、renderer の逆 parse を保存する。

次の二つの正答は分ける。

- **元の方策への正答**：source の独立 ledger と一致するか。
- **読み出しの忠実性**：実際の中間状態・文章が指定している行動を、そのまま読んだか。

特に、global argmax が提示した二つの選択肢の外の行動を指す場合、旧 A/B 課題にはその文章に忠実な正解がない。`TARGET_OUTSIDE_OPTIONS` を記録し、元の方策について偶然正しかった A/B を faithful readout と数えない。global 16-action 正答、メニュー内の相対順位、最終 A/B 正答を並記する。

### S2：追加する同一条件の neural 測定

各 panel の保存済み G row に対し、以下を同じ元の NativeChoice prompt、同じ frozen structured reader、同じ answer order で読む。

1. `soft`：保存済みの 16 次元 row。その既存 score を利用する。
2. `hard_global`：その row の global argmax の one-hot。
3. `oracle_action`：独立 ledger が指定する action の one-hot。

world address、topic、選択肢、語彙 bank は固定する。保存済み soft logits の再利用は、reader weights、prompt token IDs、answer mapping、bank、dtype、注入位置と soft row bytes が旧測定とすべて同じ場合に限る。C の共通 question は別の入力なので、意味上の問いが対応するだけではこの cache を使えない。メニュー内に argmax を制限する操作を primary に混ぜない。それを追加する場合は、選択肢の情報を読み出し側で利用した別の補助対照と明記し、測定前のマニフェストに載せる。

主な量は `soft 正 → hard 誤` と `soft 誤 → hard 正` の件数、およびそれぞれが global row error・outside-options・answer order とどう対応するかである。正しく解けた source だけに後から絞って成功率を作らず、全分母と条件付き分母を両方出す。`oracle_action` 自体を読めない例は reader の能力不足として分離する。

soft と hard の差があれば、数値状態から記号化する段階の機能的損失を認定できる。文章が hard row を正確に記していれば、同じ損失を renderer の文字列生成バグに帰属しない。D と E(D) の総正答数が一致していても、例ごとの置換と hard 化で失われた情報を別に調べる。

### 二択の範囲外について残す限界

全 16 行動を含む選択肢形式の新規測定は、今回の S・N・C に含めない。旧二択向け reader をそのまま新形式に使うと、出力形式への不慣れが新しい交絡になる。今回は `TARGET_OUTSIDE_OPTIONS` と global 16-action row correctness を残す。そのニューラル識別能力まで必要になった時は、同じ形式の oracle action／oracle literal を正しく読めることを確認する別の事前登録を用意する。

## 5. N：native writer・transport・注入・reader の切り分け

対象は既存 raw panel の全 128 sources。menus `[0,1,4,5]`、二つの left trigger、veto absent／right-on、両 pole・両 source order・二つの既存 fresh style を固定する。新しい source の探索は行わない。boundary を唯一の primary G とし、他の三演算の旧結果は変更しない。

各 source について一度だけ、以下を保存する。

| 記号 | 捕捉対象 | 検査 |
| --- | --- | --- |
| E | 既存と同じ source encoding の 8 slots | archived E hash との一致 |
| W | boundary prompt による `post_writer` の 8 rows | 実際に使われた forward の出力 |
| Z | 固定 transport を適用した `T(W)` | archived boundary hash との一致 |
| J | reader の対象層へ実際に書き込まれた tensor | dtype 変換後の Z と完全一致 |
| E* | 独立に記述した正しい successor の符号化 | 対象 text は正しいが、状態の意味保存は別途検査 |

E／boundary Z／E* を元の raw reader・元の question で読み、その forward で J を捕捉する。四世界・両 A/B 順序を対象とし、既存 score の再現と注入の receipt を同時に得る。J の検査は実際に採点した全 reader route と全注入入力で行う。同一物理入力をまとめる場合も、その完全な入力 hash と全論理観測への dispatch を保存する。C で追加する別 reader でも同じ検査を行う。

保存済み hash が再現しなければ、そのケースを別実験へ読み替えず `TECHNICAL_MISMATCH` で停止する。観測 hook の有無による full-logit parity と neutral self-replay、state isolation を確認する。state の比較では shape、dtype、層、位置を必ず保存する。

opposed、equivalent、source-order donor の対応は source マニフェストで固定する。特に opposed の同一質問・反対 target について、W と Z の厳密な衝突を調べる。

- W が同一で target が違うなら、その deterministic downstream は両側を解けない。必要な区別は遅くとも writer 出口までに利用不能になっている。
- W が違い、Z が同一になれば、その観測対の数値的な区別が transport で消えたことを示せる。
- Z が違っても reader の出力が同一なら、reader がその区別を使った証拠はない。区別が正しい意味を表すかは、この事実だけでは分からない。
- J が Z と違えば注入の実装不一致であり、意味の評価に数えない。

近い距離、cosine、norm、低い rank は記述統計に留める。近接を「意味が消えた」へ読み替えない。ハッシュがすべて違っていても、この診断が意味の所在を一意に決めるとは限らない。

E* を frozen reader が読めない場合、`G 誤答・E* 誤答` から writer／transport のどちらかを犯人にしない。E* の原文が正しいことは、その encoder が正しい意味を読みやすい座標で書いたことの証明にはならない。

## 6. C：同じ問いと読み手で六経路を対応付ける

対象は既存 raw 自由生成 subset の **16 sources × 4 worlds × 2 answer orders = 128 問い**。`free_subset` の既存規則をそのまま使う。E と boundary の保存済み 32 generations を利用し、新しく良い自由生成を選び直さない。残る 112 sources には native D が未測定であることを残し、16 sources の D 成績を全 128 sources に外挿しない。

共通 question は `supported_boundary.questions()` が生成する、世界条件を含む全文とする。同じ question、A/B 対応、full-vocabulary greedy と margin の採点を全枝に使う。reader は次の二つを**両方とも**固定して交差させる。

| reader | archived SHA256 | 位置付け |
| --- | --- | --- |
| raw reader | `c3b57dd3b5370c34eb2c880c7943b0b8ce57e4a72aea93d3c00bd9769bc3bef2` | primary 共通 reader |
| structured row reader | `427e138c8693e60ebf28074836a8452dc6c7da93e9bab1c876c03b33dc0b8909` | 事前固定の交差対照 |

両者の元の task-host binding は同じ 1,024-token host identity である。保存済み weight をその一致する host に通常の loader で読み込み、binding を緩めない。新しい full-question prompt は別の測定条件として hash を持つ。二つのうち成績の高い reader を採用して主結果を差し替えない。

structured identity／`F_inferred`／structured G は、共通 world で選んだ soft row を同じ lexical embedding bank に送る。native identity／G は cut-13 の slots を用いる。D は実際に出た本文だけを literal input として与える。再符号化は実際の本文だけを使い、原 source、元の G、gold を追加しない。構造化 row 再符号化と raw workspace 再符号化は別の route 名とする。

この交差により、次を分ける。

1. reader を固定した時に G と identity がどう違うか。
2. 同じ状態・問いで reader だけを替えると回復するか。
3. D の本文が区別を失っているか、正しい本文を reader が使えないか。
4. 再符号化で読めるようになった例と、読めなくなった例がどれか。

各 reader で oracle action one-hot、oracle canonical text、native E*、empty、zero を同じ question で測る。encoded oracle の能力を曖昧な全体合格で判定しない。同じ world・source form・source order の opposed pair について、両 pole × 両 A/B 順序の **4 回すべて**で E* が主採点に正答した対を、当該条件での正の対照とする。この条件を満たす対の件数と complete-policy 件数を報告する。満たさない対での native 意味帰属は `UNRESOLVED_READER_OR_CARRIER` とし、満たした対に条件付けた G／identity 比較と全体の比較を併記する。structured と native の総差には外部行選択・注入層という違いが残るので、G 単体の因果効果とは報告しない。

## 7. 正の対照が読めない場合に残す、次の一つの選択肢

S・N・C は推論・捕捉だけで、学習を行わない。二つの reader が native E* を読めず、読み手と状態の区別が残った場合は、その未決を結果として閉じる。今回の結果を見ながら reader を追加・選択することはしない。

その後の候補は、G の出力を学習にも選択にも使わず、正式に記述した explicit action tables を E で符号化した状態だけから、一つの診断用 reader を学習することである。この候補を実行するなら、adapter の初期値、訓練本文、検証本文、固定予算、正の対照の基準を新しいマニフェストで事前登録する。

この学習は carrier と reader の較正であり、E* が意味を正しく保持することの独立証明ではない。同じ真理値表を文面だけ変えて検証する場合、得られるのはその既知意味集合での読み出し可能性である。元の G と identity を同じ reader で読み、opposed 両側・両 answer order の切替を確かめる。成功すれば限定された正のアクセス証拠になるが、失敗した probe から意味の不在を結論しない。

## 8. 採点・分母・停止の規則

主採点は full-vocabulary greedy token が正解ラベルであり、かつ正解の A/B margin が厳密に正であること。`argmax` の tie-break で A を選んだだけの例は正答にしない。加えて、greedy token 単独、A/B margin、厳密 tie、非 A/B 出力を別列で残す。旧 raw panel での微小な成功率差を、回答内容の切替と読み替えない。

| status | 意味 |
| --- | --- |
| `SCORED_CORRECT` / `SCORED_INCORRECT` | 有効な観測で主採点が真／偽 |
| `TIE` | A/B が厳密同点 |
| `NON_LABEL` | full-vocabulary winner が A/B 以外 |
| `TARGET_OUTSIDE_OPTIONS` | 中間文章の行動に忠実な答えが、提示選択肢にない |
| `NOT_MEASURED` | 当初の subset に存在しない経路 |
| `EMPTY` / `UNENCODABLE` | D が空、または再符号化 geometry に収まらない |
| `CENSORED` | EOS 前に所定の生成上限に達した。本文を可能な範囲で採点し、打切りだけで意味の失敗にはしない |
| `TECHNICAL_MISMATCH` / `NONFINITE` | binding、replay、注入、数値の不一致。意味評価は無効 |
| `UNRESOLVED_READER_OR_CARRIER` | 正の対照不足のため、writer と reader の帰属が未決 |

全論理観測数、物理 forward 数、利用可能な対応数、各 status 数を並記する。missing・tie・非ラベルを分母から落とさない。単一 status で重なる性質を失わないよう、censoring や outside-options は独立したフラグとしても持つ。

全 route pair について `正→正／正→誤／誤→正／誤→誤` の件数を出す。complete policy は四世界・両 A/B 順序の全成功を要求する。change-required、veto absent の justified stability、論理的に disjoint な stability は別項目にする。既存 raw panel は absent stability だけを含むので、disjoint native の欄は `NOT_MEASURED` とする。この計画では、その欠測を埋めるためだけに source panel を拡張しない。

方策・問い・順序の反復を独立標本として数えない。primary の件数は固定有限 panel の全数として示す。補助的な paired bootstrap は元の topic を単位とし、10,000 draws、seed `4719`。structured は 8、raw は 4、自由 D の完全対応 subset は 2 topic しかない。2-topic subset では一般化を表す精密な区間を主張せず、topic ごとの対比較をそのまま掲載する。

### 予算と途中停止

S・N・C を合わせて 40,000 model forwards、12 million processed tokens、2,400 秒の上限を置き、先に到達した上限で止める。実行順は S2、N、C、固定入力キー順。既存の同一物理入力の scores は再利用し、再利用元 hash を各 row に記録する。正の対照に関係なく予定ケースの脱落理由を保存し、予算内で一部だけ終わった結果を全経路の成功として扱わない。

技術的不一致からの再開は、既存の入力・重み・保存済み状態・生成の hash を変えない修復に限り、停止記録と修復理由を別に残す。実装修復を scientific retry に使わない。

## 9. どの結果なら、どこまで帰属できるか

| 観測の組合せ | 許される結論 |
| --- | --- |
| 推定前提が誤り、正しい field への介入で G が正しくなる | その field の推定が当該例の失敗に寄与した。G 自体の利得とは別 |
| 同じ推定前提で G と F が完全一致 | この課題では G が形式的な解釈を状態へ先に展開した。F に対する意味的優位は未提示 |
| soft reader は正、hard reader は誤、文字列は hard row に忠実 | その reader で使えた情報の損失は遅くとも hard 化にある |
| gold に正しい文章を同じ reader が誤読 | 更新の正しさとは独立した読み出しの失敗がある |
| 文章に既に区別がなく、E(D) でも同じ | この D を通る経路では区別が失われた。元の G に意味がなかった証拠ではない |
| Z は正しく注入され、独立 reader は正、元 reader は誤 | 更新後状態の区別は、その独立 reader では利用できる。元 reader の失敗を分離できた |
| E* は読めるが G は読めず、W/Z の衝突位置が特定できない | この reader に対する native 更新パイプラインは失敗。writer と transport の意味的な帰属は未決 |
| E* も読めない | G の誤答だけで「更新なし」を判定しない |

field 介入をいくつも順に足した場合、改善の重なりを独立の寄与として加算しない。単独介入と同時介入、修復と新たな破壊の両方を記録する。

## 10. G が形式的な直接解釈を超える、という主張の判別条件

固定された完全な前提 P と今回の Boolean rule に対し、正しい形式解釈 F(P, world) が既に答えを定める。同じ情報・同じ課題で、G がそれ以上の正しい事実を増やすという勝ち方はない。identity より良いことは、pending veto を処理する必要性を示すが、F への優位とは違う。

次の候補が獲得すべき機能は、別々の対照を置いて検査する。

| 狙う機能 | 必要な比較 | 現計画での位置付け |
| --- | --- | --- |
| 推定の誤りに頑健な帰結の構成 | 同じ soft evidence にアクセスする F と G。G にだけ richer source を渡さない | S で誤りの入口を特定。hard-premise F を超えただけなら入力情報量の差を明記 |
| 後の読み手が使える状態への書き込み | 意味が正しい oracle state、同じ reader、同じ問い、opposed 両側の切替 | N・C の主対象。正の対照が不足すれば未決として閉じる |
| 言語化・再符号化後にも使える区別 | 実際の D 本文、本文に忠実な target、同じ reader の対比較 | S・C の主対象 |
| 次の更新に必要な役割・履歴の保持 | 同じ action table を持つ対立履歴。F にも同じ必要情報を許し、table-only F は弱い対照として区別 | 今回の一段診断から先の実験。現実行では二段 G の成功を主張しない |
| 繰り返し問いに答える計算上の利点 | F も事前計算・cache 可能とし、前処理、各問いの費用、保存容量を含める | 正確さの上乗せとは別。現在の小さな形式 F に優位があるとは仮定しない |

この開発試験の望ましい終点は、成功率の最大化ではなく、「前提を誤った」「正しい soft 出力を hard 化で壊した」「書けたが特定の reader が使えない」「native 更新と reader をまだ分離できない」を、対応する観測と未測定項目を添えて残せることである。どの結果でも、独立資格試験の開封、意味ゲートの承認、preprint v1 の主張拡張には進まない。
