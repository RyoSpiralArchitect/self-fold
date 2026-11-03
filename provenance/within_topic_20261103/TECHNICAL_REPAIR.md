# Exact layout replay before fitting

The first sealed attempt at 3a7d4fc98752023423b4be75a32f415b77f5525d stopped during serialization of the single-style analytic initialization, before any optimizer update or new evaluation decision. All 16 teacher hashes, 24 earlier lexical decisions and 64 earlier E hashes replayed. The first attempt is retained.

The initial projection was column-major. A default NumPy array copy during loading produced row-major storage with identical values. On the first training input the projection stride changed from (1, 18432) to (16, 1). The maximum output-row difference was 0.0001220703125. The checked native logits were exactly equal. Preserving the saved array order gave exact rows and native logits on repeated loads.

The repaired loader copies using the saved memory order. The original plan, fixture, scorer, computation rules, schedules, seeds, thresholds and weights stay bound to their original hashes. Start in a different output directory and require the original analytic checkpoint arrays and source captures to replay. Both arms still have exactly 64 optimizer updates. This technical repair precedes the first new endpoint score.
