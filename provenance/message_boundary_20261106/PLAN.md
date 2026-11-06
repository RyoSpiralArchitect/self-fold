# Message boundary over a multi-edit table: development plan

**Status: development experiment. Not a qualification run; no result here opens the sealed panel or extends preprint v1.**

Background: `docs/G_MESSAGE_BOUNDARY_PLAN.md`. The role_memory study showed a learned one-bit message carries the role information a bare action table drops. The G-stage diagnostics showed a language roundtrip preserves the named (hard) action and drops the soft row. This experiment joins the two: it asks what the symbolic residue of an intermediate state loses, and whether a learned discrete message restores it.

## Design

- Same frozen source compiler (`compiler.npz`, SHA `ab3dd6…`) and frozen native row reader (`reader.npz`, SHA `427e13…`) as `role_memory_20261104`. Same topics, masks, sources and budgets as `data/role_memory_20261104.json` unless changed in `data/message_boundary_20261106.json`.
- Controller `BoundaryMemory`: the role_memory pointer/writer, generalized to a `width`-bit message. Each arm is `(form, channel, width)`:
  - `form=soft`: the next edit receives the soft rows (L analog).
  - `form=hard`: the next edit receives one-hot rows naming each row's argmax action — the symbolic residue a verbalization preserves (T/B analog within this structured interface).
  - `channel=reward|supervised|none|supplied`: learned-by-REINFORCE message, role-supervised message, dropped message, or the convention code supplied.
- Training: two on-policy edits, native-correctness reward through the frozen reader, same 512×128 budget, same seeds {17,41,73}, same exploration floor.
- Evaluation is exhaustive by construction, not selected: all 14 nonconstant masks → all 38,416 four-edit strings; correctness at each depth; all-four-correct rate stratified by off/off role-flip parity; opposed-donor reversal with signed margins; identical-table twin histories with intact vs erased message; written-bit vs convention agreement for message decoding.
- Hard rows are scored through a second blind packet (`packet_hard`) so soft and hard forms share the same reader, menus and orders.

## Decisions frozen before running

- No arm may see gold actions, prior triggers or the donor key; rewards reach the learner as scalars only.
- If a policy's two named actions collide under `hard`, the run stops rather than silently merging roles.
- The `none` channel still trains its row pointer by reward — it is a channel control, not an untrained baseline.
- Results report repairs and damage, all denominators, and per-seed values. A successful `hard_reward` arm supports "the learned message carries what verbalization dropped"; a successful `hard_none` arm would instead mean the soft mixture carried nothing this task needed.
