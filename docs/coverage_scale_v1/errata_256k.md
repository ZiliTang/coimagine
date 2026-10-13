# coverage_scale_v1 — 256k 续跑更正记录

只记录本轮（256k 续跑）发现并修复的问题。第 1 轮的更正见 [errata.md](errata.md)。

## E10 `RootPlan.ensure` 从零重采样并整体追加（已修复；第 1 轮结果未受影响）

**现象。** 续采到第 300 shard 之后，去重率从 6.9% 突升到 33.5%：新增 123 个 shard 只贡献约
1400 个净窗口。

**原因。** `RootPlan.ensure(split, count)` 在文件不足 `count` 时调用
`sample_roots(split, count)` —— 这是**从第 0 个根重新生成**，而不是从已有前缀续接 —— 然后把
**整份新计划**追加到 `roots_<split>.jsonl`。第二次物化（`--train-shards` 330）之后文件变成
4800+5280=10080 行、只有 5280 个唯一 `root_id`。第三次物化读到 10080 行（≥ 目标 9600），不再
追加，直接返回**文件前 9600 行**，于是 shard ≥300 全部指向已经采过的根，采出的窗口逐字节重复。

**影响范围。** 第 1 轮的 128k 档只用了 287 个连续 shard，全部落在正确的 shard 0–299 内，
**128k 数据、val、test、候选对照与已发表结论完全未受影响**（已用 `transition_id` 前缀断言复核）。

**修复。**
- `ensure` 只追加 `sample_roots(split, count)[len(known):count]` 这一段，并在追加前核对已有前缀
  与标准计划逐条一致，不一致就报错退出；
- `_read` 按 `root_id` 去重（保留首次出现），所以即使文件曾被污染，读出的仍是标准前缀；
- 被污染的 shard 300–434 全部移入 `superseded_duplicate_shards/`（数据、meta 与 `_done` 标记
  一并移走），进度日志另存 `collection_progress_superseded_256k.jsonl`，不计入任何规模统计；
  这些 shard 用修复后的计划重新采集。
- 回归测试：`test_plan_extension_appends_only_the_new_block`、
  `test_duplicate_plan_rows_are_collapsed_on_read`。

## E11 `scene_id` 在约 5000 根之后发生哈希碰撞（已修复；已存场景 id 不变）

**现象。** 计划扩展到 9600 根时，注册表报 `scene_id 'S0_r107997672' already registered with
other content`。

**原因。** `scene_id = {family}_r{seed:09d}`，而 `seed = 基址 + SHA256(版本|split|family|mode|
ordinal)[:6]` 只有 **24 位**（约 1.67e7 个取值）。当一个 split 的根数到数千时，生日界使得
不同 `(family, mode, ordinal)` 撞到同一个 seed 是必然的：实测
`train/S0/goal/738` 与 `train/S0/wall/837` 同 seed，但几何不同。

**修复。** 新增 `plan.resolve_scene()`：追加根时若该 `scene_id` 已被**不同内容**占用，就把该根
确定性地改名为 `{family}_{mode}_r{seed:09d}`（仍冲突则再加 `_o{ordinal:05d}`），并让 shard 记录
写入改名后的规格。**只有真正冲突的那个根会被改名**，因此已经物化的 5280 个场景 id 逐字节不变
（已断言 `roots_train.jsonl` 前 5280 行不变）。没有改 `root_seed`（那会改变已有数据）。
回归测试：`test_scene_id_collision_is_resolved_deterministically`。

## E12 采集 worker 由 4 提升到 6（资源调整，非协议变更）

只为缩短墙钟。worker 数不影响 `shard → root block` 的映射，也不改变生成器、配方、去重或 split；
每个 shard 的内容仍只由 `(split, shard_index)` 决定，且已由 E10 的连续前缀断言保证可复现。
