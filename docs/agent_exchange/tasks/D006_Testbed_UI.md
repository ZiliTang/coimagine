# D006：Web testbed UI（D005 的交互层）

- 2026-10-29，DeepSeek v1；用户明确要求开始 D006。GPT 额度耗尽。
- 范围：只做 UI/交互层，直接消费 D005 的 `OptionSystemEnv`/`TestbedSession`；不另写 Pymunk/option/env；不训练、不改 reward/物理/观测/动作，不发布 D007。
- 目标：让用户能拖动物体/机器人、手动编 option 序列、在线运行、逐帧/播放、看基础信息和阶段颜色条带、保存多轮 run、调整 option 参数并导出 override。

## 1. 页面

- 场景：D005 的 panel 8 个场景；选择后画布显示 arena/goal/物体/两机器人；鼠标拖动位置，运行前可重置。
- 序列：两个 option 下拉（46 个 option）+ 持续秒数；加入/删除/清空；列表显示每段两侧 option 和时长。
- 运行：点击 Run 用同源 `TestbedSession` 在服务端 Pymunk 跑；返回 129 帧/state strip/coordinator；画布回放。
- 播放：Play/Pause、上一帧/下一帧、滑杆；时间线随播放移动。
- 条带：两侧 option stage、joint phase、contact/joint_active 的颜色行；点击跳帧。
- 信息：当前时间、object/robot 位姿、stage、joint phase、contact、force、success/termination/reward。
- Run 列表：保存多次 run；点击切换重放；显示成功/时长/序列。
- 参数：按 family 分组；可调 default→override；导出 `outputs/testbed/overrides/<name>.json`；Run 使用当前 override。

## 2. API

- `GET /api/scenes`、`/api/options`、`/api/params`、`/api/runs`、`/api/runs/<id>`
- `POST /api/run`：scene_id、object/robots poses、sequence、overrides、name
- `POST /api/overrides`：name + overrides；保存 JSON
- 静态文件和 API 同一本地 server；默认 `127.0.0.1:8765`。

## 3. 验收

- server 启动后 `curl` 首页/API 200；JS `node --check` 通过；至少一次 UI API run 成功（W_N 默认实例/短序列），run JSON 保存；run 列表能读回。
- 不改 D005 的 env/物理；UI 只是消费 session。
- 手动检查：拖拽、加段、跑、播放/逐帧、条带、切换 run、参数导出。
- 预算：UI 运行短序列 ≤30 s；不训练；启动本地 server 用 managed job 并验证 URL。

## 4. 交付

- `experiments/option_rl_v19/testbed_server.py`、`testbed/index.html`、`app.js`、`style.css`。
- `outputs/testbed/runs/`、`outputs/testbed/overrides/`。
- `results/D006_Testbed_UI.md`；CURRENT 到 ready_for_review。
