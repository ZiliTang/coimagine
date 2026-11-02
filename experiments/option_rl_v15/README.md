# V15：可学习的双机联合类型评分

本轮延续V14的固定8个开发训练场景与8个1cm扰动场景，集中检查到达目标后的协同制动。完整任务reward、成功判据、底层控制器、944维合法输入、flat两层256网络、分离critic、GAE10秒迹、0.8秒保持、熵0.02及确定性解码方式保持原设定。

唯一训练对照是是否增加由当前actor隐藏特征预测的14×14联合类型评分。它加到原合法联合分布的logits上；速度仍按各自类型条件选择。所有新评分初始为零，构造时不改变随机数状态，初始actor概率/value及首轮采样轨迹与独立头逐项一致。该头可以学习双方一起制动、推/随动等相关偏好；没有手工强制动作目标对齐、自动接近目标制动或特权几何mask。

保留独立版本文件以冻结旧实验。`controller.py`、`mode_env.py`和`scenes.py`复用v14；训练、worker、checkpoint与render使用v15接口；manifest记录直接文件和复用源文件的hash。旧checkpoint不直接续训该接口。

```bash
.venv/bin/python -m pytest tests/test_option_rl_v15.py
.venv/bin/python -m experiments.option_rl_v15.short_suite \
  --out outputs/option_rl_v15/joint_NEW \
  --panel outputs/option_rl_v14/learnability_002/panel.json \
  --iterations 40 --seconds-per-variant 300 --workers 4
```

`--only independent`与`--only joint_types`可分别运行；各组从头训练。首轮正式两组在`joint_ablation_002`并行运行，共8个采样worker，各自上限40轮或5分钟，到预算在本轮结束时退出。正式全随机validation/test未用于本轮调参或报告。

首轮已完成：两组均40轮、320个episode，36项聚焦测试通过，环境异常为0。最终贪心固定训练/扰动成功分别为独立头2/8、2/8，联合头4/8、4/8；随机策略诊断没有对应提升，改善集中在单机任务。详见[完整诊断报告](../../docs/handoff/20261029_Run15_Stopping_Diagnosis_and_Joint_Type_Head.md)。

前期`joint_ablation_001`的启动器误指向v14：联合组被CLI拒绝，独立组主动停止，完整结果保留但不作为配对对照。已改为根据启动器本身位置选取同版本train.py，监控启动冒烟验证interface_version为v15后才重开002。
