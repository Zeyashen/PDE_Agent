# guandi_agent

PDE Neural Operator Research Agent — AI4S CNS Challenge Task 1 (1D Burgers).

## 目录

```
guandi_agent/
├── conf/experiments/         实验配置 (yaml)
├── src/core/                 干科研代码 (不依赖 LLM)
├── src/agent/                LLM 调度层 (Phase 3 实现)
├── src/submission/           打包提交 (Phase 4 实现)
├── runs/                     训练产物
├── scripts/                  运维脚本
└── tests/                    单元/冒烟测试
```

## Phase 1 用法 (无 agent)

```bash
# 容器内
PYTHONPATH=src python -m pytest tests/test_smoke.py -v    # smoke test
PYTHONPATH=src python -m core.cli train --config conf/experiments/v1_baseline.yaml
PYTHONPATH=src python -m core.cli predict --config conf/experiments/v1_baseline.yaml
```

## 设计原则

- LLM 只能改 yaml，不能改 .py 源码
- 每个实验独立目录，永不覆盖
- 三轨日志: jsonl (机器) + SQLite (查询) + markdown (评委)
- 详见 architecture.md
