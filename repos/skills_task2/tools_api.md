# tools.py 接口规范

tools.py 由框架层维护，Agent 不直接调用这些函数。
orchestrator 通过 execute_tool() 分发，Agent 在输出 JSON 时指定工具名和参数。

## 工具调用格式

Agent 输出格式：
```json
[
  {"tool": "工具名", "参数1": "值1", "参数2": "值2"},
  {"tool": "工具名", "参数1": "值1"}
]
```

---

## 文件操作工具

### write_file
写入文件到 workspace/，自动备份旧版本到 .history/，同步到 submission/code/。

```json
{"tool": "write_file", "path": "train.py", "content": "...完整代码..."}
```

| 参数 | 说明 |
|---|---|
| `path` | 文件名，如 `"train.py"`，不要加 `workspace/` 前缀 |
| `content` | 文件完整内容 |

返回：
```json
{"success": true, "path": "workspace/train.py", "lines": 120, "version": 1}
```

### read_file
读取文件内容，支持 workspace/ 和 skills/ 两种路径。

```json
{"tool": "read_file", "path": "train.py"}
{"tool": "read_file", "path": "skills_task2/scoring.md"}
```

返回：
```json
{"success": true, "path": "workspace/train.py", "content": "...", "lines": 120}
```

### edit_file
精准替换文件中的唯一字符串，比 write_file 省 token，适合小改动。

```json
{"tool": "edit_file", "path": "train.py", "old_str": "lr=0.001", "new_str": "lr=0.0005"}
```

**注意**：`old_str` 必须在文件中唯一出现一次，否则报错。

返回：
```json
{"success": true, "path": "workspace/train.py", "lines_changed": 1, "version": 2}
```

### list_files
列出目录文件。

```json
{"tool": "list_files", "dir": "workspace"}
{"tool": "list_files", "dir": "skills"}
```

返回：
```json
{"success": true, "dir": "workspace", "files": [{"name": "train.py", "size_kb": 4.2, "modified": "05-17 10:30"}]}
```

---

## 执行工具

### run_bash
同步执行任意 bash 命令，适合短命令（<5分钟）。

```json
{"tool": "run_bash", "command": "ls workspace/checkpoints/", "timeout": 30}
```

返回：`{"stdout": "...", "stderr": "...", "returncode": 0, "elapsed": 0.5}`

### smoke_test
★ 修改代码后必须先调用。单 batch 快速验证代码能否端到端运行（30-60秒）。

```json
{"tool": "smoke_test", "script": "train.py"}
```

返回：
```json
{"success": true, "elapsed": 45.2, "error_lines": [], "log_tail": "..."}
```

**工作流**：`write_file` → `smoke_test` → （通过后）`run_training`

### run_training
封装 DDP 训练，自动处理 torchrun、NCCL、随机端口，实时写日志到 train_live.log。

```json
{"tool": "run_training", "script": "train.py", "epochs": 50, "n_gpus": 4}
{"tool": "run_training", "script": "train.py", "extra_args": "--lr 0.0005 --batch_size 16", "epochs": 30}
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `script` | `"train.py"` | 训练脚本名 |
| `epochs` | None | 指定后自动追加 --epochs N |
| `extra_args` | `""` | 额外命令行参数 |
| `n_gpus` | 4 | GPU 数量 |
| `timeout` | 3600 | 超时秒数 |

返回：
```json
{
  "success": true,
  "val_loss": 0.071,
  "best_epoch": 42,
  "elapsed_seconds": 287,
  "ckpt_path": "/mimer/.../workspace/checkpoints/best.pt",
  "log_tail": "Epoch 50/50 val=0.071 <- best"
}
```

### run_bash_async
后台执行长任务（>10分钟），立即返回 pid，用 check_process 监控进度。
Task 2 长训练时使用，Task 1 由于时间限制建议用同步的 run_training。

```json
{"tool": "run_bash_async", "command": "python workspace/train.py --epochs 200"}
```

返回：`{"success": true, "pid": 12345, "log_file": "/mimer/.../logs/bg_task_xxx.log"}`

### check_process
查看后台进程状态和日志末尾。

```json
{"tool": "check_process", "pid": 12345, "tail_n": 30}
```

返回：`{"pid": 12345, "running": true, "status": "running", "log_tail": "...", "current_val_loss": 0.08}`

### benchmark_timing
在选定大模型配置前先测单 epoch 耗时，确保不超时。

```json
{"tool": "benchmark_timing", "script": "train.py", "n_epochs": 2}
```

返回：
```json
{
  "per_epoch_seconds": 45.2,
  "per_epoch_minutes": 0.75,
  "estimated_50ep_minutes": 37.7,
  "note": "单 epoch 约 0.8 分钟，50 epoch 预计 37.7 分钟"
}
```

---

## 评估工具

### quick_eval
快速读取 checkpoint 里的 val_loss，秒级返回。训练后先用这个判断方向。

```json
{"tool": "quick_eval", "ckpt_path": "checkpoints/best.pt"}
```

返回：`{"success": true, "val_loss": 0.071, "ckpt_path": "..."}`

### full_eval
完整推理，计算 Seg1/2/3 真实竞赛得分。确认 val_loss 合理后再用。

```json
{"tool": "full_eval", "ckpt_path": "checkpoints/best.pt", "val_only": true}
```

返回：
```json
{
  "success": true,
  "total_score": 64.9,
  "seg1_score": 15.6,
  "seg2_score": 82.1,
  "seg3_score": 80.9,
  "precision_score": 48.7,
  "infer_time": 0.34,
  "infer_score": 39.9
}
```

**推荐工作流**：`run_training` → `quick_eval` → `full_eval`

---

## 工作记忆工具

### todo_write
写入任务列表到 workspace/todos.json，避免长对话中遗忘计划。

```json
{"tool": "todo_write", "todos": [
  {"content": "写 train.py",  "status": "completed",   "priority": "high"},
  {"content": "smoke_test",   "status": "completed",   "priority": "high"},
  {"content": "正式训练50ep", "status": "in_progress", "priority": "high"},
  {"content": "full_eval评分","status": "pending",      "priority": "normal"}
]}
```

status 状态机：`pending` → `in_progress` → `completed` / `failed`

### todo_read
读取当前任务列表，查看进度。

```json
{"tool": "todo_read"}
```

返回：`{"todos": [...], "completed": 2, "in_progress": 1, "pending": 1}`

---

## 查询工具

### read_training_log
训练进行中查看实时日志，了解当前 epoch 进度。

```json
{"tool": "read_training_log", "max_lines": 30}
```

### get_score_history
查看所有迭代的实验记录（从 experiment_log.md 读取）。

```json
{"tool": "get_score_history"}
```

### check_remaining_time
查看时间预算状态和预计时间分数。

```json
{"tool": "check_remaining_time"}
```

返回：
```json
{
  "elapsed_minutes": 23.5,
  "remaining_minutes": 31.5,
  "time_score_prediction": 35,
  "should_stop": false
}
```

---

## 常见错误和对策

| 错误 | 原因 | 对策 |
|---|---|---|
| `edit_file` 返回 "old_str 未找到" | 字符串不存在或有空格差异 | 先 `read_file` 确认实际内容 |
| `edit_file` 返回 "出现N次，必须唯一" | 字符串重复 | 扩大 old_str 范围使其唯一 |
| `smoke_test` 失败 | 代码有语法或运行时错误 | 看 error_lines，read_file 检查代码 |
| `run_training` success=false，val_loss=null | 训练崩溃 | 看 log_tail 里的错误，先 smoke_test |
| `full_eval` success=false | predict.py 有错误 | 看 error 字段，检查评分函数 |
| `quick_eval` Checkpoint 不存在 | 训练没有产生 best.pt | 检查 CKPT_PATH 是否写对 |