# 路径常量（所有代码必须使用这里的路径）

## 项目根目录
PROJECT = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent

## 代码目录（你写的所有代码都在这里）
WORKSPACE = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/workspace

## 提交目录
SUBMISSION = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/submission

## Task 1 数据
TASK1_TRAIN = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5
TASK1_VAL   = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_val.hdf5
TASK1_TEST  = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_test.hdf5

## Task 2 数据
TASK2_PART0 = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task2_part0_train.h5
TASK2_PART1 = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task2_part1_train.h5
TASK2_PART2 = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task2_part2_train.h5
TASK2_VAL   = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task2_val.h5
TASK2_TEST  = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task2_test.h5

## 提交输出文件
TASK1_PRED  = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/submission/task1_pred.hdf5
TASK2_PRED  = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/submission/task2_pred.hdf5

## Checkpoint 保存路径（训练脚本中使用）
CKPT_DIR    = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/workspace/checkpoints
BEST_CKPT   = /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/workspace/checkpoints/best.pt

## 使用规范
- 所有训练和推理脚本中，数据路径必须使用上面的绝对路径
- 不要使用相对路径（./data 等），容易因工作目录不同而出错
- write_file 时只传文件名，如 "model.py"，不要加 "workspace/" 前缀
- run_training 的 script 参数只传文件名，如 "train.py"
- full_eval 的 ckpt_path 传相对路径，如 "checkpoints/best.pt"

## 代码中的路径写法示例
```python
# 正确写法
TRAIN_PATH = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5"
VAL_PATH   = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_val.hdf5"
CKPT_PATH  = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/workspace/checkpoints/best.pt"
PRED_PATH  = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/submission/task1_pred.hdf5"

# 错误写法（不要用）
TRAIN_PATH = "./data/train.hdf5"          # 相对路径
CKPT_PATH  = "workspace/checkpoints/..."  # 带 workspace 前缀
```
