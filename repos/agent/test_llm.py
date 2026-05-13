"""
LLM 测试脚本 — 验证 Qwen3-14B 在计算节点上正常工作

用法:
    # GPU 0,1 给 LLM
    CUDA_VISIBLE_DEVICES=0,1 python repos/agent/test_llm.py

同时也是 orchestrator.py 的基础模板。
"""

import os
import sys
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/models/Qwen3-14B"


def load_model():
    print("加载 Qwen3-14B...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",        # 自动分配到可见 GPU
    )
    model.eval()

    elapsed = time.time() - t0
    print(f"✅ 模型加载完成，耗时 {elapsed:.1f}s")

    # 显示显存占用
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {used:.1f}GB / {total:.1f}GB")

    return tokenizer, model


def chat(tokenizer, model, user_message: str, thinking: bool = False) -> str:
    """
    调用 Qwen3-14B 生成回复。

    thinking=True  : 启用慢思考模式（更准确，更慢）
    thinking=False : 快速模式（适合代码生成）
    """
    messages = [{"role": "user", "content": user_message}]

    # Qwen3 的 thinking 模式控制
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=not thinking,   # thinking模式用贪心，快速模式用采样
            temperature=0.7 if not thinking else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    # 只取新生成的部分
    new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"   (生成耗时: {elapsed:.1f}s, {len(new_tokens)} tokens)")
    return response.strip()


def main():
    print("=" * 60)
    print("Qwen3-14B 功能测试")
    print("=" * 60)

    # 检查 GPU
    if not torch.cuda.is_available():
        print("❌ 没有可用 GPU，请在计算节点上运行")
        sys.exit(1)

    print(f"可见 GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # 加载模型
    tokenizer, model = load_model()
    print()

    # ──────────────────────────────────────────
    # 测试1: PDE 领域知识
    # ──────────────────────────────────────────
    print("【测试1】PDE 领域知识")
    print("-" * 40)
    prompt1 = """以下是 1D Burgers 方程的 FNO 模型训练日志：

Epoch  1/50 | train_loss=0.2644 | val_loss=0.0769 | val_rel_mse=0.0073
Epoch  5/50 | train_loss=0.0821 | val_loss=0.0312 | val_rel_mse=0.0028
Epoch 10/50 | train_loss=0.0634 | val_loss=0.0289 | val_rel_mse=0.0024
Epoch 20/50 | train_loss=0.0521 | val_loss=0.0271 | val_rel_mse=0.0022
Epoch 50/50 | train_loss=0.0498 | val_loss=0.0264 | val_rel_mse=0.0021

推理评测结果：
Seg1 (0-47步)   Rel-MSE=0.0286  Score=56.48
Seg2 (47-95步)  Rel-MSE=0.1832  Score=16.01
Seg3 (95-190步) RMSE=0.0691     Score=59.15
总分: 47.70 / 100

请分析：
1. 当前模型的主要瓶颈是什么？
2. 为什么 Seg2 的分数远低于 Seg1 和 Seg3？
3. 提出 2-3 个具体的改进方案，并说明理由。
"""
    response1 = chat(tokenizer, model, prompt1, thinking=False)
    print(response1)
    print()

    # ──────────────────────────────────────────
    # 测试2: 代码生成能力
    # ──────────────────────────────────────────
    print("【测试2】代码生成能力")
    print("-" * 40)
    prompt2 = """当前 FNO 训练使用的是相对 L2 损失函数：

def relative_l2_loss(pred, target):
    diff = pred - target
    return (diff.norm(dim=-1) / (target.norm(dim=-1) + 1e-8)).mean()

请在此基础上，添加 Burgers 方程的物理残差损失（Physics-Informed Loss）：
- Burgers 方程: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
- ν (粘性系数) = 0.001
- 空间步长 dx = 1/256
- 时间步长 dt 对应 reduced 轴，即真实 dt × 5

只需输出修改后的损失函数 Python 代码，不需要解释。
"""
    response2 = chat(tokenizer, model, prompt2, thinking=False)
    print(response2)
    print()

    print("=" * 60)
    print("✅ 所有测试完成，LLM 工作正常")
    print("下一步: python repos/agent/orchestrator.py")
    print("=" * 60)


if __name__ == "__main__":
    main()