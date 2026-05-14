import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate():
    # 从 stdin 读取输入参数
    input_data = json.load(sys.stdin)
    model_path = input_data["model_path"]
    messages = input_data["messages"]
    temperature = input_data.get("temperature", 0.7)
    
    # 无论外面传进来的是 "0" 还是 "1"，因为 client 已经设置了 CUDA_VISIBLE_DEVICES
    # 在这个子进程里，模型看到的合法设备名永远是 "cuda:0" 或 "auto"
    device = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # 让 transformers 自动分配到当前可见的卡上
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 构造 Prompt
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 推理
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=4096,
        temperature=temperature,
        top_p=0.9,
        do_sample=True
    )
    
    response = tokenizer.batch_decode(
        generated_ids[:, model_inputs.input_ids.shape[-1]:], 
        skip_special_tokens=True
    )[0]

    # 结果输出给 Client
    print(json.dumps({"response": response}))

if __name__ == "__main__":
    generate()
    