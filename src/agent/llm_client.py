import subprocess
import json
import os
from pathlib import Path

class LLMClient:
    def __init__(self, model_path: str, python_bin: str, device: str = "0"):
        self.model_path = model_path
        self.python_bin = python_bin
        self.device = device
        # 这里的 worker_script 是我们等下要写的独立推理脚本
        self.worker_script = Path(__file__).parent / "llm_worker.py"

    def chat(self, messages: list, temperature: float = 0.7) -> str:
        # 去掉 device 传参，完全通过环境变量控制
        input_data = json.dumps({
            "model_path": self.model_path,
            "messages": messages,
            "temperature": temperature
        })
        
        # 严格控制子进程的环境变量
        current_env = os.environ.copy()
        current_env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        current_env["PYTHONPATH"] = "src"
        # 强制将配置文件中的 GPU 分配给子进程
        current_env["CUDA_VISIBLE_DEVICES"] = str(self.device) 

        try:
            process = subprocess.Popen(
                [self.python_bin, str(self.worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=current_env
            )
            stdout, stderr = process.communicate(input=input_data)
            
            if process.returncode != 0:
                return f"LLM Worker Error: {stderr}"
            
            result = json.loads(stdout)
            return result.get("response", "")
        except Exception as e:
            return f"Client Exception: {str(e)}"

