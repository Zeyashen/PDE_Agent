import sys
import re
import json
import yaml
import logging
import torch
from pathlib import Path
from agent.llm_client import LLMClient
from agent.tracker import Tracker
from agent.tools import AgentTools
from agent.prompts import FNO_SYSTEM_PROMPT, DEEPONET_SYSTEM_PROMPT

class Orchestrator:
    def __init__(self, cfg_default: dict):
        self.work_dir = Path(cfg_default["paths"]["work_dir"])
        db_path = self.work_dir / "runs" / "registry.db"
        
        print(f"DEBUG: 正在连接数据库 {db_path}...", flush=True)
        self.tracker = Tracker(db_path)
        
        print(f"DEBUG: 正在初始化 LLM (模型路径: {cfg_default['llm']['model_path']})...", flush=True)
        self.llm = LLMClient(
            model_path=cfg_default["llm"]["model_path"],
            python_bin=cfg_default["paths"]["python_bin"],
            device=cfg_default["llm"]["gpu_device"]
        )
        
        self.tools = AgentTools(
            self.work_dir,
            cfg_default["paths"]["python_bin"],
            self.tracker
        )
        # 根据 baseline config 的架构选择 prompt
        try:
            from core.config import load_config
            _baseline_path = cfg_default.get("paths", {}).get(
                "baseline_config",
                str(root_path / "conf" / "experiments" / "v1_baseline.yaml")
            )
            _cfg = load_config(_baseline_path)
            _arch = _cfg.model.architecture
        except Exception:
            _arch = "FNO1d"
        if _arch == "DeepONet1d":
            _prompt = DEEPONET_SYSTEM_PROMPT
        else:
            _prompt = FNO_SYSTEM_PROMPT
        self.messages = [{"role": "system", "content": _prompt}]

    def parse_action(self, text: str):
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
        first_action_raw = matches[0].strip()
        try:
            return json.loads(first_action_raw)
        except json.JSONDecodeError as e:
            print(f"Action JSON 解析失败: {e}\n内容原文: {first_action_raw}", flush=True)
            return None

    def run(self, max_iterations=8):
        print("\n" + "="*40, flush=True)
        print("Start", flush=True)
        print("="*40 + "\n", flush=True)

        # ── 阶段 0：官方权重推理（幂等，只跑一次）────────────────────
        off_scores = {}
        try:
            off_scores = self.tools.run_official_inference(exp_id="v0_official")
            print(f"基线: "
                  f"Seg1={off_scores.get('seg1_score', 0):.2f} "
                  f"Seg2={off_scores.get('seg2_score', 0):.2f} "
                  f"Seg3={off_scores.get('seg3_score', 0):.2f} "
                  f"Total={off_scores.get('total_score', 0):.4f}", flush=True)
        except Exception as e:
            print(f"官方推理失败，继续执行: {e}", flush=True)

        # ── 阶段 1：找最优自训实验，没有则以官方基线作为起点 ──────────
        best_exp = self.tracker.get_best_experiment(exclude_official=True)
        print(f"DEBUG: 数据库返回的最优自训实验: {best_exp}", flush=True)

        if not best_exp:
            print("无自训记录，将以官方基线 v0_official 作为首次实验的 parent。",
                  flush=True)
            best_exp = self.tracker.get_experiment("v0_official")
            if not best_exp:
                print("官方推理也未完成，无法启动。", flush=True)
                return

        # ── 构造初始 Prompt ───────────────────────────────────────────
        off_summary = ""
        if off_scores:
            off_summary = f"""
【官方预训练权重推理结果（仅作参考基准，不可修改）】
  Seg1={off_scores.get('seg1_score', 0):.2f}  Seg2={off_scores.get('seg2_score', 0):.2f}  Seg3={off_scores.get('seg3_score', 0):.2f}  Total={off_scores.get('total_score', 0):.4f}
"""

        user_input = f"""以下是当前实验状态，请开始优化迭代。
{off_summary}
【当前参考实验 {best_exp['exp_id']}】
  Seg1={best_exp.get('seg1_score', 0):.2f}  Seg2={best_exp.get('seg2_score', 0):.2f}  Seg3={best_exp.get('seg3_score', 0):.2f}  Total={best_exp.get('total_score', 0):.4f}

约束：
1. 所有新实验必须 init_from=scratch，禁止使用官方权重做 finetune。
2. 官方得分仅作参考基准，不是优化起点。
3. 每次只修改 1-2 个参数，方便归因。
4. parent_id 填写上方【当前参考实验】的 exp_id：{best_exp['exp_id']}

请提出第一个优化方案。"""

        self.messages.append({"role": "user", "content": user_input})

        # ── 主循环 ────────────────────────────────────────────────────
        for i in range(max_iterations):
            print(f"\n [第 {i+1}/{max_iterations} 轮迭代]", flush=True)
            print("Agent is thinking...", flush=True)

            try:
                response = self.llm.chat(self.messages)
                if not response or "Error" in response:
                    print(f"LLM 调用失败: {response}", flush=True)
                    break
            except Exception as e:
                print(f"LLM 通信异常: {e}", flush=True)
                break

            print(f"\n--- Agent Response ---\n{response}\n------------------\n", flush=True)
            self.messages.append({"role": "assistant", "content": response})

            action = self.parse_action(response)

            if action and action.get("name") == "propose_and_run":
                args      = action.get("args", {})
                new_id    = args.get("new_exp_id")  or args.get("experiment_id")
                parent_id = args.get("parent_id")   or args.get("parent_exp_id")
                changes   = args.get("changes")     or args.get("modifications") or args.get("params")
                rationale = args.get("rationale", "未提供理由")

                # 安全检查：禁止 finetune 官方权重
                if changes and changes.get("init_from") in ("finetune", "pretrained"):
                    print("检测到 init_from=finetune，已强制改为 scratch。", flush=True)
                    changes["init_from"] = "scratch"
                    changes.pop("init_ckpt", None)

                if not new_id or not parent_id or not changes:
                    err = (f"Action 缺失参数: new_exp_id={new_id}, "
                           f"parent_id={parent_id}, changes={changes is not None}")
                    print(err, flush=True)
                    self.messages.append({
                        "role": "user",
                        "content": f"Observation: {err}. 请严格遵守参数模板。"
                    })
                    continue

                print(f"执行实验: {new_id} (基于 {parent_id})", flush=True)
                print(f"理由: {rationale}", flush=True)

                try:
                    observation = self.tools.propose_and_run(
                        new_exp_id=new_id,
                        parent_id=parent_id,
                        changes=changes,
                        rationale=rationale
                    )
                    print(f"观察结果: {observation}", flush=True)

                    # 追加实验排行榜和当前最优信息
                    all_exps = self.tracker.list_all(limit=20)
                    completed = [e for e in all_exps if e.get('status') == 'completed'
                                 and e.get('exp_id') != 'v0_official'
                                 and e.get('total_score') is not None]
                    completed.sort(key=lambda x: x.get('total_score', 0), reverse=True)

                    leaderboard = "\n【当前实验排行榜（自训）】\n"
                    for rank, e in enumerate(completed[:5], 1):
                        leaderboard += (
                            f"  #{rank} {e['exp_id']}: "
                            f"Seg1={e.get('seg1_score',0):.2f} "
                            f"Seg2={e.get('seg2_score',0):.2f} "
                            f"Seg3={e.get('seg3_score',0):.2f} "
                            f"Total={e.get('total_score',0):.4f}\n"
                        )

                    best = self.tracker.get_best_experiment(exclude_official=True)
                    if best:
                        leaderboard += (
                            f"\n当前最优实验: {best['exp_id']} "
                            f"(Total={best.get('total_score',0):.4f})，"
                            f"请以此为 parent_id 继续优化。"
                        )
                        # 追加最优实验的完整配置
                        try:
                            import yaml
                            cfg_path = best.get('config_path', '')
                            if cfg_path and Path(cfg_path).exists():
                                with open(cfg_path) as f:
                                    cfg = yaml.safe_load(f)
                                model_cfg = cfg.get('model', {})
                                train_cfg = cfg.get('training', {})
                                leaderboard += f"\n\n【{best['exp_id']} 当前配置】"
                                arch = model_cfg.get('architecture', 'FNO1d')
                                if arch == 'DeepONet1d':
                                    leaderboard += f"\n  模型: hidden_dim={model_cfg.get('hidden_dim')} p_dim={model_cfg.get('p_dim')} branch_layers={model_cfg.get('branch_layers')} trunk_layers={model_cfg.get('trunk_layers')}"
                                    leaderboard += f"\n  训练: lr={train_cfg.get('lr')} epochs={train_cfg.get('epochs')} batch_size={train_cfg.get('batch_size')} ics_weight={train_cfg.get('ics_weight')} weight_decay={train_cfg.get('weight_decay')}"
                                else:
                                    leaderboard += f"\n  模型: modes={model_cfg.get('modes')} width={model_cfg.get('width')} n_layers={model_cfg.get('n_layers')} window_size={model_cfg.get('window_size')}"
                                    leaderboard += f"\n  训练: lr={train_cfg.get('lr')} epochs={train_cfg.get('epochs')} batch_size={train_cfg.get('batch_size')} pushforward_steps={train_cfg.get('pushforward_steps')} pushforward_weight={train_cfg.get('pushforward_weight')} weight_decay={train_cfg.get('weight_decay')}"
                        except Exception:
                            pass

                    obs_content = f"Observation: {observation}\n{leaderboard}"
                    print(f"Observation sened to agent:\n{obs_content}", flush=True)
                    self.messages.append({
                        "role": "user",
                        "content": obs_content
                    })
                except Exception as e:
                    err = f"Tool ERROR: {str(e)}"
                    print(err, flush=True)
                    self.messages.append({
                        "role": "user",
                        "content": f"Observation: {err}"
                    })
            else:
                print("Agent proposed no legal tool_call，cycle ended。", flush=True)
                break


if __name__ == "__main__":
    try:
        current_file = Path(__file__).resolve()
        root_path    = current_file.parents[2]
        conf_path    = root_path / "conf" / "default.yaml"

        print(f"loading config files: {conf_path}", flush=True)
        if not conf_path.exists():
            print(f"no config found, path: {conf_path.absolute()}", flush=True)
            sys.exit(1)

        with open(conf_path, "r") as f:
            config = yaml.safe_load(f)

        if not torch.cuda.is_available():
            print("WARNING: no available GPU", flush=True)

        orchestrator = Orchestrator(config)
        orchestrator.run(max_iterations=8)

    except KeyboardInterrupt:
        print("\n强行停止运行。")
    except Exception as e:
        import traceback
        print("\n发生未捕获的致命错误:")
        traceback.print_exc()
        sys.exit(1)


