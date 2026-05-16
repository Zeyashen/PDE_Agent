"""
生成 methodology.pdf
不依赖 pandoc/LaTeX，纯 Python fpdf2 实现
"""
from fpdf import FPDF
from pathlib import Path

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "PDE Agent - Methodology", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def title_block(self, title, subtitle=""):
        self.set_fill_color(30, 60, 120)
        self.rect(0, 20, 210, 45, "F")
        self.set_y(28)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, title, align="C", new_x="LMARGIN", new_y="NEXT")
        if subtitle:
            self.set_font("Helvetica", "", 12)
            self.set_text_color(200, 220, 255)
            self.cell(0, 8, subtitle, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_y(72)
        self.set_text_color(0, 0, 0)

    def section(self, title):
        self.ln(4)
        self.set_fill_color(240, 245, 255)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(20, 50, 120)
        self.cell(0, 9, f"  {title}", fill=True,
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def subsection(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 80, 160)
        self.cell(0, 7, f"  {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def body(self, text, indent=8):
        self.set_font("Helvetica", "", 10)
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text, indent=12):
        self.set_font("Helvetica", "", 10)
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, 5.5, f"* {text}")

    def kv(self, key, value, indent=12):
        self.set_x(self.l_margin + indent)
        self.set_font("Helvetica", "B", 10)
        self.cell(38, 6, key + ":")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, value)

    def code_block(self, text, indent=12):
        self.set_fill_color(245, 245, 245)
        self.set_font("Courier", "", 8.5)
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, 5, text, fill=True, border=1)
        self.ln(1)


def build_pdf(out_path):
    pdf = PDF()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.title_block(
        "PDE Agent - Methodology",
        "Autonomous Scientific Research Agent for 1D Burgers Equation"
    )

    # ── Overview ─────────────────────────────────────────────
    pdf.section("1. Overview")
    pdf.body(
        "PDE Agent is an autonomous AI research system designed to optimize neural "
        "operator models for PDE prediction through a closed scientific loop. "
        "The agent continuously proposes, executes, and evaluates experiments without "
        "human intervention, targeting the 1D Burgers equation long-horizon prediction task."
    )
    pdf.body("Task: Given initial 10 steps u(x, t=0.01~0.46s), predict future 190 steps u(x, t=0.46~9.96s).")

    # ── Architecture ─────────────────────────────────────────
    pdf.section("2. Agent Architecture")

    pdf.subsection("2.1 LLM Backend")
    pdf.kv("Model", "Qwen3-14B (locally deployed on Alvis cluster, 4x A40 GPUs)")
    pdf.kv("Thinking Mode", "enable_thinking=True - chain-of-thought via <think> tags")
    pdf.kv("GPU Allocation", "LLM inference: GPU 0 only | DDP training: all 4 GPUs (serial)")
    pdf.kv("Memory", "No cross-round conversation history. Only registry.db persists.")
    pdf.ln(2)

    pdf.subsection("2.2 Orchestrator (Main Loop)")
    pdf.body(
        "The orchestrator controls the iteration loop using a time budget rather than "
        "a fixed iteration count. Default budget: 55 minutes (reserves 5 min buffer "
        "to ensure training time score <= 60 min = 35 pts)."
    )
    pdf.bullet("Build observation: full leaderboard + best config + error diagnostics")
    pdf.bullet("Call Qwen3-14B: parse <think> reasoning + JSON plan")
    pdf.bullet("Execute propose_and_run tool: train + infer + update registry")
    pdf.bullet("Log results: complete leaderboard with Seg1/Seg2/Seg3 scores")
    pdf.bullet("Stop when: remaining time < 8 min OR target score reached")
    pdf.ln(2)

    pdf.subsection("2.3 propose_and_run Tool (Single Unified Tool)")
    pdf.body(
        "The agent uses a single tool that encapsulates the complete experiment pipeline:"
    )
    pdf.bullet("Inherit config from parent experiment, apply changes")
    pdf.bullet("Auto-decide init_from: A-class changes -> scratch, else -> resume")
    pdf.bullet("Launch DDP training (torchrun, 4 GPUs, random port to avoid conflicts)")
    pdf.bullet("Run inference evaluation on val set (last 20 samples)")
    pdf.bullet("Write results to SQLite registry.db (leaderboard data source)")
    pdf.bullet("Update task1_time.csv with train_time and inference_time")
    pdf.ln(2)

    pdf.subsection("2.4 Registry DB (SQLite)")
    pdf.body(
        "All experiments are stored in registry.db. This serves as the sole persistent "
        "memory across rounds - the LLM sees the full leaderboard each iteration "
        "instead of maintaining a conversation history."
    )
    pdf.code_block(
        "experiments table: exp_id, parent_id, status, init_from,\n"
        "  branch_type, branch_depth, branch_width, latent_dim,\n"
        "  lr, epochs, val_mix_ratio,\n"
        "  seg1_score, seg2_score, seg3_score, total_score,\n"
        "  val_loss, train_time, infer_time, hypothesis, conclusion"
    )

    # ── Model ─────────────────────────────────────────────────
    pdf.section("3. Model: DeepONet (Paradigm A)")

    pdf.subsection("3.1 Architecture")
    pdf.body(
        "DeepONet Paradigm A: Direct operator mapping without autoregressive error accumulation. "
        "Unlike FNO (which autoregressively predicts step-by-step and accumulates errors over "
        "190 steps), DeepONet directly maps initial conditions to any future time point."
    )
    pdf.code_block(
        "Branch net: encodes u(x, t=0~0.46s) [10 steps] -> latent vector (B, p)\n"
        "Trunk net:  encodes query coordinates (t, x)   -> basis vectors  (B, N_q, p)\n"
        "Output:     dot(branch, trunk) + bias = u(x,t) at query points"
    )

    pdf.subsection("3.2 Branch Net Variants (explored by Agent)")
    pdf.bullet("MLP: flatten 10 steps -> MLP layers (baseline)")
    pdf.bullet("CNN: 1D convolutions preserving spatial structure (best result: 45.72/100)")
    pdf.bullet("FNO Encoder: Fourier layers for frequency-domain feature extraction")

    pdf.subsection("3.3 Trunk Net")
    pdf.body(
        "Uses Fourier feature embeddings to enhance high-frequency representation, "
        "critical for capturing Burgers shock structures. "
        "Random frequency matrix B is fixed (not trained), input coordinates are "
        "mapped as [sin(Bx), cos(Bx)] before MLP layers."
    )

    # ── Data ─────────────────────────────────────────────────
    pdf.section("4. Data Strategy")

    pdf.subsection("4.1 Training Data Coverage Problem")
    pdf.body(
        "Key insight: the official training set (1D_Burgers_Sols_Nu0.001.hdf5) covers "
        "only t=0~1.95s after downsampling, while inference requires predictions to "
        "t=9.96s (coverage: ~20%). This fundamental mismatch causes poor long-term predictions."
    )

    pdf.subsection("4.2 Solution: Mixed Training Data")
    pdf.bullet("Source A: 9000 samples from training set, t=0~1.95s (40 steps after downsampling)")
    pdf.bullet("Source B: first 80 samples from task1_val.hdf5, t=0~9.96s (200 steps) - CRITICAL")
    pdf.bullet("Eval set: last 20 samples from task1_val.hdf5 (reserved for Agent scoring)")
    pdf.bullet("val_mix_ratio: WeightedRandomSampler controls A/B sampling ratio (B-class param)")

    pdf.subsection("4.3 FNO Baseline Alignment Check")
    pdf.body(
        "At startup, the agent runs the official FNO Nu=0.001 checkpoint on the val set "
        "to verify the scoring mechanism is correct (score alignment check). "
        "This FNO inference is ONLY used for this verification - all subsequent "
        "inference uses DeepONet exclusively."
    )

    # ── Decision Framework ────────────────────────────────────
    pdf.section("5. Decision Framework")

    pdf.subsection("5.1 Parameter Classification")
    pdf.kv("Class A (Structure)", "branch_type, branch_depth, branch_width, trunk_depth,")
    pdf.set_x(pdf.l_margin + 50)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "trunk_width, latent_dim, activation, fourier_features",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(pdf.l_margin + 12)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "  -> Changes require scratch training (topology changed)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)
    pdf.kv("Class B (Training)", "lr, weight_decay, epochs (<=60), batch_size (<=64),")
    pdf.set_x(pdf.l_margin + 50)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "n_query (<=4096), val_mix_ratio (0.3~2.0)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(pdf.l_margin + 12)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "  -> Supports resume from parent checkpoint",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)
    pdf.kv("Class C (Physics)", "physics_weight (PDE residual loss weight)")
    pdf.ln(2)

    pdf.subsection("5.2 Exploration -> Exploitation")
    pdf.bullet("Exploration phase (<3 structures tested): force Class A changes, always scratch")
    pdf.bullet("  Goal: compare mlp / cnn / fno_encoder branch net architectures")
    pdf.bullet("Exploitation phase (>=3 structures): lock best structure, tune B->C params")
    pdf.bullet("  Resume from best checkpoint, focus on val_mix_ratio and lr tuning")
    pdf.ln(2)

    pdf.subsection("5.3 LLM Prompt Design")
    pdf.body(
        "The system prompt is the sole strategy source for the LLM. It contains: "
        "physical background (Burgers shock dynamics, FNO limitations), "
        "parameter classification rules, exploration/exploitation conditions, "
        "analysis of previous experiment results, and a self-reflection template "
        "requiring the LLM to diagnose bottlenecks before proposing changes."
    )

    # ── Results ───────────────────────────────────────────────
    pdf.section("6. Experimental Results (First Run)")

    pdf.body("Three structures tested in exploration phase (all depth=2, width=64):")
    pdf.ln(1)

    # Results table
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(30, 60, 120)
    pdf.set_text_color(255, 255, 255)
    col_w = [30, 20, 20, 20, 20, 25, 30]
    headers = ["Exp ID", "Branch", "Total", "Seg1", "Seg2", "Seg3", "Init"]
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    rows = [
        ["exp_001", "mlp", "27.72", "0.00", "0.07", "55.41", "scratch"],
        ["exp_002", "cnn", "45.72", "1.05", "29.91", "75.96", "scratch"],
        ["exp_003", "fno_enc", "43.65", "0.78", "23.48", "75.16", "scratch"],
    ]
    pdf.set_text_color(0, 0, 0)
    for i, row in enumerate(rows):
        pdf.set_fill_color(245, 248, 255) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.set_font("Helvetica", "B" if i == 1 else "", 9)
        for j, cell in enumerate(row):
            pdf.cell(col_w[j], 6, cell, border=1, fill=True, align="C")
        pdf.ln()
    pdf.ln(3)

    pdf.body(
        "Key finding: CNN branch achieves best total score (45.72). "
        "All models show Seg1 ~ 0, indicating insufficient short-term prediction accuracy. "
        "Root cause: network too shallow (depth=2, width=64). "
        "Next step: deepen CNN to depth=4, width=128 with val_mix_ratio=0.5."
    )

    # ── Scoring ───────────────────────────────────────────────
    pdf.section("7. Scoring Mechanism")

    pdf.subsection("7.1 Prediction Accuracy Score (max 75 pts)")
    pdf.bullet("Seg1 (steps 10-57,  25%): 100 * exp(-20 * Rel-MSE)")
    pdf.bullet("Seg2 (steps 57-105, 25%): 100 * exp(-10 * Rel-MSE)")
    pdf.bullet("Seg3 (steps 105-200,50%): max(Lorentzian, Frechet)")
    pdf.bullet("  Lorentzian = 100 / (1 + 10 * RMSE)")
    pdf.bullet("  Frechet    = 50 * exp(-FD^2)")
    pdf.bullet("Precision score = segment total (0-100) * 0.75")
    pdf.ln(2)

    pdf.subsection("7.2 Time Scores")
    pdf.bullet("Training time score (max 35): <=60min=35, <=120min=25, <=300min=20, <=500min=10")
    pdf.bullet("Inference time score (max 40): 0min=40, linear decay to 0 at 2min, >2min=0")
    pdf.bullet("DeepONet Paradigm A has near-zero inference overhead (no autoregressive loop)")

    # ── Conclusion ────────────────────────────────────────────
    pdf.section("8. Conclusion")
    pdf.body(
        "PDE Agent demonstrates that an LLM-driven autonomous research loop can "
        "effectively explore neural operator architectures for PDE prediction. "
        "The key innovation is combining DeepONet Paradigm A (eliminating autoregressive "
        "error accumulation) with a time-budget-controlled optimization loop, "
        "where the LLM reasons about physical bottlenecks and proposes structured "
        "experiments with scientific hypotheses."
    )
    pdf.body(
        "The framework is designed for extensibility: Task 2 (multi-Nu generalization) "
        "can leverage the same DeepONet architecture where the branch net implicitly "
        "encodes viscosity information from initial conditions."
    )

    pdf.output(str(out_path))
    print(f"PDF 生成成功: {out_path} ({Path(out_path).stat().st_size//1024} KB)")


if __name__ == "__main__":
    out = Path("/home/claude/methodology.pdf")
    build_pdf(out)