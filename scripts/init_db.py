import sqlite3
from pathlib import Path

DB_PATH = Path("/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/guandi_agent/runs/registry.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            exp_id TEXT PRIMARY KEY,
            parent_exp_id TEXT,
            config_path TEXT,
            status TEXT,
            train_time REAL,
            inference_time REAL,
            val_loss REAL,
            seg1_score REAL,
            seg2_score REAL,
            seg3_score REAL,
            total_score REAL,
            rationale TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ 数据库表结构初始化完成（无预置数据）。")

if __name__ == "__main__":
    init_db()