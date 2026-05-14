import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

class Tracker:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
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

    def register_experiment(self, exp_id: str, config_path: Path,
                            parent_id: Optional[str] = None, rationale: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO experiments "
                "(exp_id, config_path, parent_exp_id, rationale, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (exp_id, str(config_path), parent_id, rationale,
                 "pending", datetime.now().isoformat())
            )

    def update_metrics(self, exp_id: str, metrics: Dict[str, Any],
                       status: str = "completed"):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE experiments SET
                status=?, train_time=?, inference_time=?, val_loss=?,
                seg1_score=?, seg2_score=?, seg3_score=?, total_score=?
                WHERE exp_id=?
            ''', (
                status,
                metrics.get("train_time"),
                metrics.get("inference_time"),
                metrics.get("val_loss"),
                metrics.get("seg1_score"),
                metrics.get("seg2_score"),
                metrics.get("seg3_score"),
                metrics.get("total_score"),
                exp_id
            ))

    def get_best_experiment(self, exclude_official: bool = True) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if exclude_official:
                cursor.execute(
                    "SELECT * FROM experiments WHERE status='completed' "
                    "AND exp_id NOT LIKE 'v0_official%' "
                    "ORDER BY total_score DESC LIMIT 1"
                )
            else:
                cursor.execute(
                    "SELECT * FROM experiments WHERE status='completed' "
                    "ORDER BY total_score DESC LIMIT 1"
                )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE exp_id=?", (exp_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_all(self, limit: int = 10) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT exp_id, status, seg1_score, seg2_score, seg3_score, total_score "
                "FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
