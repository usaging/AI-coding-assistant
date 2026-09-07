"""服务层：会话管理与持久化。"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    working_dir: str = "."
    created_at: float = field(default_factory=time.time)
    history: list[dict] = field(default_factory=list)  # 会话轮次（Q/A）

    def add_turn(self, question: str, answer: str) -> None:
        self.history.append({"question": question, "answer": answer, "ts": time.time()})

    def to_dict(self) -> dict:
        return asdict(self)

class SessionStore:
    """会话 JSON 落盘（.aiassist/sessions/<id>.json）。"""

    def __init__(self, base_dir: str = ".aiassist/sessions"):
        self.dir = Path(base_dir)

    def save(self, session: Session) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / f"{session.id}.json"
        p.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def load(self, session_id: str) -> Session | None:
        p = self.dir / f"{session_id}.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return Session(
            id=data["id"], working_dir=data.get("working_dir", "."),
            created_at=data.get("created_at", time.time()),
            history=data.get("history", []),
        )
