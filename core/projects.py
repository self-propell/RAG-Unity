"""
core/projects.py — 多工程管理器

功能：
  - 注册/删除/列出 Unity 工程
  - 每个工程有独立的 RAG 数据库（ChromaDB collection）
  - 持久化到 projects.json
"""
import os
import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from core import config


@dataclass
class ProjectInfo:
    """工程注册信息"""
    project_id: str
    name: str
    path: str
    db_path: str
    collection_name: str
    unity_version: str = "2022.3 LTS"
    render_pipeline: str = "URP"
    created_at: float = 0.0
    last_indexed_at: float = 0.0
    chunk_count: int = 0
    file_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ProjectInfo":
        return ProjectInfo(**{k: v for k, v in d.items() if k in ProjectInfo.__dataclass_fields__})


def _generate_project_id(path: str) -> str:
    normalized = os.path.normpath(path).replace("\\", "/").lower()
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


class ProjectManager:
    """工程注册表管理器"""

    def __init__(self, db_path: str = None):
        self._db_path = db_path or config.PROJECTS_DB_PATH
        self._projects: dict[str, ProjectInfo] = {}
        self._active_id: Optional[str] = None
        self._load()

    def _load(self):
        if os.path.exists(self._db_path):
            try:
                with open(self._db_path, encoding="utf-8") as f:
                    data = json.load(f)
                for pid, pdata in data.get("projects", {}).items():
                    self._projects[pid] = ProjectInfo.from_dict(pdata)
                self._active_id = data.get("active_project_id")
            except (json.JSONDecodeError, KeyError):
                self._projects = {}

    def _save(self):
        data = {
            "active_project_id": self._active_id,
            "projects": {pid: p.to_dict() for pid, p in self._projects.items()},
        }
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        with open(self._db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_project(
        self,
        path: str,
        name: str = "",
        unity_version: str = "2022.3 LTS",
        render_pipeline: str = "URP",
    ) -> ProjectInfo:
        path = os.path.normpath(os.path.abspath(path))
        if not os.path.isdir(path):
            raise ValueError(f"工程路径不存在: {path}")

        project_id = _generate_project_id(path)
        if project_id in self._projects:
            return self._projects[project_id]

        if not name:
            name = os.path.basename(path)

        db_path = os.path.join(config.CHROMA_DB_ROOT, project_id)
        collection_name = f"unity_{project_id}"

        info = ProjectInfo(
            project_id=project_id,
            name=name,
            path=path,
            db_path=db_path,
            collection_name=collection_name,
            unity_version=unity_version,
            render_pipeline=render_pipeline,
            created_at=time.time(),
        )
        self._projects[project_id] = info

        if self._active_id is None:
            self._active_id = project_id

        self._save()
        return info

    def remove_project(self, project_id: str) -> bool:
        if project_id not in self._projects:
            return False
        del self._projects[project_id]
        if self._active_id == project_id:
            self._active_id = next(iter(self._projects), None)
        self._save()
        return True

    def get_project(self, project_id: str) -> Optional[ProjectInfo]:
        return self._projects.get(project_id)

    def list_projects(self) -> list[ProjectInfo]:
        return list(self._projects.values())

    def find_by_name(self, name: str) -> Optional[ProjectInfo]:
        name_lower = name.lower()
        for p in self._projects.values():
            if name_lower in p.name.lower() or name_lower in p.project_id:
                return p
        return None

    def get_active(self) -> Optional[ProjectInfo]:
        if self._active_id and self._active_id in self._projects:
            return self._projects[self._active_id]
        return None

    def set_active(self, project_id: str) -> bool:
        if project_id not in self._projects:
            return False
        self._active_id = project_id
        self._save()
        return True

    def update_index_stats(self, project_id: str, chunk_count: int, file_count: int):
        p = self._projects.get(project_id)
        if p:
            p.chunk_count = chunk_count
            p.file_count = file_count
            p.last_indexed_at = time.time()
            self._save()


_manager: Optional[ProjectManager] = None


def get_project_manager() -> ProjectManager:
    global _manager
    if _manager is None:
        _manager = ProjectManager()
    return _manager
