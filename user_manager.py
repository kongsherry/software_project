"""用户管理模块 — 注册、登录、管理员用户管理。

使用 JSON 文件持久化存储用户数据，密码经 werkzeug 哈希处理。
支持普通用户和管理员两种角色。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from werkzeug.security import generate_password_hash, check_password_hash


DEFAULT_USERS_PATH = os.getenv("ANN_USERS_PATH", "data/users.json")
DEFAULT_SESSION_SECRET = os.getenv("ANN_SESSION_SECRET", "change-this-secret-key-in-production")


@dataclass
class User:
    username: str
    password_hash: str
    role: str = "user"  # "user" 或 "admin"
    created_at: float = field(default_factory=time.time)
    is_active: bool = True

    def verify_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "role": self.role,
            "created_at": self.created_at,
            "is_active": self.is_active,
        }

    def to_safe_dict(self) -> dict[str, Any]:
        """不暴露密码哈希的安全字典"""
        return self.to_dict()


class UserManager:
    """用户数据管理器，线程安全，JSON 文件持久化。"""

    def __init__(self, users_path: str | Path = DEFAULT_USERS_PATH) -> None:
        self._path = Path(users_path)
        self._lock = threading.Lock()
        self._users: dict[str, User] = {}
        self._load()

    def _load(self) -> None:
        """从 JSON 文件加载用户数据。"""
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._users = {}
            self._save()
            return

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._users = {}
            for username, data in raw.items():
                self._users[username] = User(
                    username=data["username"],
                    password_hash=data["password_hash"],
                    role=data.get("role", "user"),
                    created_at=data.get("created_at", time.time()),
                    is_active=data.get("is_active", True),
                )
        except (json.JSONDecodeError, KeyError) as exc:
            raise RuntimeError(f"用户数据文件损坏: {exc}") from exc

    def _save(self) -> None:
        """将用户数据写入 JSON 文件。"""
        raw = {}
        for username, user in self._users.items():
            raw[username] = {
                "username": user.username,
                "password_hash": user.password_hash,
                "role": user.role,
                "created_at": user.created_at,
                "is_active": user.is_active,
            }
        self._path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ───────────────────────── 公共 API ─────────────────────────

    def register(self, username: str, password: str, role: str = "user") -> User:
        """注册新用户。

        Args:
            username: 用户名（唯一）
            password: 明文密码
            role: 角色，'user' 或 'admin'

        Returns:
            创建的用户对象

        Raises:
            ValueError: 用户名已存在或参数不合法
        """
        username = username.strip()
        if not username:
            raise ValueError("用户名不能为空")
        if len(username) < 2:
            raise ValueError("用户名至少需要 2 个字符")
        if len(password) < 6:
            raise ValueError("密码至少需要 6 个字符")

        with self._lock:
            if username in self._users:
                raise ValueError(f"用户名 '{username}' 已存在")

            user = User(
                username=username,
                password_hash=generate_password_hash(password),
                role=role,
            )
            self._users[username] = user
            self._save()
            return user

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """验证用户凭据。

        Args:
            username: 用户名
            password: 明文密码

        Returns:
            验证通过返回 User 对象，否则返回 None
        """
        username = username.strip()
        with self._lock:
            user = self._users.get(username)
            if user is None:
                return None
            if not user.is_active:
                return None
            if user.verify_password(password):
                return user
            return None

    def get_user(self, username: str) -> Optional[User]:
        """获取用户对象。"""
        with self._lock:
            return self._users.get(username)

    def list_users(self) -> list[dict[str, Any]]:
        """返回所有用户的安全信息列表（仅管理员可用）。"""
        with self._lock:
            return [
                user.to_safe_dict()
                for user in sorted(
                    self._users.values(),
                    key=lambda u: u.created_at,
                    reverse=True,
                )
            ]

    def delete_user(self, username: str) -> bool:
        """删除用户（不能删除自身或最后一个管理员）。"""
        with self._lock:
            if username not in self._users:
                return False

            # 检查是否为最后一个管理员
            if self._users[username].role == "admin":
                admin_count = sum(
                    1 for u in self._users.values() if u.role == "admin"
                )
                if admin_count <= 1:
                    raise ValueError("不能删除最后一个管理员账号")

            del self._users[username]
            self._save()
            return True

    def change_role(self, username: str, new_role: str) -> User:
        """修改用户角色。"""
        if new_role not in ("user", "admin"):
            raise ValueError("角色必须是 'user' 或 'admin'")

        with self._lock:
            user = self._users.get(username)
            if user is None:
                raise KeyError(f"用户 '{username}' 不存在")

            # 如果降级最后一个管理员，阻止操作
            if user.role == "admin" and new_role == "user":
                admin_count = sum(
                    1 for u in self._users.values() if u.role == "admin"
                )
                if admin_count <= 1:
                    raise ValueError("不能降级最后一个管理员账号")

            user.role = new_role
            self._save()
            return user

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """修改用户密码。"""
        if len(new_password) < 6:
            raise ValueError("新密码至少需要 6 个字符")

        with self._lock:
            user = self._users.get(username)
            if user is None:
                return False
            if not user.verify_password(old_password):
                return False
            user.password_hash = generate_password_hash(new_password)
            self._save()
            return True

    def ensure_admin_exists(self, username: str = "admin", password: str = "admin123") -> None:
        """确保至少有一个管理员账号存在（用于首次初始化）。"""
        with self._lock:
            has_admin = any(u.role == "admin" for u in self._users.values())
            if not has_admin:
                # 检查是否已存在同名普通用户
                if username in self._users:
                    self._users[username].role = "admin"
                else:
                    self._users[username] = User(
                        username=username,
                        password_hash=generate_password_hash(password),
                        role="admin",
                    )
                self._save()
