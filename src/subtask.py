from dataclasses import dataclass, field
from typing import Literal

from src.message import Msg

TaskStatus = Literal["pending", "running", "completed", "failed"]


@dataclass
class SubTask:
    """
    子任务数据类

    编排者通过 LLM 拆解出的单个子任务。
    包含任务描述、分配给哪个 Worker、依赖关系、执行结果等信息。
    """

    id: int
    """子任务唯一标识"""

    desc: str
    """子任务描述（作为 Worker.reply() 的输入消息）"""

    worker: str
    """目标 Worker 的名称（对应 WorkerRegistry 中注册的 worker）"""

    depends_on: list[int] = field(default_factory=list)
    """前置依赖的子任务 ID 列表。空列表表示无依赖，可立即执行。"""

    status: TaskStatus = "pending"
    """当前状态：pending / running / completed / failed"""

    result: Msg | None = None
    """Worker 执行完成后返回的结果消息"""

    retries: int = 0
    """已重试次数"""

    max_retries: int = 2
    """最大重试次数。设置为 0 表示失败后不重试。"""

    error: str | None = None
    """失败时的错误信息（重试耗尽后填充）"""
