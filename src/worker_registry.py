from src.loop_agent import AgentBase


class WorkerRegistry:
    """
    Worker 注册管理器，类比 Toolkit 管理工具。

    每个 Worker 包含：
    - name: 唯一标识，LLM 输出的 worker 字段对应此名称
    - capability: 自然语言能力描述，放在编排者 Prompt 中让 LLM 选择
    - agent: AgentBase 实例（通常是 ReActAgent）

    Usage:
        registry = WorkerRegistry()

        registry.register_worker(
            "搜索专家",
            ReActAgent(...),
            capability="擅长网络搜索和信息检索",
        )
    """

    def __init__(self) -> None:
        self._workers: dict[str, AgentBase] = {}
        self._capabilities: dict[str, str] = {}

    def register_worker(
        self,
        name: str,
        agent: AgentBase,
        capability: str,
    ) -> None:
        if not name.strip():
            raise ValueError("Worker 名称不能为空")
        self._workers[name] = agent
        self._capabilities[name] = capability

    def remove_worker(self, name: str) -> None:
        self._workers.pop(name, None)
        self._capabilities.pop(name, None)

    def get_worker(self, name: str) -> AgentBase:
        if name not in self._workers:
            available = list(self._workers.keys())
            raise KeyError(
                f"Worker '{name}' 未注册。可用 Worker: {available}"
            )
        return self._workers[name]

    def get_capability_text(self) -> str:
        if not self._capabilities:
            return "（暂无可用 Worker）"
        lines = []
        for name, cap in self._capabilities.items():
            lines.append(f"- {name}：{cap}")
        return "\n".join(lines)

    def has_worker(self, name: str) -> bool:
        return name in self._workers

    def list_workers(self) -> list[str]:
        return list(self._workers.keys())

    def clear(self) -> None:
        self._workers.clear()
        self._capabilities.clear()

    def __getitem__(self, name: str) -> AgentBase:
        return self.get_worker(name)

    def __contains__(self, name: str) -> bool:
        return self.has_worker(name)

    def __len__(self) -> int:
        return len(self._workers)

    def __repr__(self) -> str:
        return f"WorkerRegistry(workers={list(self._workers.keys())})"
