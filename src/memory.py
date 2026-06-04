from abc import abstractmethod
import json
import os
import threading

from src.message import Msg


class MemoryBase:
    """
    记忆基类
    """

    @abstractmethod
    async def add(self, msg: Msg | list[Msg], allow_duplicate: bool = False) -> None:
        """
        添加消息到记忆
        :param msg: 要添加的消息
        :param allow_duplicate: 是否允许重复消息（基于ID）
        """
        pass

    @abstractmethod
    async def get_memory(self) -> list[Msg]:
        """
        获取记忆中的所有消息
        :return: 消息列表
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        清空记忆
        :return:
        """
        pass

    @abstractmethod
    async def size(self) -> int:
        """
        获取记忆中消息的数量
        :return:
        """
        pass

class InMemoryMemory(MemoryBase):
    """
    基于内存的简单记忆实现
    """
    def __init__(self) -> None:
        """
        初始化记忆对象
        """
        self.content: list[Msg] = []

    async def add(
        self,
        msg: Msg | list[Msg] | None,
        allow_duplicate: bool = False,
    ) -> None:
        """
        添加消息到记忆
        :param msg: 要添加的消息
        :param allow_duplicate: 是否允许重复消息（基于ID）
        """
        if msg is None:
            return

        if isinstance(msg, Msg):
            message = [msg]
        else:
            message = list(msg)

        if not allow_duplicate:
            existing_ids = {msg.id for msg in self.content}
            message = [msg for msg in message if msg.id not in existing_ids]

        self.content.extend(message)

    async def get_memory(self) -> list[Msg]:
        """
        获取所有记忆消息
        :return:
        """
        return self.content

    async def clear(self) -> None:
        """
        清空记忆
        :return:
        """
        self.content = []

    async def size(self) -> int:
        """
        获取消息数量
        :return:
        """
        return len(self.content)

class FileMemory(MemoryBase):
    """
    基于 JSON 文件的消息持久化记忆实现

    文件格式:
    {
        "_counter": 15,       // 全局消息计数器，进程重启后继续递增
        "messages": [...]     // 消息列表，每项为 Msg.to_dict() 的结果
    }

    兼容旧格式（纯数组），加载时会自动转换为新格式。
    """

    def __init__(self, file_path: str = "agent_memory.json") -> None:
        self.file_path = file_path
        self._lock = threading.Lock()

    async def _load(self) -> list[Msg]:
        if not os.path.exists(self.file_path):
            return []

        import src.message as message_mod

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            counter = data.get("_counter", 0)
            message_mod._counter = max(message_mod._counter, counter)
            items = data.get("messages", [])
        else:
            # 兼容旧格式（纯数组），从中推导 counter
            items = data
            if items:
                message_mod._counter = max(message_mod._counter, max(
                    item.get("id", 0) for item in items
                ))

        return [Msg.from_dict(item) for item in items]

    async def _save(self, msgs: list[Msg]) -> None:
        import src.message as message_mod

        data = {
            "_counter": message_mod._counter,
            "messages": [msg.to_dict() for msg in msgs],
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def add(
        self,
        msg: Msg | list[Msg] | None,
        allow_duplicate: bool = False,
    ) -> None:
        if msg is None:
            return

        if isinstance(msg, Msg):
            messages = [msg]
        else:
            messages = list(msg)

        with self._lock:
            existing = await self._load()

            if not allow_duplicate:
                existing_ids = {m.id for m in existing}
                messages = [m for m in messages if m.id not in existing_ids]

            existing.extend(messages)
            await self._save(existing)

    async def get_memory(self) -> list[Msg]:
        return await self._load()

    async def clear(self) -> None:
        with self._lock:
            await self._save([])

    async def size(self) -> int:
        msgs = await self._load()
        return len(msgs)