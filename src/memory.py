from abc import abstractmethod

from src.message import Msg


class MemoryBase:
    """
    记忆基类
    """

    @abstractmethod
    async def add(self, msg: Msg | list[Msg]) -> None:
        """
        添加消息到记忆
        :param msg: 要添加的消息
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