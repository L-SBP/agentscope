from datetime import datetime
from typing import Union, Sequence

from typing_extensions import TypedDict, Literal, Required

class TextBlock(TypedDict):
    """
    文本内容块
    """
    type: Required[Literal["text"]]
    text: str

class ToolUseBlock(TypedDict):
    """
    工具调用块
    """
    type: Required[Literal["tool_use"]]
    id: Required[str]   # 调用的唯一标识
    name: Required[str] # 工具函数名
    args: Required[dict[str, object]]   # 调用的参数

class ToolResultBlock(TypedDict):
    """
    调用工具返回的结果
    """
    type: Required[Literal["tool_result"]]
    id: Required[str]
    name: Required[str]
    output: Required[str | list]


ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock]

_counter = 0
class Msg:
    """
    消息类
    """
    def __init__(
        self,
        name: str,
        content: str | Sequence[ContentBlock],
        role: Literal["user", "system", "assistant"],
        metadata: dict | None = None,
        timestamp: str | None = None,
    ) -> None:
        """
        初始化消息对象
        :param name: 发送者名称，"user"是用户，"system"是工具返回，"assistant"是模型返回
        :param content:
        :param role:
        :param timestamp:
        """
        global _counter
        _counter += 1

        self.name = name
        self.content = content
        self.role = role

        self.id = _counter
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_text_content(self, separator: str = '\n') -> str | None:
        """
        获取消息中的纯文本内容
        :param separator:
        :return:
        """
        if isinstance(self.content, str):
            return self.content

        texts = []
        for block in self.content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))

        return separator.join(texts) if texts else None

    def get_content_block(self, block_type: Literal["text", "tool_use", "tool_result"] | None = None) -> Sequence[ContentBlock]:
        """
        获取特定类型的内容块
        :param block_type:
        :return:
        """
        if isinstance(self.content, str):
            blocks = [TextBlock(type="text", text=self.content)]
        else:
            blocks = list(self.content) if self.content else []

        if block_type:
            blocks = [b for b in blocks if b.get("type") == block_type]

        return blocks