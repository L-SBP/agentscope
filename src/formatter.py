import json
from abc import abstractmethod
from typing import Any

from src.message import Msg


class FormatterBase:
    """
    将Msg转为 API 要求的格式
    """
    @abstractmethod
    async def format(self, msgs: list[Msg]):
        """
        将消息列表转成 API 要求的格式
        :param self:
        :param msgs:
        :return:
        """

    @staticmethod
    def _assert_msgs(msgs: list[Msg]):
        """
        验证msgs 是不是Msg列表
        :param msgs:
        :return:
        """
        if not isinstance(msgs, list):
            raise TypeError('msgs must be a list')
        for msg in msgs:
            if not isinstance(msg, Msg):
                raise TypeError('msg must be a Msg')


class OpenAIFormatter(FormatterBase):
    """
    OpenAI API 格式转换器
    """

    async def format(self, msgs: list[Msg]) -> list[dict[str, Any]]:
        """
        格式化消息列表
        :param msgs:
        :return:
        """
        self._assert_msgs(msgs)

        formatted_msgs = []

        for msg in msgs:
            content_blocks = []
            tool_calls = []

            for block in msg.get_content_block():
                block_type = block.get('type')

                if block_type == 'text':
                    # 文本块
                    content_blocks.append({
                        "type": "text",
                        "text": block.get('text'),
                    })

                elif block_type == 'tool_use':
                    # 工具调用块
                    tool_calls.append({
                        "id": block.get('id'),
                        "type": "function",
                        "function": {
                            "name": block.get('name'),
                            "arguments": json.dumps(
                                block.get("args", {}),
                                ensure_ascii=False
                            )
                        }
                    })

                elif block_type == 'tool_result':
                    # 工具结果块，单个 tool 消息
                    output = block.get('output', "")
                    if isinstance(output, list):
                        texts = [
                            b['text'] for b in output
                            if isinstance(b, dict) and b.get('type') == 'text'
                        ]
                        output = '\n'.join(texts)

                    formatted_msgs.append({
                        "role": "tool",
                        "tool_call_id": block.get("id"),
                        "name": block.get("name", ""),
                        "content": str(output),
                    })

            if content_blocks or tool_calls:
                openai_msg = {
                    "role": msg.role,
                    "name": msg.name,
                }

                if content_blocks:
                    openai_msg["content"] = content_blocks
                else:
                    openai_msg["content"] = None

                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls

                # 回传推理/思考内容（Kimi k2.x / DeepSeek 等需要 reasoning_content）
                if msg.reasoning_content is not None:
                    openai_msg["reasoning_content"] = msg.reasoning_content

                formatted_msgs.append(openai_msg)

        return formatted_msgs