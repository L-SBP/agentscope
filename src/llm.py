import json
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, AsyncGenerator

from typing_extensions import Any

from openai import AsyncOpenAI

from src.message import ContentBlock, TextBlock, ToolUseBlock


@dataclass
class ChatUsage:
    """
    Token 使用统计
    """
    input_token: int = 0
    output_token: int = 0
    time: float = 0.0

@dataclass
class ChatResponse:
    """
    模型响应数据结构
    """
    content: list[ContentBlock]
    usage: ChatUsage | None = None
    reasoning_content: str | None = None
    """推理/思考内容（Kimi k2.x / DeepSeek 等模型的 reasoning_content 字段）"""

class ChatModelBase:
    """
    模型基类
    """

    model_name: str
    stream: bool

    def __init__(self, model_name: str, stream: bool):
        self.model_name = model_name
        self.stream = stream

    @abstractmethod
    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "required"] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """
        调用模型生成响应
        :param messages: 消息列表，格式为API要求的格式
        :param tools: 可用工具的 JSON schema列表
        :param tool_choice: 工具选择模式
        :param kwargs: 其他参数
        :return:
        """
        pass

class OpenAIChatModel(ChatModelBase):
    """
    OpenAI Chat API 模型实现
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        初始化 OpenAI 模型
        :param model_name: 模型名称
        :param api_key: API密钥
        :param base_url: API基础URL
        :param stream: 是否流式输出
        :param kwargs: 传给 AsyncOpenAI 的其他参数；extra_body 会被保留用于 API 调用
        """
        super().__init__(model_name, stream)

        self._extra_body = kwargs.pop("extra_body", None)

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "required"] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """
        调用 Chat API
        :param messages: 上下文
        :param tools: 工具 JSON schema
        :param tool_choice: 工具使用
        :param kwargs: 其他参数
        :return:
        """

        start_time = datetime.now()

        extra_body = kwargs.pop("extra_body", self._extra_body) or {}
        if "thinking" in extra_body:
            extra_body["thinking"]["type"] = "enabled"

        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": self.stream,
            **kwargs,
        }
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        # 添加工具相关参数
        if tools:
            request_kwargs["tools"] = tools
        if tool_choice:
            request_kwargs["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**request_kwargs)

        if self.stream:
            return self._parse_stream_response(response, start_time)
        else:
            return self._parse_response(response, start_time)

    def _parse_response(
        self,
        response: Any,
        start_time: datetime,
    ) -> ChatResponse:
        """
        解析非流式 API 响应
        :param self:
        :param response:
        :param start_time:
        :return:
        """
        content_blocks = []
        reasoning_content: str | None = None

        if response.choices:
            choice = response.choices[0]

            if choice.message.content:
                content_blocks.append(
                    TextBlock(type="text", text=choice.message.content)
                )

            # Kimi k2.x / DeepSeek 推理内容（非标准 OpenAI SDK 字段）
            if hasattr(choice.message, "reasoning_content"):
                reasoning_content = getattr(
                    choice.message, "reasoning_content"
                ) or None

            for tool_call in choice.message.tool_calls or []:
                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=tool_call.id,
                        name=tool_call.function.name,
                        args=json.loads(tool_call.function.arguments or "{}"),
                    )
                )
        usage = None
        if response.usage:
            usage=ChatUsage(
                input_token=response.usage.prompt_tokens,
                output_token=response.usage.completion_tokens,
                time=(datetime.now() - start_time).total_seconds(),
            )
        return ChatResponse(
            content=content_blocks,
            usage=usage,
            reasoning_content=reasoning_content,
        )

    async def _parse_stream_response(
        self,
        response: Any,
        start_time: datetime,
    ) -> AsyncGenerator[ChatResponse, None]:
        """
        解析流式 API 响应
        :param response: 模型返回
        :param start_time: 开始时间
        :return:
        """
        usage = None
        text = ""
        reasoning_content = ""
        tool_calls: dict[int, dict] = {}
        async for chunk in response:
            if chunk.usage:
                usage=ChatUsage(
                    input_token=chunk.usage.prompt_tokens,
                    output_token=chunk.usage.completion_tokens,
                    time=(datetime.now() - start_time).total_seconds(),
                )

            if not chunk.choices:
                if usage:
                    yield self._build_stream_response(
                        text, tool_calls, usage, reasoning_content
                    )
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                text += delta.content

            # Kimi k2.x / DeepSeek 推理内容（非标准 OpenAI SDK 字段）
            if hasattr(delta, "reasoning_content"):
                rc = getattr(delta, "reasoning_content")
                if rc:
                    reasoning_content += rc

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    idx = tool_call_delta.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "type": "tool_use",
                            "id": tool_call_delta.id or "",
                            "name": tool_call_delta.function.name or "",
                            "args": tool_call_delta.function.arguments or "",
                        }
                    else:
                        if tool_call_delta.function.name:
                            tool_calls[idx]["name"] = tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            tool_calls[idx]["args"] += tool_call_delta.function.arguments

            yield self._build_stream_response(
                text, tool_calls, usage, reasoning_content
            )


    def _build_stream_response(
        self,
        text: str,
        tool_calls: dict[int, dict],
        usage: ChatUsage | None = None,
        reasoning_content: str = "",
    ) -> ChatResponse:
        """
        构建流式响应的 ChatResponse
        :param self:
        :param text:
        :param tool_calls:
        :param usage:
        :param reasoning_content: 推理/思考内容
        :return:
        """
        content_blocks = []

        if text:
            content_blocks.append(
                TextBlock(type="text", text=text)
            )

        for tool in tool_calls.values():
            try:
                args_dict = json.loads(tool["args"] or "{}")
            except json.decoder.JSONDecodeError:
                args_dict = {}

            content_blocks.append(
                ToolUseBlock(
                    type="tool_use",
                    id=tool["id"],
                    name=tool["name"],
                    args=args_dict,
                )
            )
        return ChatResponse(
            content=content_blocks,
            usage=usage,
            reasoning_content=reasoning_content if reasoning_content else None,
        )