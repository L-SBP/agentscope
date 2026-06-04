#
from abc import abstractmethod
from collections.abc import AsyncGenerator

from src.formatter import FormatterBase
from src.llm import ChatModelBase, ChatResponse, ChatUsage
from src.memory import MemoryBase, InMemoryMemory
from src.message import Msg, ToolUseBlock, ToolResultBlock
from src.tools import Toolkit, ToolResponse


class AgentBase:
    """
    智能体基类
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    async def reply(
        self,
        msg: Msg | list[Msg] | None = None,
    ) -> Msg:
        """
        生成回复消息
        :param msg: 输入消息
        :return: 回复消息
        """
        pass

    @abstractmethod
    async def observe(
        self,
        msg: Msg | list[Msg] | None
    ) -> None:
        """
        观察消息，不产生回复
        :param msg: 要观察的消息
        :return:
        """
        pass

    async def __call__(
        self,
        msg: Msg | list[Msg] | None = None,
    ) -> Msg:
        return await self.reply(msg)

class UserAgent(AgentBase):
    """
    用户智能体 — 从命令行获取用户输入
    """

    def __init__(self, name: str = "User") -> None:
        super().__init__(name)

    async def reply(
        self,
        msg: Msg | list[Msg] | None = None,
    ) -> Msg:
        if isinstance(msg, Msg):
            text = msg.get_text_content()
            if text:
                print()
        elif isinstance(msg, list):
            for m in msg:
                text = m.get_text_content()
                if text:
                    print()

        user_input = input(f"{self.name}: ")

        return Msg(
            name=self.name,
            content=user_input,
            role="user",
        )

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        pass


class ReActAgent(AgentBase):
    """
    ReAct 智能体，实现Reasoning + Action 循环

    组件：
    llm: LLM大模型，用于推理
    memory: 记忆，储存对话历史
    toolkit: 工具集，提供可调用的函数

    工作流程：
    用户输入 -> Memory 储存 -> LLM 推理 -> 有工具调用？
    -> 是 -> 执行工具 -> 存储结果 -> 继续推理/跳出循环
    -> 否 -> 存储结果 -> 继续推理
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: ChatModelBase,
        formatter: FormatterBase,
        toolkit: Toolkit| None = None,
        memory: MemoryBase | None = None,
        max_iters: int = 10,
        verbose: bool = True,
    ) -> None:
        """
        初始化 ReAct 智能体
        :param name: 智能体名称
        :param sys_prompt: 系统提示词，定义智能体的角色和行为
        :param model: LLM 模型
        :param formatter: 消息格式化器
        :param toolkit: 工具集
        :param memory: 记忆模块
        :param max_iters: 最大推理
        :param verbose: 是否打印详细日志。作为 Orchestrator 的 Worker 时建议设为 False
        """
        super().__init__(name)
        self.sys_prompt = sys_prompt
        self.model = model
        self.formatter = formatter
        self.toolkit = toolkit or Toolkit()
        self.memory = memory or InMemoryMemory()
        self.max_iters = max_iters
        self.verbose = verbose

    async def reply(
        self,
        msg: Msg | list[Msg] | None = None,
    ) -> Msg:
        """
        生成回复，ReAct 循环的主逻辑
        :param msg:
        :return:
        """
        # 1 储存输入消息
        await self.memory.add(msg)

        # 2 ReAct 循环
        for _ in range(self.max_iters):
            # 推理步骤
            response_msg = await self._reasoning()

            # 检查是否有工具调用
            tool_use_blocks = response_msg.get_content_block("tool_use")

            if not tool_use_blocks:
                # 没有工具调用，直接返回
                return response_msg

            # 执行工具调用
            for tool_call in tool_use_blocks:
                await self._acting(tool_call)

        # 超过最大迭代，强制总结
        return await self._summarize()

    async def _reasoning(self) -> Msg:
        """
        推理步骤，调用 LLM 生成响应
        :return:
        """
        # 构成消息列表
        msgs = [
            Msg(name="system", content=self.sys_prompt, role="system"),
            *await self.memory.get_memory(),
        ]

        # 格式化消息
        formatted_msg = await self.formatter.format(msgs)

        # 获取工具 schema
        tools = self.toolkit.get_json_schemas() or None

        # 调用模型
        response = await self.model(
            messages=formatted_msg,
            tools=tools,
            tool_choice="auto" if tools else None,
        )

        # 处理响应
        if isinstance(response, AsyncGenerator):
            final_response = None
            async for chunk in response:
                final_response = chunk
            response = final_response

        response_msg = Msg(
            name=self.name,
            content=list(response.content) if response else [],
            role="assistant",
        )
        await self.memory.add(response_msg)

        self._print_response(response_msg)

        if response and response.usage:
            self._print_token_usage(response.usage)

        return response_msg

    async def _acting(self, tool_call: ToolUseBlock) -> None:
        """
        行动步骤，执行工具调用
        :param tool_call:
        :return:
        """
        # 打印工具调用日志
        self._print_tool_call(tool_call)

        # 执行工具
        tool_result = await self.toolkit.call_tool_function(tool_call)

        # 构造工具结果消息
        result_msg = Msg(
            name="system",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    output=tool_result.content,
                )
            ],
            role="system",
        )

        # 储存到记忆
        await self.memory.add(result_msg)

        # 打印工具结果
        self._print_tool_result(tool_call, tool_result)

    async def _summarize(self) -> Msg:
        """
        超过最大迭代次数时的总结
        :return:
        """
        # 添加提示消息
        hint_msg = Msg(
            name="system",
            content="你已经达到最大迭代次数，请直接给出总结性回答。",
            role="user",
        )

        msgs = [
            Msg(name="system", content=self.sys_prompt, role="system"),
            *await self.memory.get_memory(),
            hint_msg,
        ]

        await self.memory.add(hint_msg)

        # 格式化并调用模型（不使用工具）
        formatted_msg = await self.formatter.format(msgs)
        response = await self.model(messages=formatted_msg)

        # 处理流式响应
        if isinstance(response, AsyncGenerator):
            final_response = None
            async for chunk in response:
                final_response = chunk
            response = final_response

        response_msg = Msg(
            name=self.name,
            content=list(response.content) if response else [],
            role="assistant",
        )

        await self.memory.add(response_msg)

        self._print_response(response_msg)

        return response_msg

    async def observe(self, msg: Msg | list[Msg] | None = None) -> None:
        """
        观察消息，存入记忆但不产生回复
        :param msg:
        :return:
        """
        await self.memory.add(msg)

    async def handle_interrupt(
        self,
    ) -> Msg:
        """
        处理用户中断
        :return:
        """
        resource = Msg(
            name=self.name,
            content="我注意到您中断了我的执行。请问需要我做什么？",
            role="assistant",
            metadata={"_is_interrupted": True}
        )

        await self.memory.add(resource)

        # 打印响应
        print(f"\n{self.name}: {resource.get_text_content()}")

        return resource

    def _print_llm_request(self, messages: list[dict], tools: list[dict] | None) -> None:
        """打印 LLM 请求日志"""
        if not self.verbose:
            return
        import os

        # 检查是否启用详细日志
        verbose = os.environ.get("NANO_AGENTSCOPE_VERBOSE", "0") == "1"

        if not verbose:
            return

        print("\n" + "=" * 80)
        print("🤖 [LLM 请求]")
        print("=" * 80)

        # 打印消息列表
        print(f"\n📝 消息数量: {len(messages)}")
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # 截取内容预览
            if isinstance(content, str):
                preview = content[:200] + "..." if len(content) > 200 else content
            elif isinstance(content, list):
                preview = f"[{len(content)} 个内容块]"
            else:
                preview = str(content)[:200]

            print(f"  {i}. [{role}] {preview}")

        # 打印工具信息
        if tools:
            print(f"\n🔧 可用工具: {len(tools)}")
            for tool in tools:
                func = tool.get("function", {})
                print(f"  - {func.get('name', 'unknown')}: {func.get('description', '')[:100]}")
        else:
            print("\n🔧 可用工具: 无")

        print("=" * 80)

    def _print_tool_call(self, tool_call: ToolUseBlock) -> None:
        """打印工具调用日志"""
        if not self.verbose:
            return
        import json
        import os

        env_verbose = os.environ.get("NANO_AGENTSCOPE_VERBOSE", "0") == "1"

        if env_verbose:
            print(f"\n🔧 [调用工具] {tool_call['name']}")
            print(f"  参数: {json.dumps(tool_call.get('args', {}), ensure_ascii=False, indent=2)}")
        else:
            # 简洁模式
            params_str = json.dumps(tool_call.get('args', {}), ensure_ascii=False)
            if len(params_str) > 100:
                params_str = params_str[:100] + "..."
            print(f"  [调用工具] {tool_call['name']}: {params_str}")

    def _print_token_usage(self, usage: ChatUsage) -> None:
        """打印 Token 使用统计"""
        if not self.verbose:
            return
        import os

        verbose = os.environ.get("NANO_AGENTSCOPE_VERBOSE", "0") == "1"

        if not verbose:
            return

        print(f"\n📊 [Token 使用]")
        print(f"  输入: {usage.input_token} tokens")
        print(f"  输出: {usage.output_token} tokens")
        print(f"  总计: {usage.input_token + usage.output_token} tokens")
        print(f"  耗时: {usage.time:.2f}s")

    def _print_streaming(self, chunk: ChatResponse) -> None:
        """打印流式响应"""
        text_blocks = [b for b in chunk.content if b.get("type") == "text"]
        if text_blocks:
            text = text_blocks[-1].get("text", "")
            print(f"\r{self.name}: {text}", end="", flush=True)

    def _print_response(self, msg: Msg) -> None:
        """打印响应消息"""
        if not self.verbose:
            return
        text = msg.get_text_content()
        if text:
            print(f"{msg.name}: {text}")

        # 打印工具调用
        for block in msg.get_content_block("tool_use"):
            print(f"  [调用工具] {block['name']}({block.get('input', {})})")

    def _print_tool_result(
            self,
            tool_call: ToolUseBlock,
            result: ToolResponse,
    ) -> None:
        """打印工具执行结果"""
        if not self.verbose:
            return
        import os

        text = ""
        for block in result.content:
            if block.get("type") == "text":
                text += block.get("text", "")

        # 从环境变量读取最大长度配置，默认 2000，0 表示不截断
        max_length_str = os.environ.get("NANO_AGENTSCOPE_LOG_MAX_LENGTH", "2000")
        try:
            max_length = int(max_length_str)
        except ValueError:
            max_length = 2000

        # 根据配置决定是否截断
        if 0 < max_length < len(text):
            print(f"  [工具结果] {tool_call['name']}:")
            print(f"    {text[:max_length]}")
            print(f"    ... (已截断，总长度: {len(text)} 字符，设置 NANO_AGENTSCOPE_LOG_MAX_LENGTH=0 查看完整日志)")
        else:
            # 不截断或文本长度在限制内
            print(f"  [工具结果] {tool_call['name']}: {text}")


class OrchestratorAgent(AgentBase):
    """
    OrchestratorAgent 是一个代理，用于管理多个代理，并执行任务。
    """