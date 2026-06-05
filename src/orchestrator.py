import asyncio
import json
import os
from collections.abc import AsyncGenerator

from src.formatter import FormatterBase
from src.llm import ChatModelBase, ChatResponse
from src.memory import MemoryBase, InMemoryMemory
from src.message import Msg
from src.scheduler import TaskScheduler
from src.subtask import SubTask
from src.worker_registry import WorkerRegistry
from src.loop_agent import AgentBase


DEFAULT_PLAN_PROMPT = """你是一个任务编排者，负责将复杂任务拆解成子任务，分派给不同的 Worker 执行。

【可用的 Worker 及其能力】
{worker_capabilities}

【规则】
1. 严禁直接回答用户的问题，只做任务拆解
2. 每个子任务必须指定分配给哪个 Worker
3. 子任务之间如果有依赖关系，标注 depends_on（填写依赖的子任务编号）
4. 拆解 {min_subtasks}-{max_subtasks} 个子任务，过多或过少都不合适

【输出格式】严格按以下 JSON 输出，不要包含任何其他文字：
{{
  "subtasks": [
    {{"id": 1, "desc": "子任务描述", "worker": "Worker名称", "depends_on": []}},
    {{"id": 2, "desc": "子任务描述", "worker": "Worker名称", "depends_on": [1]}}
  ]
}}"""


DEFAULT_SUMMARIZE_PROMPT = """你是一个任务编排者。以下是各 Worker 完成子任务后的结果。

【用户原始问题】
{original_question}

【子任务执行结果】
{worker_results}

请基于以上信息，给出完整、准确的回答。如果某个子任务失败了，请基于已有信息尽力回答，并说明哪些部分因执行失败而缺失。"""


class OrchestratorAgent(AgentBase):
    """
    编排者智能体

    工作流程：
    1. 接收用户消息
    2. 调用 LLM 拆解任务（使用 plan_prompt）
    3. 通过 TaskScheduler 执行子任务
    4. 调用 LLM 汇总结果（使用 summarize_prompt）
    5. 返回最终答案
    """

    def __init__(
        self,
        name: str,
        model: ChatModelBase,
        formatter: FormatterBase,
        workers: WorkerRegistry,
        plan_prompt: str | None = None,
        summarize_prompt: str | None = None,
        memory: MemoryBase | None = None,
        scheduler: TaskScheduler | None = None,
        max_subtasks: int = 10,         # 最多拆解的子任务数量
        min_subtasks: int = 1,          # 最少拆解的子任务数量
        max_parse_retries: int = 3,     # 允许失败的子任务数量
        result_max_length: int = 3000,
    ) -> None:
        super().__init__(name)
        self.model = model
        self.formatter = formatter
        self.workers = workers
        self.plan_prompt = plan_prompt or DEFAULT_PLAN_PROMPT
        self.summarize_prompt = summarize_prompt or DEFAULT_SUMMARIZE_PROMPT
        self.memory = memory or InMemoryMemory()
        self.scheduler = scheduler or TaskScheduler()
        self.max_subtasks = max_subtasks
        self.min_subtasks = min_subtasks
        self.max_parse_retries = max_parse_retries
        self.result_max_length = result_max_length
        self._verbose = os.environ.get("NANO_AGENTSCOPE_VERBOSE", "0") == "1"

    # ── 主流程 ─────────────────────────────────────────────

    async def reply(
        self,
        msg: Msg | list[Msg] | None = None,
    ) -> Msg:
        if isinstance(msg, list):
            msg = msg[-1] if msg else None
        if msg is None:
            return Msg(name=self.name, content="输入为空", role="assistant")

        user_text = (msg.get_text_content() or "").strip()
        if len(user_text) < 5:
            return Msg(
                name=self.name,
                content="请提供更详细的任务描述（至少5个字符）。",
                role="assistant",
            )

        await self.memory.add(msg)

        original_question = user_text

        # 1. 拆解
        self._log(f"\n{'='*60}\n阶段 1/3: 任务拆解\n{'='*60}")
        subtasks = await self._plan(msg)
        self._log(f"拆解出 {len(subtasks)} 个子任务:")
        for st in subtasks:
            self._log(f"  [{st.id}] {st.desc[:60]} -> {st.worker} "
                      f"(depends_on={st.depends_on})")

        # 2. 执行
        self._log(f"\n{'='*60}\n阶段 2/3: 执行子任务\n{'='*60}")
        results = await self._execute_plan(subtasks)
        self._log(f"执行完成，{len(results)}/{len(subtasks)} 个任务有结果")

        # 打印任务执行摘要
        for st in subtasks:
            status = st.status
            icon = "[OK]" if status == "completed" else "[FAIL]"
            result_preview = ""
            if st.result:
                text = st.result.get_text_content() or ""
                result_preview = text[:80].replace("\n", " ")
            print(f"  {icon} 子任务{st.id} [{st.worker}] {st.desc[:40]} -> {result_preview}")

        # 3. 汇总
        self._log(f"\n{'='*60}\n阶段 3/3: 结果汇总\n{'='*60}")
        answer = await self._summarize(msg, results, subtasks)
        await self.memory.add(answer)

        return answer

    # ── 阶段 1: 拆解 ───────────────────────────────────────

    async def _plan(self, msg: Msg) -> list[SubTask]:
        last_error: str | None = None

        for attempt in range(self.max_parse_retries + 1):
            raw_text = await self._call_llm_for_plan(msg, last_error)
            try:
                subtasks = await self._parse_plan(raw_text)
                await self._validate_subtasks(subtasks)
                if last_error:
                    self._log(f"JSON 解析重试成功（第 {attempt} 次）")
                return subtasks
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = f"第 {attempt + 1} 次解析失败: {e}\nLLM 输出: {raw_text[:500]}"
                self._log(last_error)
                if attempt >= self.max_parse_retries:
                    raise ValueError(
                        f"JSON 解析失败，已重试 {self.max_parse_retries} 次。"
                        f"最后错误: {e}"
                    ) from e

        raise ValueError("无法解析 LLM 输出的任务计划")

    @staticmethod
    def _extract_text(response: ChatResponse) -> str:
        """从 ChatResponse 中提取纯文本"""
        texts = []
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)

    async def _call_llm_for_plan(
        self,
        user_msg: Msg,
        error_context: str | None = None,
    ) -> str:
        worker_caps = self.workers.get_capability_text()
        system_content = self.plan_prompt.format(
            worker_capabilities=worker_caps,
            min_subtasks=self.min_subtasks,
            max_subtasks=self.max_subtasks,
        )

        msgs = [Msg(name="system", content=system_content, role="system")]

        if error_context:
            msgs.append(Msg(
                name="system",
                content=f"上次输出格式错误，请严格按 JSON 格式重新输出。\n"
                        f"错误信息: {error_context}",
                role="user",
            ))

        msgs.append(user_msg)
        formatted = await self.formatter.format(msgs)

        response = await self.model(messages=formatted)
        if isinstance(response, AsyncGenerator):
            response = await self._collect_stream(response)

        if not response or not response.content:
            raise ValueError("LLM 返回空响应")

        return self._extract_text(response)

    async def _parse_plan(self, raw_text: str) -> list[SubTask]:
        text = raw_text.strip()

        # 去除 markdown 代码块包裹
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # 找到第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise json.JSONDecodeError("未找到 JSON 对象", text, 0)

        json_str = text[start:end + 1]
        data = json.loads(json_str)

        if "subtasks" not in data:
            raise KeyError("JSON 中缺少 'subtasks' 字段")

        subtasks = []
        for item in data["subtasks"]:
            subtasks.append(SubTask(
                id=item["id"],
                desc=item["desc"],
                worker=item["worker"],
                depends_on=item.get("depends_on", []),
            ))
        return subtasks

    async def _validate_subtasks(self, subtasks: list[SubTask]) -> None:
        if not subtasks:
            raise ValueError("子任务列表为空")

        count = len(subtasks)
        if count < self.min_subtasks:
            raise ValueError(
                f"子任务数量 {count} 少于最小值 {self.min_subtasks}"
            )
        if count > self.max_subtasks:
            raise ValueError(
                f"子任务数量 {count} 超过最大值 {self.max_subtasks}"
            )

        seen_ids: set[int] = set()
        for st in subtasks:
            if st.id in seen_ids:
                raise ValueError(f"子任务 ID 重复: {st.id}")
            seen_ids.add(st.id)

            if not self.workers.has_worker(st.worker):
                available = self.workers.list_workers()
                raise ValueError(
                    f"Worker '{st.worker}' 未注册。"
                    f"可用: {available}"
                )

    # ── 阶段 2: 执行 ───────────────────────────────────────

    async def _execute_plan(self, subtasks: list[SubTask]) -> dict[int, Msg]:
        return await self.scheduler.run(subtasks, self.workers)

    # ── 阶段 3: 汇总 ───────────────────────────────────────

    async def _summarize(
        self,
        original_question: Msg,
        results: dict[int, Msg],
        subtasks: list[SubTask],
    ) -> Msg:
        worker_results_text = await self._format_worker_results_for_prompt(
            results, subtasks
        )

        question_text = original_question.get_text_content() or str(
            original_question.content
        )

        system_content = self.summarize_prompt.format(
            original_question=question_text,
            worker_results=worker_results_text,
        )

        msgs = [Msg(name="system", content=system_content, role="system")]
        formatted = await self.formatter.format(msgs)

        response = await self.model(messages=formatted)
        if isinstance(response, AsyncGenerator):
            response = await self._collect_stream(response)

        if response and response.content:
            text = self._extract_text(response)
            print(f"\n{self.name}: {text}")
            return Msg(
                name=self.name,
                content=text,
                role="assistant",
            )

        return Msg(
            name=self.name,
            content="无法生成汇总结果",
            role="assistant",
        )

    async def _format_worker_results_for_prompt(
        self,
        results: dict[int, Msg],
        subtasks: list[SubTask],
    ) -> str:
        task_map = {t.id: t for t in subtasks}
        lines: list[str] = []

        for task_id in sorted(results.keys()):
            task = task_map.get(task_id)
            if task is None:
                lines.append(f"### 子任务 {task_id}")
                result_text = results[task_id].get_text_content() or ""
                lines.append(f"状态: 未知任务\n结果: {result_text}\n")
                continue

            status_label = {
                "pending": "未执行",
                "running": "执行中",
                "completed": "已完成",
                "failed": "失败",
            }.get(task.status, task.status)

            result_text = results[task_id].get_text_content() or ""
            if len(result_text) > self.result_max_length:
                result_text = (
                    result_text[:self.result_max_length]
                    + f"\n... (已截断，原长度 {len(result_text)} 字符)"
                )

            lines.append(
                f"### 子任务 {task_id}: {task.desc}\n"
                f"- Worker: {task.worker}\n"
                f"- 状态: {status_label}\n"
                f"- 结果:\n{result_text}\n"
            )

        # 补充失败/未完成的任务
        for task in subtasks:
            if task.id not in results:
                lines.append(
                    f"### 子任务 {task.id}: {task.desc}\n"
                    f"- Worker: {task.worker}\n"
                    f"- 状态: 失败\n"
                    f"- 错误: {task.error or '未知错误'}\n"
                )

        return "\n".join(lines) if lines else "（无子任务结果）"

    # ── 工具方法 ───────────────────────────────────────────

    async def _collect_stream(
        self,
        stream: AsyncGenerator,
    ) -> ChatResponse | None:
        final_text = ""
        reasoning_content = ""
        tool_calls: dict[str, dict] = {}
        usage = None

        try:
            async for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                for block in chunk.content:
                    if block.get("type") == "text":
                        final_text = block.get("text", "")
                    elif block.get("type") == "tool_use":
                        tool_calls[block["id"]] = block
                if chunk.reasoning_content:
                    reasoning_content = chunk.reasoning_content
        except Exception:
            pass

        content_blocks = []
        if final_text:
            content_blocks.append({"type": "text", "text": final_text})
        for tc in tool_calls.values():
            content_blocks.append(tc)

        return ChatResponse(
            content=content_blocks,
            usage=usage,
            reasoning_content=reasoning_content or None,
        )

    async def observe(self, msg: Msg | list[Msg] | None = None) -> None:
        await self.memory.add(msg)

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg)
