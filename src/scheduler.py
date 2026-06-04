import asyncio
import os

from src.message import Msg
from src.subtask import SubTask
from src.worker_registry import WorkerRegistry


class TaskScheduler:
    """
    任务调度器

    负责：
    1. 解析子任务之间的依赖关系（DAG 拓扑排序）
    2. 按批次并行执行无依赖的子任务
    3. 处理 Worker 超时、失败重试
    4. 收集所有结果返回
    """

    def __init__(
        self,
        max_parallel: int = 3,      # 最大并发任务数
        timeout: float = 120.0,
    ) -> None:
        self.max_parallel = max_parallel
        self.timeout = timeout
        self._verbose = os.environ.get("NANO_AGENTSCOPE_VERBOSE", "0") == "1"

    async def run(
        self,
        subtasks: list[SubTask],
        registry: WorkerRegistry,
    ) -> dict[int, Msg]:
        if not subtasks:
            return {}

        await self._verify_no_cycles(subtasks)

        results: dict[int, Msg] = {}
        completed_ids: set[int] = set()
        remaining = list(subtasks)
        # 使用信号量限制并发执行的子任务数量，防止资源耗尽
        semaphore = asyncio.Semaphore(self.max_parallel)

        while remaining:
            ready = self._get_ready_tasks(remaining, completed_ids)
            if not ready:
                pending_ids = {t.id for t in remaining}
                blocked = pending_ids - {t.id for t in ready}
                self._log(
                    f"死锁：无法找到可执行任务。"
                    f"已完成: {completed_ids}，阻塞中: {blocked}"
                )
                break

            self._log(
                f"批次执行 {len(ready)} 个任务: "
                f"{[(t.id, t.desc[:40]) for t in ready]}"
            )

            batch_results = await self._execute_with_semaphore(
                ready, registry, semaphore, results
            )

            for task in ready:
                task_id = task.id
                if task_id in batch_results:
                    results[task_id] = batch_results[task_id]
                    task.status = "completed"
                    task.result = batch_results[task_id]
                else:
                    task.status = "failed"
                completed_ids.add(task_id)
                remaining = [t for t in remaining if t.id != task_id]

        return results

    async def _verify_no_cycles(self, subtasks: list[SubTask]) -> None:
        """
        使用深度优先搜索 (DFS) 检测子任务依赖图中是否存在环。

        逻辑说明：
        1. 维护两个集合：
           - visited: 记录所有已经完整遍历过的节点（及其子树），避免重复计算。
           - rec_stack: 记录当前 DFS 递归路径上的节点。如果在该路径中再次遇到某个节点，说明存在环。
        2. 对每个未访问的任务启动 DFS。
        3. 在 DFS 中，遍历当前任务的所有依赖项：
           - 如果依赖项未被访问，递归检查。
           - 如果依赖项已在当前递归栈 (rec_stack) 中，说明找到了环，返回 True。
        4. 回溯时从 rec_stack 移除当前节点。
        """
        # 构建任务ID到任务对象的映射，用于在DFS中快速查找依赖任务
        task_map = {t.id: t for t in subtasks}
        visited: set[int] = set()
        # 递归调用栈，用于检测当前DFS路径中是否存在环
        rec_stack: set[int] = set()

        def dfs(node_id: int) -> bool:
            """
            深度优先搜索检测环。
            :param node_id: 当前检查的任务ID
            :return: 如果发现环返回 True，否则返回 False
            """
            visited.add(node_id)
            rec_stack.add(node_id)
            
            task = task_map.get(node_id)
            if task:
                for dep_id in task.depends_on:
                    # 情况1: 依赖节点未被访问过，继续深入搜索
                    if dep_id not in visited:
                        if dfs(dep_id):
                            return True
                    # 情况2: 依赖节点已被访问，且仍在当前的递归栈中
                    # 这意味着从 dep_id 出发又回到了 dep_id (或者路径上的某个祖先)，构成环
                    elif dep_id in rec_stack:
                        return True
            
            # 当前节点的所有依赖都检查完毕，没有发现环，将其从递归栈中移除（回溯）
            rec_stack.discard(node_id)
            return False

        # 遍历所有任务，确保处理非连通图的情况
        for task in subtasks:
            if task.id not in visited:
                if dfs(task.id):
                    raise ValueError(f"检测到循环依赖，涉及任务 ID: {task.id}")

    def _get_ready_tasks(
        self,
        subtasks: list[SubTask],
        completed_ids: set[int],
    ) -> list[SubTask]:
        ready = []
        for task in subtasks:
            if task.id in completed_ids:
                continue
            # 只有当任务的所有依赖项 ID 都在已完成集合中时，该任务才就绪
            if all(dep_id in completed_ids for dep_id in task.depends_on):
                ready.append(task)
        return ready

    async def _execute_one_task(
        self,
        task: SubTask,
        registry: WorkerRegistry,
        dep_results: dict[int, Msg] | None = None,
    ) -> Msg:
        worker = registry.get_worker(task.worker)

        # 构造任务内容：原始描述 + 前置任务的结果
        task_content = task.desc
        if dep_results and task.depends_on:
            dep_lines = []
            for dep_id in task.depends_on:
                dep_msg = dep_results.get(dep_id)
                if dep_msg:
                    dep_text = dep_msg.get_text_content() or ""
                    header = f"[前置任务 {dep_id} 结果]"
                    if len(dep_text) > 1500:
                        dep_text = dep_text[:1500] + "...(已截断)"
                    dep_lines.append(f"{header}:\n{dep_text}")
            if dep_lines:
                task_content = (
                    f"{task.desc}\n\n---\n以下是你所依赖的上游任务执行结果，"
                    f"请基于这些信息完成你的任务：\n\n"
                    + "\n\n".join(dep_lines)
                )

        for attempt in range(task.max_retries + 1):
            task.retries = attempt
            if attempt > 0:
                self._log(f"任务 {task.id} 第 {attempt} 次重试")

            task.status = "running"
            try:
                task_msg = Msg(
                    name="orchestrator",
                    content=task_content,
                    role="user",
                )

                result = await asyncio.wait_for(
                    worker.reply(task_msg),
                    timeout=self.timeout,
                )
                return result

            except asyncio.TimeoutError:
                task.error = f"超时 ({self.timeout}s)"
                self._log(f"任务 {task.id} [{task.worker}] 超时")
            except Exception as e:
                task.error = str(e)
                self._log(f"任务 {task.id} [{task.worker}] 异常: {e}")

        task.status = "failed"
        return Msg(
            name=task.worker,
            content=f"[执行失败] 任务 '{task.desc[:100]}' 经过 "
                    f"{task.max_retries + 1} 次尝试后仍失败。"
                    f"错误: {task.error}",
            role="user",
        )

    async def _execute_with_semaphore(
        self,
        tasks: list[SubTask],
        registry: WorkerRegistry,
        semaphore: asyncio.Semaphore,
        dep_results: dict[int, Msg],
    ) -> dict[int, Msg]:
        """
        使用信号量限制并发，并行执行一批子任务。

        逻辑说明：
        1. 为每个任务创建一个协程，该协程在获取信号量后执行具体任务。
        2. 使用 asyncio.gather 并行等待所有协程完成。
        3. 收集结果并映射回任务 ID。
        """
        async def _bounded(task: SubTask) -> tuple[int, Msg]:
            # 获取信号量，确保同一时间只有 max_parallel 个任务在运行
            async with semaphore:
                result = await self._execute_one_task(task, registry, dep_results)
                return task.id, result

        # 创建所有任务的协程对象
        futures = [asyncio.create_task(_bounded(t)) for t in tasks]
        
        # 并行执行所有任务，return_exceptions=True 防止单个任务异常导致整体崩溃
        done_results = await asyncio.gather(*futures, return_exceptions=True)

        results: dict[int, Msg] = {}
        for item in done_results:
            if isinstance(item, Exception):
                self._log(f"批次执行异常: {item}")
                # 注意：此处无法直接关联到具体 task_id，因为异常发生时可能未返回 tuple
                # 如果需要更精细的错误处理，建议在 _bounded 内部捕获并返回特定错误 Msg
                continue
            task_id, msg = item
            results[task_id] = msg
        return results

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[Scheduler] {msg}")
