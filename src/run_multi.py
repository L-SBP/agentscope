import asyncio
import os
import sys

from dotenv import load_dotenv

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_SRC_DIR)

load_dotenv(os.path.join(_SRC_DIR, ".env"))

from src.memory import InMemoryMemory
from src.formatter import OpenAIFormatter
from src.llm import OpenAIChatModel
from src.llm_config import _model_configs
from src.loop_agent import ReActAgent, UserAgent
from src.tools import default_tools, Toolkit
from src.worker_registry import WorkerRegistry
from src.orchestrator import OrchestratorAgent
from src.scheduler import TaskScheduler


def _make_model(model_name: str) -> OpenAIChatModel:
    """用 MultiModelConfig 创建一个模型实例"""
    cfg = _model_configs.get(model_name)
    return OpenAIChatModel(
        model_name=cfg.base_config.model,
        api_key=cfg.api_key,
        base_url=cfg.base_config.base_url,
        stream=cfg.stream,
        extra_body=_model_configs.get_extra_body(model_name),
    )


async def main() -> None:
    formatter = OpenAIFormatter()

    # ── 模型分配 ────────────────────────────────────────────
    # 编排者：DeepSeek V4 Pro — 最强推理，做规划和汇总
    # Worker1：Kimi K2.6       — 综合能力强，做搜索和分析
    # Worker2：Doubao Lite     — 轻量快速，做计算和文件操作

    orchestrator_model = "Deepseek"
    search_worker_model = "Kimi"
    code_worker_model = "Doubao"

    # ── Worker 1: 搜索分析师（Kimi） ────────────────────────
    search_toolkit = Toolkit()
    search_toolkit.register_tool_function(
        default_tools().tools["search_web"][0]
    )
    search_toolkit.register_tool_function(
        default_tools().tools["read_file"][0]
    )

    search_worker = ReActAgent(
        name="搜索分析师",
        sys_prompt=(...),
        model=_make_model(search_worker_model),
        formatter=formatter,
        toolkit=search_toolkit,
        memory=InMemoryMemory(),
        max_iters=5,
        verbose=False,
    )

    # ── Worker 2: 代码执行者（Doubao） ──────────────────────
    code_toolkit = Toolkit()
    code_toolkit.register_tool_function(
        default_tools().tools["calculator"][0]
    )
    code_toolkit.register_tool_function(
        default_tools().tools["read_file"][0]
    )
    code_toolkit.register_tool_function(
        default_tools().tools["write_file"][0]
    )

    code_worker = ReActAgent(
        name="代码执行者",
        sys_prompt=(...),
        model=_make_model(code_worker_model),
        formatter=formatter,
        toolkit=code_toolkit,
        memory=InMemoryMemory(),
        max_iters=5,
        verbose=False,
    )

    # ── 注册 Worker ─────────────────────────────────────────

    registry = WorkerRegistry()
    registry.register_worker(
        "搜索分析师", search_worker,
        "擅长网络搜索、信息检索、资料收集和整理"
    )
    registry.register_worker(
        "代码执行者", code_worker,
        "擅长数学计算、文件读写和数据处理"
    )

    # ── 编排者（DeepSeek） ──────────────────────────────────

    orchestrator = OrchestratorAgent(
        name="编排者",
        model=_make_model(orchestrator_model),
        formatter=formatter,
        workers=registry,
        memory=InMemoryMemory(),
        scheduler=TaskScheduler(max_parallel=2, timeout=120),
        max_subtasks=6,
        min_subtasks=1,
        max_parse_retries=3,
    )

    user = UserAgent("User")

    # ── CLI ─────────────────────────────────────────────────

    print("=" * 60)
    print("  AgentScope Multi-Agent (Orchestrator-Worker)")
    print(f"  编排者: {orchestrator_model}")
    print(f"  Worker1: {search_worker_model} (搜索分析师)")
    print(f"  Worker2: {code_worker_model} (代码执行者)")
    print("  Commands: /exit 退出  /clear 清空记忆  /workers 查看Worker")
    print("=" * 60)

    while True:
        try:
            user_msg = await user.reply()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        user_text = (user_msg.get_text_content() or "").strip()

        if user_text.startswith("/"):
            cmd = user_text[1:].lower()
            if cmd == "exit":
                print("再见！")
                break
            elif cmd == "clear":
                await orchestrator.memory.clear()
                for w_name in registry.list_workers():
                    await registry[w_name].memory.clear()
                print("[所有记忆已清空]")
                continue
            elif cmd == "workers":
                print("\n已注册 Worker:")
                print(registry.get_capability_text())
                continue
            else:
                print(f"未知命令: /{cmd}")
                continue

        if not user_text:
            continue

        await orchestrator.reply(user_msg)
        print()


if __name__ == "__main__":
    asyncio.run(main())
