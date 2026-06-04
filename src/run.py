import asyncio
import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_SRC_DIR)

from src.formatter import OpenAIFormatter
from src.llm import OpenAIChatModel
from src.llm_config import llm_config
from src.loop_agent import ReActAgent, UserAgent
from src.tools import default_tools


async def main() -> None:
    model = OpenAIChatModel(
        model_name=llm_config.base_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_config.base_url,
        stream=llm_config.stream,
        extra_body={"thinking": {"type": "enable"}},
    )

    toolkit = default_tools()
    user = UserAgent("User")

    agent = ReActAgent(
        name="Agent",
        sys_prompt=(
            "你是一个有用的AI助手，名叫 AgentScope。"
            "你可以使用工具来完成用户的任务。"
            "每次只能调用一个工具，收到工具结果后再决定下一步。"
            "如果不需要工具就直接回答用户。"
        ),
        model=model,
        formatter=OpenAIFormatter(),
        toolkit=toolkit,
        max_iters=10,
    )

    print("=" * 60)
    print("  AgentScope ReAct Agent")
    print("  Commands: /exit 退出  /clear 清空记忆  /tools 查看工具")
    print("=" * 60)

    while True:
        try:
            user_msg = await user.reply()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        user_text = user_msg.get_text_content() or ""

        if user_text.startswith("/"):
            cmd = user_text[1:].lower()
            if cmd == "exit":
                print("再见！")
                break
            elif cmd == "clear":
                await agent.memory.clear()
                print("[记忆已清空]")
                continue
            elif cmd == "tools":
                print("\n已注册工具:")
                for name, (_, schema) in toolkit.tools.items():
                    desc = schema.get("function", {}).get("description", "(无描述)")
                    print(f"  - {name}: {desc[:80]}")
                continue
            else:
                print(f"未知命令: /{cmd}")
                continue

        if not user_text:
            continue

        await agent.reply(user_msg)
        print()


if __name__ == "__main__":
    asyncio.run(main())
