import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any, Literal
from docstring_parser import parse
from pydantic import Field, create_model

from src.message import ToolUseBlock, TextBlock


@dataclass
class ToolResponse:
    """
    工具执行结果
    """
    content: list[TextBlock] = field(default_factory=list)
    metadata: dict | None = None
    is_last: bool = True
    is_interrupted: bool = False    # 用于实时中断标记


def _parse_function_to_schema(func: Callable) -> dict:
    """
    从函数签名和 docstring 解析 JSON Schema
    :param func:
    :return:
    """
    # 解析 docstring
    docstring = parse(func.__doc__ or "")
    params_doc = {p.arg_name: p.description for p in docstring.params}

    descriptions = []
    if docstring.short_description:
        descriptions.append(docstring.short_description)
    if docstring.long_description:
        descriptions.append(docstring.long_description)
    func_description = '\n'.join(descriptions)

    fields = {}
    for name, param in inspect.signature(func).parameters.items():
        if name in ["self", "cls"]:
            continue

        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            annotation = Any

        if param.default == inspect.Parameter.empty:
            default = ...
        else:
            default = param.default

        description = params_doc.get(name, None)

        fields[name] = (annotation, Field(default=default, description=description))

    if fields:
        DynamicModel = create_model("DynamicModel", **fields)
        params_schema = DynamicModel.model_json_schema()

        params_schema.pop("title", None)
        for prop in params_schema.get("properties", {}).values():
            prop.pop("title", None)
    else:
        params_schema = {"type": "object"}

    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "parameters": params_schema,
        }
    }

    if func_description:
        schema["function"]["description"] = func_description

    return schema

class Toolkit:
    """
    工具管理器，注册、管理和执行工具函数
    """

    def __init__(self) -> None:
        """
        初始化工具管理器
        """
        self._tools: dict[str, tuple[Callable, dict]] = {}

    def register_tool_function(
        self,
        func: Callable,
        description: str | None = None,
    ) -> Callable:
        """
        注册工具函数，可作为装饰器使用

        Usage:
            toolkit = Toolkit()

            @toolkit.register_tool_function
            def my_tool(x: int) -> str:
                '''示例工具'''
                return str(x)
        :param func:
        :param description:
        :return: 原函数，支持装饰器用法
        """
        schema = _parse_function_to_schema(func)

        if description:
            schema['function']['description'] = description

        self._tools[func.__name__] = (func, schema)
        return func

    def remove_tool_function(self, name: str) -> None:
        """
        移除工具函数
        :param name:
        :return:
        """
        self._tools.pop(name, None)

    def get_json_schemas(self) -> list[dict]:
        """
        获取所有工具的 JSON Schema列表
        :return:
        """
        return [schema for _, schema in self._tools.values()]

    @property
    def tools(self) -> dict[str, tuple[Callable, dict]]:
        """
        获取所有注册的工具
        :return:
        """
        return self._tools

    async def call_tool_function(
        self,
        tool_call: ToolUseBlock,
    ) -> ToolResponse:
        """
        执行工具函数
        :param tool_call:
        :return:
        """
        func_name = tool_call["name"]

        # 检查函数是否存在
        if func_name not in self._tools:
            return ToolResponse(
                content=[TextBlock(
                    type="text",
                    text=f"Error: 找不到工具函数 '{func_name}'"
                )],
            )

        func, _ = self._tools[func_name]
        kwargs = tool_call.get("args", {}) or {}

        try:
            # 执行函数
            if inspect.iscoroutinefunction(func):
                # 异步函数
                result = await func(**kwargs)
            else:
                # 同步函数
                result = func(**kwargs)

            # 确保返回 ToolResponse
            if isinstance(result, ToolResponse):
                return result
            else:
                return ToolResponse(
                    content=[TextBlock(type="text", text=result)],
                )
        except Exception as e:
            return ToolResponse(
                content=[TextBlock(type="text", text=str(e))],
            )
    def clear(self) -> None:
        """
        清空所有工具
        :return:
        """
        self._tools.clear()


_default_toolkit = Toolkit()


def tool(func: Callable) -> Callable:
    """
    装饰器：自动注册工具函数到默认工具集

    Usage:
        @tool
        def my_tool(x: int) -> str:
            '''示例工具'''
            return str(x)

    工具函数名、参数schema和描述会从其签名和docstring自动解析。
    """
    return _default_toolkit.register_tool_function(func)


def default_tools() -> Toolkit:
    """获取包含所有内置工具的默认工具集"""
    return _default_toolkit


# ---- 内置工具函数 ----

@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式并返回结果。仅支持加减乘除、幂运算和括号。"""
    allowed = set("0123456789+-*/().^% ")
    cleaned = expression.replace("**", "^").replace("math.", "")
    if not all(c in allowed for c in cleaned):
        return f"Error: 表达式包含不允许的字符，仅支持数字和 + - * / ( ) . ^ % 运算符"
    try:
        safe = expression.replace("^", "**")
        result = eval(
            safe,
            {"__builtins__": {}},
            {
                "abs": abs, "round": round, "min": min, "max": max,
                "pow": pow, "int": int, "float": float, "sqrt": lambda x: x ** 0.5,
            },
        )
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def search_web(
    query: str,
    search_type: Literal["auto", "instant", "fast", "deep"] = "auto",
    category: str = "",
    max_results: int = 5,
    include_full_text: bool = False,
) -> str:
    """
    使用 Exa AI 搜索引擎搜索网络，支持多种搜索模式和内容分类筛选
    :param query: 搜索关键词或问题
    :param search_type: 搜索类型，默认 auto。auto=自动选择最佳模式, instant=最快(180ms), fast=低延迟, deep=深度研究(适合复杂问题)
    :param category: 内容分类筛选。可选: research paper(论文), news(新闻), company(公司), personal site(个人网站), financial report(财报), people(人物)。留空则不筛选
    :param max_results: 返回结果数量，默认 5
    :param include_full_text: 是否包含网页全文。默认 False 仅返回 AI 提取的关键摘要，设 True 可获得更详细内容但消耗更多 token
    :return: 搜索结果摘要
    """
    try:
        from exa_py import Exa
        import os

        search_api = os.getenv("EXA_SEARCH_API")
        if not search_api:
            return "Error: 未设置 SEARCH_API_KEY 环境变量"

        exa = Exa(api_key=search_api)

        contents: dict = {"highlights": True}
        if include_full_text:
            contents["text"] = {"maxCharacters": 3000}

        kwargs: dict = {
            "query": query,
            "type": search_type,
            "num_results": max_results,
            "system_prompt": "Prefer official sources and recent information, avoid duplicate results",
            "contents": contents,
        }
        if category:
            kwargs["category"] = category

        results = exa.search(**kwargs)

        formatted_results = []
        for i, item in enumerate(results.results, 1):
            title = item.title or "无标题"
            url = item.url or "无链接"
            text = item.text or ""
            highlights = item.highlights or []

            if highlights:
                snippet = " | ".join(h[:300] for h in highlights[:3])
            elif text:
                snippet = text[:800] + "..." if len(text) > 800 else text
            else:
                snippet = "(无摘要)"

            formatted_results.append(
                f"{i}. {title}\n   URL: {url}\n   摘要: {snippet}\n"
            )

        return "\n".join(formatted_results) if formatted_results else "未找到相关结果"

    except ImportError:
        return "Error: 未安装 exa-py 库，请运行 pip install exa-py"
    except Exception as e:
        return f"Error: 搜索失败 - {str(e)}"

@tool
def read_file(
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """读取文件内容，支持指定行偏移和行数限制
    :param file_path: 要读取的文件路径（支持相对路径和绝对路径）
    :param offset: 起始行号（0表示从第一行开始）
    :param limit: 最大读取行数，默认2000行
    :return: 文件内容
    """
    import os

    full_path = os.path.abspath(file_path)

    if not os.path.exists(full_path):
        return f"Error: 文件不存在: {file_path}"

    if os.path.isdir(full_path):
        return f"Error: 路径为目录，非文件: {file_path}"

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(full_path, "r", encoding="gbk", errors="replace") as f:
            lines = f.readlines()
    except PermissionError:
        return f"Error: 没有权限读取文件: {file_path}"

    total = len(lines)
    if offset >= total:
        return f"Error: 偏移 {offset} 超出文件总行数 {total}"

    sliced = lines[offset:offset + limit]

    output = []
    for i, line in enumerate(sliced, start=offset + 1):
        output.append(f"{i}: {line.rstrip()}")

    result = "\n".join(output)
    if offset + limit < total:
        result += f"\n\n... (已截断，第 {offset + limit + 1} 行及之后未显示，共 {total} 行)"

    return result

@tool
def write_file(
    file_path: str,
    content: str,
    overwrite: bool = True,
) -> str:
    """将内容写入文件，默认覆盖已有文件
    :param file_path: 要写入的文件路径（支持相对路径和绝对路径）
    :param content: 要写入的文本内容
    :param overwrite: 是否覆盖已有文件，默认True。设为False时文件已存在则返回错误
    :return: 写入结果
    """
    import os

    full_path = os.path.abspath(file_path)

    if not overwrite and os.path.exists(full_path):
        return f"Error: 文件已存在，不允许覆盖: {file_path}"

    parent_dir = os.path.dirname(full_path)
    if parent_dir:
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except PermissionError:
            return f"Error: 没有权限创建目录: {parent_dir}"

    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
    except PermissionError:
        return f"Error: 没有权限写入文件: {file_path}"
    except Exception as e:
        return f"Error: 写入失败 - {e}"

    line_count = content.count("\n") + 1
    char_count = len(content)
    return f"成功写入文件: {full_path} ({line_count} 行, {char_count} 字符)"


