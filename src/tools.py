import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any
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