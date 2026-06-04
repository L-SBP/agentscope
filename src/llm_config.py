# llm模块，用来对接各种大模型
import os
import yaml
from pydantic.v1 import BaseModel, BaseSettings


def _read_env(env_path: str | None = None) -> dict:
    """手动读取 .env 文件，避免依赖 python-dotenv"""
    if env_path is None:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
    result = {}
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                sep = "=" if "=" in line else ":"
                key, _, value = line.partition(sep)
                result[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return result


class LLMModel(BaseModel):
    """
    大模型基础配置
    """
    model: str
    base_url: str


class ThinkingConfig(BaseModel):
    type: str


class DeepseekModel(BaseSettings):
    """
    deepseek大模型配置
    """
    base_config: LLMModel
    stream: bool
    thinking: ThinkingConfig
    reasoning_effort: str
    api_key: str


class AppConfig(BaseModel):
    LLM_Config: dict

    @classmethod
    def from_yaml(cls, yaml_path: str = "config.yaml"):
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**yaml_data)

app_config = AppConfig.from_yaml()
_env_vars = _read_env()

llm_config = DeepseekModel(
    **app_config.LLM_Config["Deepseek"],
    api_key=_env_vars.get("DEEPSEEK_API", app_config.LLM_Config["Deepseek"].get("DEEPSEEK_API", "")),
)