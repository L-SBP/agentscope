import os
import yaml
from pydantic.v1 import BaseModel, BaseSettings


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

llm_config = DeepseekModel(
    **app_config.LLM_Config["Deepseek"],
    api_key=os.getenv("DEEPSEEK_API", app_config.LLM_Config["Deepseek"].get("DEEPSEEK_API", "")),
)