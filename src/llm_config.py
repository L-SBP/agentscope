import os
import yaml
from pydantic.v1 import BaseModel, BaseSettings


class LLMModel(BaseModel):
    model: str
    base_url: str


class ThinkingConfig(BaseModel):
    type: str


class ModelConfig(BaseSettings):
    base_config: LLMModel
    stream: bool = True
    thinking: ThinkingConfig | None = None
    reasoning_effort: str | None = None
    api_key: str = ""


# ── 多模型配置加载器 ───────────────────────────────────────

_ENV_KEY_MAP = {
    "Deepseek": "DEEPSEEK_API",
    "Kimi": "KIMI_API",
    "Doubao": "DOUBAO_API",
}


class MultiModelConfig:
    def __init__(self, yaml_path: str | None = None) -> None:
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config.yaml"
            )
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)

        configs = yaml_data.get("LLM_Config", {})
        self._models: dict[str, ModelConfig] = {}

        for name, cfg in configs.items():
            env_key = _ENV_KEY_MAP.get(name, "")
            api_key = os.getenv(env_key, "")

            # thinking 字段在 yaml 中是 dict，需要转换
            if cfg.get("thinking"):
                cfg["thinking"] = ThinkingConfig(**cfg["thinking"])

            self._models[name] = ModelConfig(**cfg, api_key=api_key)

    def get(self, name: str) -> ModelConfig:
        if name not in self._models:
            raise KeyError(
                f"模型 '{name}' 未在 config.yaml 中配置。"
                f"可用: {list(self._models.keys())}"
            )
        return self._models[name]

    def get_extra_body(self, name: str) -> dict | None:
        """构建传给 OpenAIChatModel 的 extra_body"""
        cfg = self.get(name)
        extra: dict = {}
        if cfg.thinking:
            extra["thinking"] = {"type": cfg.thinking.type}
        if cfg.reasoning_effort:
            extra["reasoning_effort"] = cfg.reasoning_effort
        return extra if extra else None

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def __getitem__(self, name: str) -> ModelConfig:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._models


# ── 全局实例 ──────────────────────────────────────────────

_model_configs = MultiModelConfig()
"""多模型配置单例"""


# ── 兼容旧代码（run.py 仍可用） ─────────────────────────────

class DeepseekModel(BaseSettings):
    base_config: LLMModel
    stream: bool
    thinking: ThinkingConfig
    reasoning_effort: str
    api_key: str


ds_cfg = _model_configs.get("Deepseek")
llm_config = DeepseekModel(
    base_config=ds_cfg.base_config,
    stream=ds_cfg.stream,
    thinking=ds_cfg.thinking or ThinkingConfig(type="enabled"),
    reasoning_effort=ds_cfg.reasoning_effort or "max",
    api_key=ds_cfg.api_key,
)
