import os
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Dict, Optional

from src.core import LLM, LLMGeneration


def deep_getattr(obj: Any, attr_chain: str, default: Any = None) -> Any:
    """Safely get nested attributes from objects or dicts."""
    def _get(obj: Any, attr: str) -> Any:
        if obj is None:
            raise AttributeError
        
        # Handle dict access
        if isinstance(obj, dict):
            if attr in obj:
                return obj[attr]
        
        # Handle array indexing
        if '[' in attr and attr.endswith(']'):
            name, idx_part = attr[:-1].split('[', 1)
            target = obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name) if name else obj
            if target is None:
                raise AttributeError
            idx = int(idx_part)
            if isinstance(target, (list, tuple)):
                return target[idx]
            if isinstance(target, dict):
                return target.get(idx, target.get(str(idx), default))
            raise AttributeError
        
        # Regular attribute/key access
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr)
    
    try:
        return reduce(_get, attr_chain.split('.'), obj)
    except (AttributeError, IndexError, KeyError, ValueError, TypeError):
        return default


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI-compatible endpoints."""
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[int] = None
    max_retries: Optional[int] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    extra_create_args: Dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleLLM(LLM):
    """OpenAI-compatible chat completion wrapper."""
    
    def __init__(self, config: OpenAIConfig) -> None:
        self.config = config
        from openai import OpenAI
        
        client_kwargs = {
            "api_key": config.api_key,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        
        self._client = OpenAI(**client_kwargs)

    def generate(self, prompt: str) -> LLMGeneration:
        params = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        response = self._client.chat.completions.create(**params)
        
        return LLMGeneration(
            raw_out=response,
            content=deep_getattr(response, "choices[0].message.content", default=""),
            prompt_tokens=deep_getattr(response, "usage.prompt_tokens", default=0),
            completion_tokens=deep_getattr(response, "usage.completion_tokens", default=0),
            finish_reason=deep_getattr(response, "choices[0].finish_reason", default=None),
        )


def build_llm(llm_cfg: Dict[str, Any]) -> LLM:
    """Build LLM instance from configuration."""
    if not llm_cfg:
        raise ValueError("LLM configuration is required.")
    
    llm_type = str(llm_cfg.get("type"))
    if llm_type != "openai":
        raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    config = OpenAIConfig(
        model=llm_cfg.get("model"),
        api_key=os.getenv(llm_cfg.get("api_key_env")),
        base_url=llm_cfg.get("base_url"),
        timeout=llm_cfg.get("timeout", 120),
        max_retries=llm_cfg.get("max_retries", 1),
        temperature=float(llm_cfg.get("temperature", 0.7)),
        max_tokens=llm_cfg.get("max_tokens"),
        extra_create_args=llm_cfg.get("extra_args", {}),
    )
    
    return OpenAICompatibleLLM(config)