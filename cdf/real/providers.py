from typing import List, Dict, Optional
from pydantic import BaseModel
import os
import json
from pathlib import Path

class ProviderConfig(BaseModel):
    id: str
    name: str
    models: List[str]
    base_url: Optional[str] = None
    is_openai_compatible: bool = True
    default_max_retries: int = 5
    default_timeout: int = 60

class ProviderRegistry:
    """
    Manages available providers and their configurations.
    """
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.path = Path(config_path)
        else:
            self.path = Path.home() / ".cdf_providers.json"
            
        self._default_providers = {
            "openai": ProviderConfig(
                id="openai", 
                name="OpenAI", 
                models=["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"],
                is_openai_compatible=True
            ),
            "openrouter": ProviderConfig(
                id="openrouter",
                name="OpenRouter",
                models=["meta-llama/llama-3-70b-instruct", "anthropic/claude-3.5-sonnet"],
                base_url="https://openrouter.ai/api/v1",
                is_openai_compatible=True
            ),
            "gemini": ProviderConfig(
                id="gemini",
                name="Google Gemini",
                models=["gemini-1.5-flash", "gemini-1.5-pro"],
                is_openai_compatible=False
            )
        }
        
    def _read_custom(self) -> Dict[str, ProviderConfig]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {k: ProviderConfig(**v) for k, v in data.items()}
        except (json.JSONDecodeError, IOError):
            return {}
            
    def _write_custom(self, custom: Dict[str, ProviderConfig]) -> None:
        with open(self.path, 'w', encoding='utf-8') as f:
            data = {k: v.model_dump() for k, v in custom.items()}
            json.dump(data, f, indent=2)

    def get_all_providers(self) -> Dict[str, ProviderConfig]:
        providers = dict(self._default_providers)
        custom = self._read_custom()
        providers.update(custom)
        return providers

    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        return self.get_all_providers().get(provider_id)
        
    def add_or_update_provider(self, config: ProviderConfig) -> None:
        custom = self._read_custom()
        custom[config.id] = config
        self._write_custom(custom)

    def delete_provider(self, provider_id: str) -> bool:
        custom = self._read_custom()
        if provider_id in custom:
            del custom[provider_id]
            self._write_custom(custom)
            return True
        return False
