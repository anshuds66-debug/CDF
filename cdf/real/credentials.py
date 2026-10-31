import os
import json
import stat
from pathlib import Path
from typing import Dict, Optional

class CredentialManager:
    """
    Manages API keys securely using a local restricted file.
    Does not use keyring to ensure compatibility across all environments.
    """
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path:
            self.path = Path(storage_path)
        else:
            self.path = Path.home() / ".cdf_credentials.json"
            
    def _read_credentials(self) -> Dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
            
    def _write_credentials(self, creds: Dict[str, str]) -> None:
        # Create or overwrite file
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(creds, f, indent=2)
            
        # Secure the file (Unix-like)
        try:
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            # On Windows this might not map perfectly, but best effort
            pass

    def set_api_key(self, provider_id: str, api_key: str) -> None:
        creds = self._read_credentials()
        creds[provider_id] = api_key
        self._write_credentials(creds)
        
    def get_api_key(self, provider_id: str) -> Optional[str]:
        creds = self._read_credentials()
        return creds.get(provider_id)
        
    def delete_api_key(self, provider_id: str) -> bool:
        creds = self._read_credentials()
        if provider_id in creds:
            del creds[provider_id]
            self._write_credentials(creds)
            return True
        return False
        
    def get_masked_api_key(self, provider_id: str) -> Optional[str]:
        key = self.get_api_key(provider_id)
        if not key:
            return None
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}...{key[-4:]}"
