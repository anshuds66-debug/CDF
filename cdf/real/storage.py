import json
import os
import asyncio
from pathlib import Path
from cdf.real.events import BaseEvent
from cdf.utils.serialization import NumpyEncoder

class JSONLStorage:
    def __init__(self, directory: str, experiment_id: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.filepath = self.directory / f"{experiment_id}.jsonl"
        
    def _write_sync(self, data: str):
        with open(self.filepath, mode='a', encoding='utf-8') as f:
            f.write(data + "\n")
            
    async def append_event(self, event: BaseEvent):
        """
        Appends a single event to the JSONL file immediately.
        Uses asyncio.to_thread to prevent blocking the event loop.
        """
        data = json.dumps(event.model_dump(), cls=NumpyEncoder)
        await asyncio.to_thread(self._write_sync, data)

    def get_filepath(self) -> str:
        return str(self.filepath)
