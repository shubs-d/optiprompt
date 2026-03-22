import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

class JSONLogger:
    def __init__(self, log_dir: str = "/data", filename: str = "optimization_logs.json"):
        self.log_file = Path(log_dir) / filename
        self._lock = threading.Lock()
        self._initialize_file()

    def _initialize_file(self):
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_file.exists() or self.log_file.stat().st_size == 0:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write("[\n]")
        except PermissionError:
            # Fallback for local testing without root /data
            self.log_file = Path("data") / self.log_file.name
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_file.exists() or self.log_file.stat().st_size == 0:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write("[\n]")

    def log(self, record: dict):
        with self._lock:
            try:
                record_json = json.dumps(record, indent=2)
                with open(self.log_file, 'rb+') as f:
                    f.seek(0, 2)
                    size = f.tell()
                    pos = size - 1
                    found = False
                    while pos >= 0:
                        f.seek(pos)
                        if f.read(1) == b']':
                            found = True
                            break
                        pos -= 1
                    
                    if found:
                        pos_before = pos - 1
                        needs_comma = False
                        while pos_before >= 0:
                            f.seek(pos_before)
                            c = f.read(1)
                            if c in b' \n\r\t':
                                pos_before -= 1
                                continue
                            if c != b'[':
                                needs_comma = True
                            break
                        
                        f.seek(pos)
                        if needs_comma:
                            f.write(b",\n")
                        else:
                            f.write(b"\n")
                        f.write(record_json.encode('utf-8'))
                        f.write(b"\n]")
            except Exception as e:
                print(f"Logging failed: {e}")

# Global singleton
logger = JSONLogger()
