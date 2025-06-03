import sys
import time
from typing import Callable, Dict

ERROR_DEBOUNCE_INTERVAL = 3.0  # seconds

class ConsoleInterceptor:

    """
    Intercepts stdout/stderr to capture and filter robot messages for the GUI console.
    """
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._last_logged: Dict[str, float] = {}
        self._debounce_interval = ERROR_DEBOUNCE_INTERVAL

    def write(self, msg: str) -> None:
        msg = msg.strip()
        if not msg:
            return

        now = time.time()
        key = None

        # Filter and categorize messages
        if "MX_ST_JOINT_OVER_LIMIT" in msg:
            key = "joint_limit"
        elif "MX_ST_ALREADY_ERR" in msg:
            key = "already_error"

        # Apply debouncing to avoid message spam
        if key:
            last_time = self._last_logged.get(key, 0)
            if now - last_time > self._debounce_interval:
                clean_msg = msg.split("Command:")[0].strip()
                self.callback(f"⚠️ {clean_msg}")
                self._last_logged[key] = now

        # Always write to original stdout
        self._stdout.write(msg + "\n")

    def flush(self) -> None:
        self._stdout.flush()
        self._stderr.flush()
