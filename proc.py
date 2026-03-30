from typing import Dict, Protocol, Any


class Logger(Protocol):
    def log(self, data: Dict[str, Any]) -> None: ...

    pass
