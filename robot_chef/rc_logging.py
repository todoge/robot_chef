from collections import OrderedDict
from pathlib import Path

import structlog


def reorder_keys(_, __, event_dict: dict) -> OrderedDict:
    ordered = OrderedDict()
    for key in ("timestamp", "event", "src"):
        if key in event_dict:
            ordered[key] = event_dict.pop(key)
    for key, value in event_dict.items():
        ordered[key] = value
    return ordered

def get_logger(script: str) -> structlog.BoundLogger:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            reorder_keys,
            structlog.processors.JSONRenderer(indent=4),
        ]
    )
    return structlog.get_logger(src=Path(script).stem)
