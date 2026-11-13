"""Read Windows power status without changing system or GPU settings."""

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json


class PowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", wintypes.BYTE),
        ("BatteryFlag", wintypes.BYTE),
        ("BatteryLifePercent", wintypes.BYTE),
        ("SystemStatusFlag", wintypes.BYTE),
        ("BatteryLifeTime", wintypes.DWORD),
        ("BatteryFullLifeTime", wintypes.DWORD),
    ]


def read_status():
    value = PowerStatus()
    okay = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(value))
    return dict(
        utc=datetime.now(timezone.utc).isoformat(),
        okay=bool(okay),
        **{name: getattr(value, name) for name, _ in value._fields_},
    )


if __name__ == "__main__":
    print(json.dumps(read_status(), indent=2))
