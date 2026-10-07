#!/usr/bin/env python3
"""The GPU's temperature on Apple silicon, from the SMC, without root.

The System Management Controller publishes every sensor as a four-character
key, and any user may read them through the `AppleSMC` IOKit service: no
`sudo powermetrics`, no helper. The GPU's are the `Tg**` keys: 168 on the M5
Ultra, all within about 5 C of each other, 46 C idle and 59 C average, 64 C
hottest, after 18 s of Splash decoding (2026-10-07). The hottest is reported,
as the one the GPU throttles on.

The keys are found once, by walking the SMC's index (~3,000 keys, ~0.1 s);
after that a reading is one call per GPU key, ~20 ms for all of them, and is
kept for `_TTL_S` so concurrent pollers share it. Anything that fails --
another architecture, a sandbox, a key that reads as nonsense -- yields None,
which the UI renders as no reading rather than as a cold GPU.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct
import threading
import time

_KERNEL_INDEX_SMC = 2  # kSMCHandleYPCEvent: the user client's one struct method
_CMD_READ_KEY = 5
_CMD_KEY_FROM_INDEX = 8
_CMD_KEY_INFO = 9
_FLOAT = struct.unpack(">I", b"flt ")[0]
_GPU_PREFIX = "Tg"
# A die sensor below this reads unpowered or uncalibrated, above it nonsense.
_PLAUSIBLE_C = (5.0, 130.0)
_TTL_S = 2.0


class _Vers(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint8), ("minor", ctypes.c_uint8), ("build", ctypes.c_uint8),
                ("reserved", ctypes.c_uint8), ("release", ctypes.c_uint16)]


class _PLimit(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint16), ("length", ctypes.c_uint16),
                ("cpu", ctypes.c_uint32), ("gpu", ctypes.c_uint32), ("mem", ctypes.c_uint32)]


class _KeyInfo(ctypes.Structure):
    _fields_ = [("data_size", ctypes.c_uint32), ("data_type", ctypes.c_uint32),
                ("data_attributes", ctypes.c_uint8)]


class _KeyData(ctypes.Structure):
    """`SMCKeyData_t`, the 80-byte struct the SMC user client takes and returns."""
    _fields_ = [("key", ctypes.c_uint32), ("vers", _Vers), ("p_limit", _PLimit),
                ("key_info", _KeyInfo), ("result", ctypes.c_uint8), ("status", ctypes.c_uint8),
                ("data8", ctypes.c_uint8), ("data32", ctypes.c_uint32),
                ("bytes", ctypes.c_uint8 * 32)]


def _code(key: str) -> int:
    return struct.unpack(">I", key.encode("latin-1"))[0]


def _name(code: int) -> str:
    return struct.pack(">I", code).decode("latin-1")


class SMC:
    """One open connection to `AppleSMC`."""

    def __init__(self):
        iokit = ctypes.CDLL(ctypes.util.find_library("IOKit") or "IOKit")
        iokit.IOServiceMatching.restype = ctypes.c_void_p
        iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
        iokit.IOServiceGetMatchingService.restype = ctypes.c_uint32
        iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        iokit.IOServiceOpen.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                        ctypes.POINTER(ctypes.c_uint32)]
        iokit.IOConnectCallStructMethod.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]
        self._iokit = iokit
        service = iokit.IOServiceGetMatchingService(0, iokit.IOServiceMatching(b"AppleSMC"))
        if not service:
            raise OSError("no AppleSMC service")
        task = ctypes.c_uint32.in_dll(ctypes.CDLL(None), "mach_task_self_").value
        connection = ctypes.c_uint32()
        rc = iokit.IOServiceOpen(service, task, 0, ctypes.byref(connection))
        iokit.IOObjectRelease(service)
        if rc != 0:
            raise OSError(f"IOServiceOpen(AppleSMC) returned {rc:#x}")
        self._connection = connection.value

    def _call(self, request: _KeyData) -> _KeyData:
        reply = _KeyData()
        size = ctypes.c_size_t(ctypes.sizeof(_KeyData))
        rc = self._iokit.IOConnectCallStructMethod(
            self._connection, _KERNEL_INDEX_SMC, ctypes.byref(request), ctypes.sizeof(_KeyData),
            ctypes.byref(reply), ctypes.byref(size))
        if rc != 0 or reply.result != 0:
            raise OSError(f"SMC call {request.data8} returned {rc:#x}/{reply.result}")
        return reply

    def key_info(self, key: str) -> tuple[int, int]:
        reply = self._call(_KeyData(key=_code(key), data8=_CMD_KEY_INFO))
        return reply.key_info.data_size, reply.key_info.data_type

    def read(self, key: str, size: int) -> bytes:
        request = _KeyData(key=_code(key), data8=_CMD_READ_KEY)
        request.key_info.data_size = size
        return bytes(self._call(request).bytes[:size])

    def keys(self):
        size, _ = self.key_info("#KEY")
        count = int.from_bytes(self.read("#KEY", size), "big")
        for index in range(count):
            yield _name(self._call(_KeyData(data8=_CMD_KEY_FROM_INDEX, data32=index)).key)


def hottest(readings) -> float | None:
    """The hottest plausible reading, or None when there is none."""
    low, high = _PLAUSIBLE_C
    values = [value for value in readings if value is not None and low <= value <= high]
    return round(max(values), 1) if values else None


class GpuThermometer:
    def __init__(self, open_smc=SMC):
        self._open_smc = open_smc
        self._smc = None
        self._keys: list[str] | None = None
        self._failed = False
        self._lock = threading.Lock()
        self._cached: tuple[float, float | None] = (0.0, None)

    def _gpu_keys(self, smc) -> list[str]:
        keys = []
        for key in smc.keys():
            if key.startswith(_GPU_PREFIX):
                size, data_type = smc.key_info(key)
                if data_type == _FLOAT and size == 4:
                    keys.append(key)
        return keys

    def _read(self, smc, key: str) -> float | None:
        try:
            return struct.unpack("<f", smc.read(key, 4))[0]
        except OSError:
            return None

    def celsius(self) -> float | None:
        with self._lock:
            at, value = self._cached
            if time.monotonic() - at < _TTL_S or self._failed:
                return value
            try:
                if self._smc is None:
                    self._smc = self._open_smc()
                    self._keys = self._gpu_keys(self._smc)
                value = hottest(self._read(self._smc, key) for key in self._keys)
            except (OSError, AttributeError, ValueError):
                # No SMC here, or not one that answers like this: stop asking.
                self._failed = True
                value = None
            self._cached = (time.monotonic(), value)
            return value


_THERMOMETER = GpuThermometer()


def gpu_temperature_c() -> float | None:
    """The hottest GPU sensor in degrees C, or None when it cannot be read."""
    return _THERMOMETER.celsius()


__all__ = ["GpuThermometer", "SMC", "gpu_temperature_c", "hottest"]
