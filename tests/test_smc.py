"""The GPU's temperature on Apple silicon, read from the SMC."""

from __future__ import annotations

import pathlib
import struct
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
from platforms import smc  # noqa: E402

FLT = struct.unpack(">I", b"flt ")[0]
UI8 = struct.unpack(">I", b"ui8 ")[0]


class FakeSMC:
    """Keys as {name: (data_type, celsius or OSError)}."""

    def __init__(self, keys):
        self.table = keys
        self.reads = 0

    def keys(self):
        return iter(self.table)

    def key_info(self, key):
        return 4, self.table[key][0]

    def read(self, key, size):
        self.reads += 1
        value = self.table[key][1]
        if isinstance(value, Exception):
            raise value
        return struct.pack("<f", value)


class GpuThermometerTests(unittest.TestCase):
    def thermometer(self, keys):
        fake = FakeSMC(keys)
        return smc.GpuThermometer(open_smc=lambda: fake), fake

    def test_the_hottest_gpu_sensor_is_reported(self):
        thermometer, _ = self.thermometer({
            "Tg0A": (FLT, 47.1), "Tg1b": (FLT, 63.8), "Tg2c": (FLT, 52.0),
            "Tp01": (FLT, 90.0),  # a CPU core: not the GPU
            "TgXX": (UI8, 99.0),  # not a float sensor
        })
        self.assertEqual(thermometer.celsius(), 63.8)

    def test_implausible_and_failing_sensors_are_skipped(self):
        thermometer, _ = self.thermometer({
            "Tg00": (FLT, 0.0), "Tg01": (FLT, 1e9), "Tg02": (FLT, OSError("gone")),
            "Tg03": (FLT, 55.5),
        })
        self.assertEqual(thermometer.celsius(), 55.5)

    def test_no_gpu_sensors_is_no_reading(self):
        thermometer, _ = self.thermometer({"Tp01": (FLT, 50.0)})
        self.assertIsNone(thermometer.celsius())

    def test_concurrent_pollers_share_one_reading(self):
        thermometer, fake = self.thermometer({"Tg00": (FLT, 50.0)})
        thermometer.celsius()
        thermometer.celsius()
        self.assertEqual(fake.reads, 1)

    def test_no_smc_means_none_and_no_more_attempts(self):
        opened = mock.Mock(side_effect=OSError("no AppleSMC service"))
        thermometer = smc.GpuThermometer(open_smc=opened)
        self.assertIsNone(thermometer.celsius())
        with mock.patch.object(smc.time, "monotonic", return_value=1e12):
            self.assertIsNone(thermometer.celsius())
        opened.assert_called_once()

    def test_the_request_struct_is_the_80_bytes_the_kernel_expects(self):
        self.assertEqual(smc.ctypes.sizeof(smc._KeyData), 80)


if __name__ == "__main__":
    unittest.main()
