import unittest
from unittest.mock import patch

from scripts.profile_memory import process_role, summarize


class MemoryProfileTests(unittest.TestCase):
    def test_workers_in_an_owned_process_group_are_attributed(self):
        with patch("scripts.profile_memory.os.getpgid", return_value=123):
            self.assertEqual(process_role(456, {123: "brain"}), "brain")
            self.assertIsNone(process_role(456, {789: "avatar"}))

    def test_simultaneous_peak_is_not_sum_of_individual_peaks(self):
        rows = []
        for index, used in enumerate(([80, 20], [30, 90])):
            rows.append({"monotonic": float(index), "processes": [], "devices": [
                {"index": gpu, "name": "test", "total_mib": 200, "used_mib": value}
                for gpu, value in enumerate(used)]})
        report = summarize(rows)
        self.assertEqual(report["max_simultaneous_device_mib"], 120)
        self.assertEqual(report["sum_individual_device_peaks_mib"], 170)
        self.assertEqual(report["actual_interval_seconds"]["median"], 1)
