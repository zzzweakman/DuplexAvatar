import unittest
from unittest.mock import patch

from scripts import manage


class DevicePlanTests(unittest.TestCase):
    def test_explicit_four_gpu_machine_configuration(self):
        with patch.dict(manage.CONFIG, allowed_gpus=[0, 1, 2, 3], brain_gpu=0, tts_gpu=2, avatar_gpu=1, asr_gpu=3):
            self.assertEqual(manage.role_devices("brain"), [0, 2])
            self.assertEqual(manage.role_devices("avatar"), 1)
            self.assertEqual(manage.role_devices("asr"), 3)
            env = manage.process_env("brain", manage.role_devices("brain"))
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,2")
            with patch.dict(manage.CONFIG, tts_gpu=4):
                with self.assertRaisesRegex(ValueError, "authorized"):
                    manage.role_devices("brain")

    def test_four_gpu_plan_and_single_gpu_voice_rollback(self):
        with patch.dict(manage.CONFIG, brain_gpu=4, tts_gpu=6, avatar_gpu=5, asr_gpu=7):
            self.assertEqual(manage.role_devices("brain"), [4, 6])
            self.assertEqual(manage.role_devices("avatar"), 5)
            self.assertEqual(manage.role_devices("asr"), 7)
            self.assertIsNone(manage.role_devices("gateway"))
            with patch.dict(manage.CONFIG, tts_gpu=4):
                self.assertEqual(manage.role_devices("brain"), [4])

    def test_rejects_unauthorized_devices_and_avatar_contention(self):
        for field, role in (("brain_gpu", "brain"), ("tts_gpu", "brain"), ("avatar_gpu", "avatar"), ("asr_gpu", "asr")):
            with self.subTest(field=field), patch.dict(manage.CONFIG, **{field: 0}):
                with self.assertRaisesRegex(ValueError, "authorized"):
                    manage.role_devices(role)
        with patch.dict(manage.CONFIG, brain_gpu=4, tts_gpu=5, avatar_gpu=5):
            with self.assertRaisesRegex(ValueError, "separate"):
                manage.role_devices("brain")
