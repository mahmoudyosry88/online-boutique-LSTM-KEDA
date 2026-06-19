import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import unittest
import tempfile
from datetime import datetime, timezone

from scripts.pipeline.run_pipeline import get_users_for_segment


class TestPipeline(unittest.TestCase):

    def test_get_users_for_segment_returns_int(self):
        for i in range(72):
            users = get_users_for_segment(i)
            self.assertIsInstance(users, int)
            self.assertGreaterEqual(users, 40)
            self.assertLessEqual(users, 200)

    def test_get_users_for_segment_cycles(self):
        self.assertEqual(get_users_for_segment(0), get_users_for_segment(36))
        self.assertEqual(get_users_for_segment(1), get_users_for_segment(37))
        self.assertEqual(get_users_for_segment(35), get_users_for_segment(71))

    def test_get_users_for_segment_pattern(self):
        self.assertEqual(get_users_for_segment(0), 40)
        self.assertEqual(get_users_for_segment(7), 200)
        self.assertEqual(get_users_for_segment(8), 200)
        self.assertEqual(get_users_for_segment(11), 120)
        self.assertEqual(get_users_for_segment(14), 40)

    def test_get_users_for_segment_all_unique_values(self):
        values = set(get_users_for_segment(i) for i in range(36))
        expected = {40, 50, 60, 70, 80, 90, 100, 120, 130, 140, 150, 160, 180, 200}
        self.assertEqual(values, expected)

    def test_total_segments_calculation(self):
        total_hours = 12
        segment_minutes = 10
        total_seconds = total_hours * 3600
        segment_seconds = segment_minutes * 60
        total_segments = int(total_seconds // segment_seconds)
        self.assertEqual(total_segments, 72)

    def test_log_users_segment_format(self):
        data = {
            'segment': 1,
            'users': 100,
            'start_time': datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            'end_time': datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        }

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        tmp.write(json.dumps(data) + '\n')
        tmp.close()

        with open(tmp.name) as f:
            for line in f:
                loaded = json.loads(line)
                self.assertEqual(loaded['segment'], 1)
                self.assertEqual(loaded['users'], 100)
                self.assertIn('start_time', loaded)
                self.assertIn('end_time', loaded)
        os.unlink(tmp.name)


if __name__ == '__main__':
    unittest.main()
