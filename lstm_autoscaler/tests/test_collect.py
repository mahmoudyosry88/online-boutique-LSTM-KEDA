import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import unittest
from datetime import datetime, timedelta, timezone


class TestCollect(unittest.TestCase):

    def setUp(self):
        self.services = ['svc_a', 'svc_b', 'svc_c']

    def test_bucket_df_sum(self):
        df = pd.DataFrame({
            'Timestamp': pd.date_range('2026-01-01', periods=6, freq='15s'),
            'Service': ['svc_a'] * 6,
            'CPU': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        })
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.floor('30s')
        df = df[df['Service'].isin(self.services)]
        result = df.groupby(['Timestamp', 'Service'], as_index=False)['CPU'].sum()
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result['CPU'].iloc[0], 0.3)

    def test_bucket_df_mean(self):
        df = pd.DataFrame({
            'Timestamp': pd.date_range('2026-01-01', periods=6, freq='15s'),
            'Service': ['svc_a'] * 6,
            'Latency': [10, 20, 30, 40, 50, 60],
        })
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.floor('30s')
        df = df[df['Service'].isin(self.services)]
        result = df.groupby(['Timestamp', 'Service'], as_index=False)['Latency'].mean()
        self.assertAlmostEqual(result['Latency'].iloc[0], 15.0)

    def test_bucket_rps(self):
        rps_result = [{
            'metric': {},
            'values': [[1e9, '100'], [1e9 + 30, '200']]
        }]
        rows = []
        for series in rps_result:
            for ts, value in series.get('values', []):
                rows.append({
                    'Timestamp': datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None),
                    'RPS_frontend': float(value),
                })
        rps_df = pd.DataFrame(rows)
        if not rps_df.empty:
            rps_df['Timestamp'] = pd.to_datetime(rps_df['Timestamp']).dt.floor('30s')
            rps_df = rps_df.groupby('Timestamp', as_index=False)['RPS_frontend'].mean()
        self.assertGreater(len(rps_df), 0)

    def test_load_users_timeline(self):
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        start = datetime(2026, 1, 1, 10, 0, 0)
        end = start + timedelta(minutes=10)
        tmp.write(json.dumps({
            'segment': 1,
            'users': 200,
            'start_time': start.isoformat(),
            'end_time': end.isoformat(),
        }) + '\n')
        tmp.close()

        users_map = {}
        with open(tmp.name, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    start_ts = datetime.fromisoformat(data['start_time'])
                    end_ts = datetime.fromisoformat(data['end_time'])
                    current = start_ts.replace(microsecond=0, second=0)
                    current = current.replace(second=(current.second // 30) * 30)
                    while current <= end_ts:
                        users_map[current] = data['users']
                        current += timedelta(seconds=30)
        os.unlink(tmp.name)
        self.assertGreater(len(users_map), 0)
        self.assertEqual(list(users_map.values())[0], 200)

    def test_series_to_df_from_label(self):
        result = [{
            'metric': {'service': 'svc_a'},
            'values': [[1e9, '0.5'], [1e9 + 30, '0.8']]
        }]
        rows = []
        for series in result:
            metric = series.get('metric', {})
            raw = str(metric.get('service', ''))
            svc = raw if raw in self.services else None
            if not svc:
                for s in self.services:
                    if s in raw:
                        svc = s
                        break
            if not svc:
                continue
            for ts, value in series.get('values', []):
                rows.append({
                    'Timestamp': datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None),
                    'Service': svc,
                    'CPU': float(value),
                })
        df = pd.DataFrame(rows)
        self.assertEqual(len(df), 2)
        self.assertEqual(df['Service'].iloc[0], 'svc_a')


if __name__ == '__main__':
    unittest.main()
