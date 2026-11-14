"""Synthetic-only D1 regression tests; no NASA files needed in CI."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiment_d1 import analyze, digest, ensure_timeline, flat_runs, load_rules, load_source, run


HEADERS = ['Controller_Time_GMT','Temp_degC_Ground','Temp_degC_ISS',
           'RH_percent_Ground','RH_percent_ISS','CO2_ppm_Ground','CO2_ppm_ISS']


def sample(times=None, co2=None):
    times = times or ['2020-01-01 00:00','2020-01-01 00:05','2020-01-01 00:10','2020-01-01 00:15']
    co2 = co2 or [3999,4000,4001,4002]
    return pd.DataFrame({'Controller_Time_GMT':pd.to_datetime(times,utc=True),
        'Temp_degC_Ground':[20,21,22,23],'Temp_degC_ISS':[20,20,20,20],
        'RH_percent_Ground':[30,31,32,33],'RH_percent_ISS':[40,40,40,40],
        'CO2_ppm_Ground':[500,501,502,503],'CO2_ppm_ISS':co2})


class D1Tests(unittest.TestCase):
    def setUp(self):
        self.rules,_ = load_rules()
        self.rules['flat_runs']['min_consecutive_rows']=3
        self.spec={'kind':'telemetry'}

    def test_co2_comparison_and_interval_weighting(self):
        x = sample(times=['2020-01-01 00:00','2020-01-01 00:05','2020-01-01 00:07','2020-01-01 00:15'])
        out = analyze(x,self.spec,self.rules,'RR-1')
        self.assertEqual(out['co2_above_4000_rows'],2)
        self.assertEqual(out['co2_at_or_above_4000_rows'],3)
        self.assertEqual(out['co2_above_4000_supported_minutes'],8)
        self.assertEqual(out['supported_minutes'],15)
        self.assertEqual(out['intervals_shorter_than_nominal'],1)

    def test_gap_exact_boundary_and_above(self):
        x = sample(times=['2020-01-01 00:00','2020-01-01 00:10','2020-01-01 00:20','2020-01-01 00:31'])
        out = analyze(x,self.spec,self.rules,'RR-1')
        self.assertEqual(out['gaps_over_10_min'],1)
        self.assertEqual(out['supported_minutes'],20)
        self.assertEqual(out['unknown_gap_minutes'],11)

    def test_multi_channel_runs_start_end(self):
        x = sample()
        runs = flat_runs(x,['Temp_degC_ISS','RH_percent_ISS'],3)
        self.assertEqual(len(runs),2)
        self.assertTrue(all(v['start_row']==0 and v['end_row']==3 for v in runs))
        self.assertEqual(analyze(x,self.spec,self.rules,'RR-1')['multi_channel_flat_rows'],4)

    def test_no_double_count_overlapping_flags(self):
        x=sample()
        out=analyze(x,self.spec,self.rules,'RR-1')
        self.assertEqual(out['quality_flagged_minutes'],15)
        self.assertEqual(out['quality_unflagged_minutes'],0)

    def test_rejects_bad_time_header_and_numbers_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.csv'
            cases=[]
            cases.append(sample(times=['2020-01-01 00:00','2020-01-01 00:05','2020-01-01 00:05','2020-01-01 00:15']))
            cases.append(sample(times=['2020-01-01 00:00','2020-01-01 00:10','2020-01-01 00:05','2020-01-01 00:15']))
            cases.append(sample().rename(columns={'CO2_ppm_ISS':'co2'}))
            cases.append(sample().assign(CO2_ppm_ISS=[3999,4000,'bad',4002]))
            for frame in cases:
                frame.to_csv(path,index=False,date_format='%Y-%m-%d %H:%M')
                spec={'kind':'telemetry','sha256':digest(path)}
                with self.assertRaises((ValueError,TypeError)):
                    load_source(path,spec,self.rules)
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                load_source(path,{'kind':'telemetry','sha256':'0'*64},self.rules)

    def test_summary_rejected_as_timeline(self):
        with self.assertRaisesRegex(ValueError,'summary only'):
            ensure_timeline({'kind':'summary_only'})

    def test_rule_phase_requires_citation(self):
        with tempfile.TemporaryDirectory() as folder:
            r=copy.deepcopy(self.rules)
            r['mission_phases']['RR-1']['verified']=True
            p=Path(folder)/'rules.json';p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'citations'):
                load_rules(p)

    def test_output_dir_is_isolated(self):
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
            with tempfile.TemporaryDirectory() as folder:
                p=Path(folder)
                for name in ['RR-1_EDA_Telemetry_table.csv','RR-6_EDA_Telemetry_table.csv','RR-9_EDA_Summary_table.csv']:
                    (p/name).write_text('fake')
                run(p,output_dir=p/'astrostress_output')


if __name__=='__main__':
    unittest.main()
