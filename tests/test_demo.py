"""Numerical checks for the educational model; no network calls required."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pv_demo import (
    DC_AC_RATIO, DESIGN_COLUMNS, LOWER_BOUNDS, UPPER_BOUNDS,
    Design, PVSimulator, build_model, load_weather, nondominated,
    objective_matrix, optimize_designs, rank_compromises,
    summarize_power, validate_weather, weather_for_model,
)


class NumericalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weather, cls.metadata = load_weather()

    def test_constant_power_has_correct_energy_and_zero_variability(self):
        power = pd.Series(2.0, index=self.weather.index)
        metrics = summarize_power(power)
        self.assertEqual(metrics["annual_energy_kwh"], 2 * 8760)
        self.assertEqual(metrics["daily_cv"], 0)

    def test_daily_variability_matches_analytic_sequence_and_ignores_size(self):
        # Construct 365 known daily energy values; every hour carries 1/24.
        daily = np.arange(1, 366, dtype=float)
        power = pd.Series(np.repeat(daily / 24, 24), index=self.weather.index)
        expected = np.std(daily, ddof=0) / np.mean(daily)
        self.assertAlmostEqual(summarize_power(power)["daily_cv"], expected)
        self.assertAlmostEqual(summarize_power(7 * power)["daily_cv"], expected)

    def test_reference_scaling_matches_full_size_pvlib_and_ac_rating(self):
        simulator = PVSimulator(self.weather)
        design = Design(25, 120, 73)
        model = build_model(design)
        model.run_model(weather_for_model(self.weather))
        direct_kw = (model.results.ac / 1000).clip(lower=0)
        np.testing.assert_allclose(simulator.power(design), direct_kw,
                                   rtol=1e-10, atol=1e-10)
        self.assertGreater(direct_kw.sum(), 0)
        self.assertLessEqual(direct_kw.max(), design.size_kwp / DC_AC_RATIO + 1e-10)
        inverter = model.system.inverter_parameters
        self.assertAlmostEqual(inverter["pdc0"] * inverter["eta_inv_nom"],
                               design.size_kwp * 1000 / DC_AC_RATIO)

    def test_pvgis_offset_is_applied_without_changing_weather_values(self):
        aligned = weather_for_model(self.weather)
        self.assertEqual(aligned.index[0].minute, 30)
        self.assertTrue(((aligned.index - self.weather.index) == pd.Timedelta(minutes=30)).all())
        np.testing.assert_array_equal(aligned.to_numpy(), self.weather.to_numpy())
        self.assertEqual(aligned.resample("D").size().tolist(), [24] * 365)

    def test_weather_rejects_missing_hours_nan_naive_time_and_negative_values(self):
        bad_cases = [self.weather.iloc[1:].copy(), self.weather.copy(),
                     self.weather.copy(), self.weather.copy()]
        bad_cases[1].iloc[12, 0] = np.nan
        bad_cases[2].index = bad_cases[2].index.tz_localize(None)
        bad_cases[3].iloc[12, 0] = -1
        for bad_weather in bad_cases:
            with self.subTest(rows=len(bad_weather)):
                with self.assertRaises(ValueError):
                    validate_weather(bad_weather)

    def test_offline_cache_and_checksum_validation(self):
        # Fail immediately if the supposedly offline path attempts a request.
        with patch("pvlib.iotools.get_pvgis_tmy", side_effect=AssertionError("network")):
            cached, _ = load_weather()
            pd.testing.assert_frame_equal(cached, self.weather)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "weather_metadata.json").write_text(
                json.dumps(self.metadata), encoding="utf-8",
            )
            (target / "weather.csv").write_text("corrupted cache", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_weather(target)

    def test_dominance_on_hand_calculated_candidates(self):
        # A and B trade energy for cost; C is strictly dominated by A.
        frame = pd.DataFrame([
            {"tilt_deg": 10, "azimuth_deg": 180, "size_kwp": 10,
             "annual_energy_kwh": 10, "investment_usd": 10, "daily_cv": 0.2},
            {"tilt_deg": 20, "azimuth_deg": 180, "size_kwp": 20,
             "annual_energy_kwh": 20, "investment_usd": 20, "daily_cv": 0.2},
            {"tilt_deg": 30, "azimuth_deg": 180, "size_kwp": 30,
             "annual_energy_kwh": 9, "investment_usd": 11, "daily_cv": 0.3},
        ])
        result = nondominated(frame)
        self.assertEqual(set(result.tilt_deg), {10, 20})

    def test_preference_ranking_is_unit_invariant_and_rejects_bad_weights(self):
        frame = pd.DataFrame({
            "annual_energy_kwh": [1000, 2000, 1500],
            "investment_usd": [100, 300, 180],
            "daily_cv": [0.2, 0.15, 0.12],
        })
        ranked = rank_compromises(frame)
        converted = frame * np.array([1 / 1000, 1 / 1000, 100])
        np.testing.assert_allclose(
            ranked.compromise_score,
            rank_compromises(converted).compromise_score,
        )
        for weights in [(0, 0, 0), (-1, 1, 1), (1, 2), (np.nan, 1, 1)]:
            with self.assertRaises(ValueError):
                rank_compromises(frame, weights)

    def test_seeded_optimizer_bounds_repeatability_and_independent_dominance(self):
        runs = [optimize_designs(PVSimulator(self.weather), population=8,
                                generations=2, seed=42) for _ in range(2)]
        first_pareto, first_evaluated, _ = runs[0]
        pd.testing.assert_frame_equal(first_pareto, runs[1][0])
        pd.testing.assert_frame_equal(first_evaluated, runs[1][1])
        designs = first_evaluated[DESIGN_COLUMNS].to_numpy()
        self.assertTrue((designs >= LOWER_BOUNDS).all())
        self.assertTrue((designs <= UPPER_BOUNDS).all())
        self.assertFalse(first_pareto.empty)
        objectives = objective_matrix(first_evaluated)
        for row in objective_matrix(first_pareto):
            dominated = (objectives <= row).all(axis=1) & (objectives < row).any(axis=1)
            self.assertFalse(dominated.any())


if __name__ == "__main__":
    unittest.main()
