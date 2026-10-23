"""Unit and integration tests for AI Weather Ensemble Engine."""

import pytest
import numpy as np

from src.ai_weather_engine import (
    AIWeatherEngine,
    snap_to_grid,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    LAT_STEP,
    LON_STEP,
    N_MEMBERS,
    FORECAST_HOURS,
    KNOWN_ASSETS,
)


def test_grid_bounds_and_snapping():
    """Verify coordinate grid domain and nearest neighbor snapping."""
    # Test Forties Charlie snapping (57.73, 0.90 -> 57.75, 1.0)
    snapped_lat, snapped_lon, lat_idx, lon_idx = snap_to_grid(57.73, 0.90)
    assert snapped_lat == 57.75
    assert snapped_lon == 1.0
    assert lat_idx >= 0
    assert lon_idx >= 0

    # Test clamping coordinates outside domain
    # Far southwest: 40.0N, -20.0W -> clamped to (49.0N, -10.0W)
    c_lat, c_lon, _, _ = snap_to_grid(40.0, -20.0)
    assert c_lat == LAT_MIN
    assert c_lon == LON_MIN

    # Far northeast: 70.0N, 10.0E -> clamped to (60.0N, 2.0E)
    c_lat, c_lon, _, _ = snap_to_grid(70.0, 10.0)
    assert c_lat == LAT_MAX
    assert c_lon == LON_MAX


def test_ensemble_quantiles_ordering():
    """Verify that P10 <= P50 <= P90 strictly holds across all variables and hours."""
    engine = AIWeatherEngine(seed=42)
    forecast = engine.generate_point_ensemble(lat=57.73, lon=0.90)

    assert forecast.members_count == N_MEMBERS
    assert len(forecast.hours) == FORECAST_HOURS
    assert len(forecast.timestamps) == FORECAST_HOURS

    for var_name in ["temperature", "wind_speed", "wave_height", "surface_pressure", "hdd"]:
        series = getattr(forecast, var_name)
        assert len(series.p10) == FORECAST_HOURS
        assert len(series.p50) == FORECAST_HOURS
        assert len(series.p90) == FORECAST_HOURS

        for h in range(FORECAST_HOURS):
            p10 = series.p10[h]
            p50 = series.p50[h]
            p90 = series.p90[h]
            # Quantiles must be monotonically non-decreasing
            assert p10 <= p50 + 1e-5, f"{var_name} at hour {h}: P10 ({p10}) > P50 ({p50})"
            assert p50 <= p90 + 1e-5, f"{var_name} at hour {h}: P50 ({p50}) > P90 ({p90})"


def test_storm_cyclogenesis_pressure_drop():
    """Verify that storm depression causes deep central pressure drop."""
    engine = AIWeatherEngine(seed=42)
    # Check storm eye central pressure over 24 hours
    central_pressures = [engine._storm_center(h, member_idx=0)[2] for h in range(FORECAST_HOURS)]
    min_pressure = min(central_pressures)
    assert min_pressure < 980.0, f"Central pressure should drop below 980 hPa, got {min_pressure}"
    # Verify explosive deepening from initial pressure
    assert central_pressures[0] - min_pressure >= 24.0, "Storm should drop > 24 hPa in 24h (explosive cyclogenesis)"


def test_forties_charlie_swh_breaches_threshold():
    """Verify that wave height (SWH) breaches 5.0m across central North Sea (Forties Charlie)."""
    engine = AIWeatherEngine(seed=42)
    forecast = engine.generate_point_ensemble(lat=57.73, lon=0.90)

    swh_p50 = np.array(forecast.wave_height.p50)
    swh_p90 = np.array(forecast.wave_height.p90)

    # SWH must breach 5.0m
    max_p50_swh = float(np.max(swh_p50))
    max_p90_swh = float(np.max(swh_p90))
    assert max_p50_swh >= 5.0, f"Ensemble median peak SWH at Forties Charlie should exceed 5.0m, got {max_p50_swh}"
    assert max_p90_swh >= 5.5, f"P90 peak SWH at Forties Charlie should exceed 5.5m, got {max_p90_swh}"

    # Wind speed should ramp up significantly
    max_wind_p50 = float(np.max(forecast.wind_speed.p50))
    assert max_wind_p50 >= 25.0, f"Wind speed ramp should exceed 25 m/s, got {max_wind_p50}"


def test_hdd_spike_breach():
    """Verify Arctic cold plunge causes HDD spike >= 0.30 for gas trading."""
    engine = AIWeatherEngine(seed=42)
    # Check gas demand center (London) and gas terminals (Bacton / St Fergus)
    forecast = engine.generate_point_ensemble(lat=51.50, lon=-0.12)
    hdd_p50 = np.array(forecast.hdd.p50)
    hdd_spike = float(np.max(hdd_p50) - np.min(hdd_p50))
    assert hdd_spike >= 0.30, f"HDD spike should be >= 0.30, got {hdd_spike}"


def test_marine_energy_alerts_evaluation():
    """Verify that risk advisory engine outputs required operational alerts."""
    engine = AIWeatherEngine(seed=42)
    report = engine.evaluate_marine_energy_alerts()

    assert report.total_alerts > 0
    assert report.critical_alerts > 0

    # Ensure Forties Charlie has a CRITICAL SWH alert
    forties_alert = next(
        (a for a in report.alerts if a.asset_id == "forties-charlie" and a.category == "MARINE_OFFSHORE"),
        None,
    )
    assert forties_alert is not None, "Forties Charlie must have a MARINE_OFFSHORE alert"
    assert forties_alert.severity == "CRITICAL"
    assert forties_alert.observed_peak_p50 >= 5.0
    assert forties_alert.threshold_value == 5.0
    assert len(forties_alert.breach_hours) > 0

    # Ensure Gas Trading HDD spike alert is present
    gas_alerts = [a for a in report.alerts if a.category == "GAS_TRADING"]
    assert len(gas_alerts) > 0
    assert any(a.threshold_value == 0.30 for a in gas_alerts)
