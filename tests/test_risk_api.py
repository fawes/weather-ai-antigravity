"""Integration tests for FastAPI Weather & Risk Advisory Service."""

import pytest
from fastapi.testclient import TestClient

from src.risk_api import app

client = TestClient(app)


def test_healthz_endpoint():
    """Verify GET /healthz returns 200 and healthy ensemble status."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ensemble_status"]["members"] == 10
    assert data["ensemble_status"]["forecast_horizon_hours"] == 24
    assert data["ensemble_status"]["quantiles"] == ["P10", "P50", "P90"]
    assert data["domain"]["region"] == "UK & North Sea"
    assert data["domain"]["lat_bounds"] == [49.0, 60.0]
    assert data["domain"]["lon_bounds"] == [-10.0, 2.0]
    assert data["domain"]["resolution_deg"] == 0.25


def test_forecast_ensemble_forties_charlie():
    """Verify GET /api/forecast/ensemble returns 200 and valid snapped time series."""
    # Query Forties Charlie location (57.73N, 0.90E)
    response = client.get("/api/forecast/ensemble?lat=57.73&lon=0.90")
    assert response.status_code == 200
    data = response.json()

    assert data["requested_lat"] == 57.73
    assert data["requested_lon"] == 0.90
    assert data["snapped_lat"] == 57.75
    assert data["snapped_lon"] == 1.00
    assert data["members_count"] == 10

    assert len(data["hours"]) == 25
    assert len(data["timestamps"]) == 25

    # Check that all variables have p10, p50, p90 arrays of length 25
    for var in ["temperature", "wind_speed", "wave_height", "surface_pressure", "hdd"]:
        assert var in data
        assert len(data[var]["p10"]) == 25
        assert len(data[var]["p50"]) == 25
        assert len(data[var]["p90"]) == 25

    # Forties Charlie SWH peak should breach 5.0m
    max_swh_p50 = max(data["wave_height"]["p50"])
    assert max_swh_p50 >= 5.0, f"Forties Charlie peak SWH should exceed 5.0m, got {max_swh_p50}"


def test_forecast_ensemble_boundary_clamping():
    """Verify endpoint handles boundary and out-of-domain snapping."""
    response = client.get("/api/forecast/ensemble?lat=48.0&lon=-12.0")
    assert response.status_code == 200
    data = response.json()
    assert data["snapped_lat"] == 49.0
    assert data["snapped_lon"] == -10.0


def test_forecast_ensemble_validation_error():
    """Verify endpoint rejects invalid parameters (e.g. missing lat)."""
    response = client.get("/api/forecast/ensemble?lat=55.0")
    assert response.status_code == 422  # Unprocessable Entity (missing lon)


def test_marine_energy_alerts_endpoint():
    """Verify GET /api/alerts/marine-energy returns 200 with critical advisory."""
    response = client.get("/api/alerts/marine-energy")
    assert response.status_code == 200
    data = response.json()

    assert "total_alerts" in data
    assert "critical_alerts" in data
    assert data["total_alerts"] > 0
    assert data["critical_alerts"] > 0

    alerts = data["alerts"]
    # Check Forties Charlie SWH alert
    swh_alert = next((a for a in alerts if a["asset_id"] == "forties-charlie" and a["category"] == "MARINE_OFFSHORE"), None)
    assert swh_alert is not None
    assert swh_alert["severity"] == "CRITICAL"
    assert swh_alert["threshold_value"] == 5.0
    assert swh_alert["observed_peak_p50"] >= 5.0

    # Check Gas trading alert
    gas_alert = next((a for a in alerts if a["category"] == "GAS_TRADING"), None)
    assert gas_alert is not None
    assert gas_alert["threshold_value"] == 0.30


def test_assets_endpoint():
    """Verify GET /api/assets returns list of key marine energy assets."""
    response = client.get("/api/assets")
    assert response.status_code == 200
    assets = response.json()
    assert len(assets) >= 5
    asset_ids = [a["id"] for a in assets]
    assert "forties-charlie" in asset_ids
    assert "dogger-bank-a" in asset_ids
    assert "st-fergus-gas" in asset_ids


def test_storm_track_endpoint():
    """Verify GET /api/storm-track returns hourly trajectory of storm center."""
    response = client.get("/api/storm-track")
    assert response.status_code == 200
    data = response.json()
    assert data["storm_name"] == "Storm Antigravity"
    assert len(data["track"]) == 25
    # Verify storm tracks east-northeast
    assert data["track"][0]["lon"] < data["track"][-1]["lon"]


def test_dashboard_serving():
    """Verify GET / serves web/index.html with HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<!DOCTYPE html>" in response.text
    assert "AI Weather Ensemble & Risk Advisory" in response.text
