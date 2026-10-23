"""FastAPI Service for AI Weather Ensemble & Risk Advisory Microservice.

Exposes endpoints:
- GET /healthz: Health check & ensemble status
- GET /api/forecast/ensemble: Snaps lat/lon and returns 24-hour P10/P50/P90 time series
- GET /api/alerts/marine-energy: Critical threshold alerts (SWH >= 5.0m, HDD spike >= 0.30)
- GET /api/assets: Catalog of key offshore and trading assets
- GET /api/storm-track: 24-hour storm center trajectory for geospatial visualization
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.ai_weather_engine import (
    AIWeatherEngine,
    AlertItem,
    EnsemblePointForecast,
    KNOWN_ASSETS,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    RiskAdvisoryReport,
    snap_to_grid,
)

app = FastAPI(
    title="AI Weather Ensemble & Risk Advisory API",
    description=(
        "Probabilistic AI Weather forecasting and marine/energy risk advisory service "
        "covering the UK & North Sea basin."
    ),
    version="1.0.0",
)

# Enable CORS for web dashboard and cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton weather engine instance
weather_engine = AIWeatherEngine()

# Web dashboard static files directory
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@app.get("/healthz", summary="Service health check and ensemble status")
def healthz() -> Dict[str, Any]:
    """Return API health status and ensemble configuration."""
    return {
        "status": "healthy",
        "service": "AI Weather Ensemble & Risk Advisory Microservice",
        "version": "1.0.0",
        "ensemble_status": {
            "members": 10,
            "forecast_horizon_hours": 24,
            "quantiles": ["P10", "P50", "P90"],
            "model_run": weather_engine.base_time.isoformat(),
        },
        "domain": {
            "region": "UK & North Sea",
            "lat_bounds": [LAT_MIN, LAT_MAX],
            "lon_bounds": [LON_MIN, LON_MAX],
            "resolution_deg": 0.25,
            "grid_points_lat": weather_engine.num_lats,
            "grid_points_lon": weather_engine.num_lons,
        },
        "active_storm": {
            "name": "Storm Antigravity",
            "type": "Explosive Cyclogenesis (Weather Bomb)",
            "peak_central_pressure_hpa": 967.5,
        },
        "monitored_assets_count": len(KNOWN_ASSETS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get(
    "/api/forecast/ensemble",
    response_model=EnsemblePointForecast,
    summary="Get 24-hour ensemble forecast snapped to nearest grid point",
)
def get_ensemble_forecast(
    lat: float = Query(..., description="Latitude coordinate (e.g. 57.73 for Forties Charlie)", ge=45.0, le=65.0),
    lon: float = Query(..., description="Longitude coordinate (e.g. 0.90 for Forties Charlie)", ge=-15.0, le=10.0),
) -> EnsemblePointForecast:
    """Snap coordinates to nearest 0.25° grid node and return 24h P10/P50/P90 envelope.

    Forecast variables included:
    - Temperature (°C)
    - 10m Wind Speed (m/s)
    - Significant Wave Height (m)
    - Surface Pressure (hPa)
    - Heating Degree Days (HDD)
    """
    try:
        forecast = weather_engine.generate_point_ensemble(lat, lon)
        return forecast
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generating ensemble forecast: {str(exc)}")


@app.get(
    "/api/alerts/marine-energy",
    response_model=RiskAdvisoryReport,
    summary="Get active threshold alerts for marine energy and gas trading",
)
def get_marine_energy_alerts() -> RiskAdvisoryReport:
    """Evaluate and return critical operational threshold alerts.

    Alert rules:
    - Marine Offshore: Significant Wave Height (SWH) >= 5.0m
    - Gas Trading: Heating Degree Days (HDD) spike >= 0.30
    - Offshore Wind: Wind ramp-up >= 25.0 m/s cutoff
    """
    try:
        report = weather_engine.evaluate_marine_energy_alerts()
        return report
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generating risk advisory: {str(exc)}")


@app.get("/api/assets", summary="List predefined marine energy and trading assets")
def get_assets() -> List[Dict[str, Any]]:
    """Return catalogue of monitored platforms, wind arrays, and gas terminals."""
    return [
        {
            "id": asset.id,
            "name": asset.name,
            "lat": asset.lat,
            "lon": asset.lon,
            "asset_type": asset.asset_type,
            "critical_swh_threshold": asset.critical_swh_threshold,
        }
        for asset in KNOWN_ASSETS
    ]


@app.get("/api/storm-track", summary="Get 24-hour storm center trajectory")
def get_storm_track() -> Dict[str, Any]:
    """Return hourly trajectory of storm center for geospatial map rendering."""
    track = []
    for h in range(25):
        s_lat, s_lon, s_p = weather_engine._storm_center(h, member_idx=0)
        track.append({
            "hour": h,
            "lat": round(s_lat, 3),
            "lon": round(s_lon, 3),
            "central_pressure_hpa": round(s_p, 1),
        })
    return {
        "storm_name": "Storm Antigravity",
        "description": "Explosive North Sea cyclogenesis with central pressure drop < 970 hPa",
        "track": track,
    }


# Serve the web dashboard directly from root if index.html exists
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, summary="Web Dashboard")
    def serve_dashboard():
        index_file = WEB_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse("<h1>Web dashboard index.html not found</h1>", status_code=404)
