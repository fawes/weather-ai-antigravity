"""AI Weather Ensemble & Risk Advisory Microservice.

Generates a 10-member probabilistic ensemble weather forecast over the
UK & North Sea domain (49°N to 60°N, -10°W to 2°E, 0.25° grid) simulating
storm cyclogenesis, wind ramps, wave height (SWH) breaching 5.0m, and
energy heating degree day (HDD) demand spikes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

# Domain definition
LAT_MIN: float = 49.0
LAT_MAX: float = 60.0
LAT_STEP: float = 0.25

LON_MIN: float = -10.0
LON_MAX: float = 2.0
LON_STEP: float = 0.25

N_MEMBERS: int = 10
FORECAST_HOURS: int = 25  # 0 to 24 inclusive

# Base temperature for Heating Degree Days calculation (standard UK/EU gas trading)
HDD_BASE_TEMP: float = 15.5


@dataclass(frozen=True)
class MarineAsset:
    """Predefined critical marine energy and trading assets."""
    id: str
    name: str
    lat: float
    lon: float
    asset_type: str  # 'oil_gas_platform', 'offshore_wind', 'gas_terminal', 'demand_hub'
    critical_swh_threshold: float = 5.0  # meters


KNOWN_ASSETS: List[MarineAsset] = [
    MarineAsset(
        id="forties-charlie",
        name="Forties Charlie Platform",
        lat=57.73,
        lon=0.90,
        asset_type="oil_gas_platform",
        critical_swh_threshold=5.0,
    ),
    MarineAsset(
        id="dogger-bank-a",
        name="Dogger Bank Wind Array",
        lat=54.80,
        lon=1.90,
        asset_type="offshore_wind",
        critical_swh_threshold=4.5,
    ),
    MarineAsset(
        id="brent-charlie",
        name="Brent Field Charlie",
        lat=60.00,
        lon=1.70,
        asset_type="oil_gas_platform",
        critical_swh_threshold=5.0,
    ),
    MarineAsset(
        id="hornsea-one",
        name="Hornsea One Offshore Wind",
        lat=53.88,
        lon=1.85,
        asset_type="offshore_wind",
        critical_swh_threshold=4.0,
    ),
    MarineAsset(
        id="st-fergus-gas",
        name="St Fergus Gas Terminal",
        lat=57.58,
        lon=-1.85,
        asset_type="gas_terminal",
        critical_swh_threshold=3.5,
    ),
    MarineAsset(
        id="bacton-gas",
        name="Bacton Gas Landfall Hub",
        lat=52.87,
        lon=1.45,
        asset_type="gas_terminal",
        critical_swh_threshold=3.0,
    ),
    MarineAsset(
        id="london-demand-hub",
        name="London Energy Demand Hub",
        lat=51.50,
        lon=-0.12,
        asset_type="demand_hub",
        critical_swh_threshold=0.0,
    ),
]


class QuantileSeries(BaseModel):
    """P10, P50, and P90 confidence interval series."""
    p10: List[float] = Field(..., description="10th percentile (conservative lower bound)")
    p50: List[float] = Field(..., description="50th percentile (ensemble median)")
    p90: List[float] = Field(..., description="90th percentile (conservative upper bound)")


class EnsemblePointForecast(BaseModel):
    """24-hour ensemble forecast at a specific location."""
    requested_lat: float
    requested_lon: float
    snapped_lat: float
    snapped_lon: float
    grid_lat_idx: int
    grid_lon_idx: int
    hours: List[int]
    timestamps: List[str]
    temperature: QuantileSeries  # °C
    wind_speed: QuantileSeries   # m/s
    wave_height: QuantileSeries  # SWH in meters
    surface_pressure: QuantileSeries  # hPa
    hdd: QuantileSeries          # Heating Degree Days
    members_count: int = N_MEMBERS
    model_run: str


class AlertItem(BaseModel):
    """Risk alert item for marine energy and gas trading operations."""
    alert_id: str
    category: str  # 'MARINE_OFFSHORE', 'GAS_TRADING', 'WIND_POWER'
    severity: str  # 'CRITICAL', 'WARNING', 'ADVISORY'
    asset_id: Optional[str] = None
    asset_name: str
    lat: float
    lon: float
    trigger_variable: str
    threshold_value: float
    observed_peak_p50: float
    observed_peak_p90: float
    breach_hours: List[int]
    message: str
    recommended_action: str


class RiskAdvisoryReport(BaseModel):
    """Aggregated risk advisory report."""
    timestamp: str
    domain: str
    total_alerts: int
    critical_alerts: int
    warning_alerts: int
    alerts: List[AlertItem]


def get_grid_coordinates() -> Tuple[np.ndarray, np.ndarray]:
    """Return 1D arrays of domain latitude and longitude grid points."""
    lats = np.round(np.arange(LAT_MIN, LAT_MAX + 1e-5, LAT_STEP), 3)
    lons = np.round(np.arange(LON_MIN, LON_MAX + 1e-5, LON_STEP), 3)
    return lats, lons


def snap_to_grid(lat: float, lon: float) -> Tuple[float, float, int, int]:
    """Snap an arbitrary coordinate to nearest grid node in UK/North Sea domain.

    Returns:
        (snapped_lat, snapped_lon, lat_index, lon_index)
    """
    lats, lons = get_grid_coordinates()

    # Clamp coordinates to domain boundaries
    clamped_lat = max(LAT_MIN, min(LAT_MAX, float(lat)))
    clamped_lon = max(LON_MIN, min(LON_MAX, float(lon)))

    lat_idx = int(np.argmin(np.abs(lats - clamped_lat)))
    lon_idx = int(np.argmin(np.abs(lons - clamped_lon)))

    return float(lats[lat_idx]), float(lons[lon_idx]), lat_idx, lon_idx


class AIWeatherEngine:
    """Physics-informed 10-member probabilistic ensemble weather engine.

    Simulates a North Sea storm depression ('Storm Antigravity'):
    - Rapid cyclogenesis: surface pressure drops below 970 hPa
    - Severe gale/storm winds exceeding 30 m/s
    - Significant wave height (SWH) breaching 5.0m across Central North Sea assets
    - Arctic polar maritime surge causing sharp temperature drops and HDD spikes
    """

    def __init__(self, seed: int = 42, base_time: Optional[datetime] = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.base_time = base_time or datetime(2026, 10, 23, 12, 0, 0, tzinfo=timezone.utc)
        self.lats, self.lons = get_grid_coordinates()
        self.num_lats = len(self.lats)
        self.num_lons = len(self.lons)

        # Ensemble member perturbations (speed, lateral track offset, deepening factor)
        self.member_perturbations = self._generate_member_perturbations()

    def _generate_member_perturbations(self) -> List[Dict[str, float]]:
        """Generate physically consistent perturbations for the 10 members."""
        perturbations = []
        for m in range(N_MEMBERS):
            if m == 0:
                # Member 0: Control forecast
                p = {
                    "track_lat_offset": 0.0,
                    "track_lon_offset": 0.0,
                    "speed_factor": 1.0,
                    "deepening_factor": 1.0,
                    "wind_scaling": 1.0,
                    "temp_offset": 0.0,
                }
            else:
                # Stochastic perturbations around control trajectory
                sub_rng = np.random.default_rng(self.seed + m * 97)
                p = {
                    "track_lat_offset": float(sub_rng.normal(0.0, 0.25)),   # +/- 0.25 deg lat
                    "track_lon_offset": float(sub_rng.normal(0.0, 0.35)),   # +/- 0.35 deg lon
                    "speed_factor": float(np.clip(sub_rng.normal(1.0, 0.08), 0.85, 1.15)),
                    "deepening_factor": float(np.clip(sub_rng.normal(1.0, 0.06), 0.88, 1.14)),
                    "wind_scaling": float(np.clip(sub_rng.normal(1.0, 0.05), 0.90, 1.12)),
                    "temp_offset": float(sub_rng.normal(0.0, 0.4)),
                }
            perturbations.append(p)
        return perturbations

    def _storm_center(self, hour: float, member_idx: int = 0) -> Tuple[float, float, float]:
        """Compute storm center (lat, lon, central_pressure) at given forecast hour.

        Base trajectory moves from west of Scotland across Northern UK into
        Central North Sea (near Forties Charlie around hour 16-19) then exiting
        towards Norway.
        """
        p = self.member_perturbations[member_idx]
        eff_hour = hour * p["speed_factor"]

        # Trajectory path parameterized along hour:
        # t=0:  lat 54.8, lon -9.5 (Atlantic, West of Scotland)
        # t=8:  lat 56.0, lon -5.0 (Scottish Highlands)
        # t=16: lat 57.3, lon  0.2 (Entering Central North Sea)
        # t=20: lat 57.8, lon  1.2 (Directly passing Forties Charlie: 57.73N, 0.90E)
        # t=24: lat 58.5, lon  2.2 (Norwegian Basin)

        t_frac = np.clip(eff_hour / 24.0, 0.0, 1.2)
        base_lat = 54.8 + 3.7 * t_frac
        base_lon = -9.5 + 11.7 * t_frac

        # Add member track perturbation
        storm_lat = base_lat + p["track_lat_offset"]
        storm_lon = base_lon + p["track_lon_offset"]

        # Central pressure trajectory (explosive cyclogenesis / bomb drop):
        # Starts ~1004 hPa, deepens to ~967 hPa at t=17-19, slight filling to 975 at t=24
        deepening = math.exp(-((eff_hour - 18.0) ** 2) / (2.0 * (5.5 ** 2)))
        central_p = 1004.0 - (37.0 * p["deepening_factor"] * deepening) - (0.35 * eff_hour)

        return float(storm_lat), float(storm_lon), float(central_p)

    def simulate_point(
        self,
        lat: float,
        lon: float,
        member_idx: int,
        hour: int,
    ) -> Dict[str, float]:
        """Simulate weather variables at (lat, lon) for one member and one hour."""
        p = self.member_perturbations[member_idx]
        storm_lat, storm_lon, storm_p = self._storm_center(hour, member_idx)

        # Distance to storm center in kilometers (haversine / flat-earth approximation)
        d_lat_km = (lat - storm_lat) * 111.0
        d_lon_km = (lon - storm_lon) * 111.0 * math.cos(math.radians((lat + storm_lat) / 2.0))
        r_km = math.hypot(d_lat_km, d_lon_km)

        # Direction angle from storm center (for asymmetric wind field in southern flank)
        bearing = math.atan2(d_lat_km, d_lon_km)  # radians from East

        # 1. Surface Pressure (Holland-style vortex profile)
        r_max_km = 90.0  # radius of maximum winds
        ambient_p = 1016.0
        delta_p = ambient_p - storm_p
        vortex_factor = math.exp(-(r_km / (r_max_km * 2.2)) ** 1.3)
        pressure = ambient_p - delta_p * vortex_factor

        # 2. 10m Wind Speed
        # Cyclostrophic/gradient wind + strong southern quadrant gale jet
        gradient_wind = (delta_p * 0.72) * (r_km / (r_max_km + 1e-4)) * math.exp(-0.5 * (r_km / r_max_km) ** 1.4)
        # Asymmetry: southern/eastern quadrant has higher winds due to storm translation
        quadrant_boost = 1.0 - 0.28 * math.sin(bearing)  # bearing < 0 is southern half
        base_wind = 7.5 + (gradient_wind * quadrant_boost * p["wind_scaling"])

        # Small stochastic turbulence for realism
        seed_offset = int((lat * 100 + lon * 10 + hour * 10 + member_idx) % 100000)
        local_rng = np.random.default_rng(self.seed + seed_offset)
        wind_gust_noise = float(local_rng.normal(0.0, 0.6))
        wind_speed = float(np.clip(base_wind + wind_gust_noise, 3.5, 36.0))

        # 3. Significant Wave Height (SWH, meters)
        # Physical dynamic wave growth: SWH depends on wind speed and offshore fetch
        # Central North Sea / Forties (lat > 56, lon > -1) has massive oceanic fetch
        is_central_north_sea = (lat >= 55.5 and lat <= 59.5 and lon >= -0.5 and lon <= 3.0)
        fetch_multiplier = 1.22 if is_central_north_sea else 0.95

        # Calibrated wave growth: 10 m/s -> ~1.8m SWH; 28 m/s -> ~6.3m SWH (breaching 5.0m threshold)
        synoptic_wave = 1.2 + (0.0058 * (wind_speed ** 1.92) * fetch_multiplier)

        # Lag effect: ocean swells peak slightly as energy accumulates
        lag_weight = min(1.0, 0.4 + (hour / 20.0))
        swh = float(np.clip(synoptic_wave * lag_weight, 0.8, 9.5))

        # 4. 2m Temperature (°C)
        # Cold front passage behind the low: cold polar maritime flow surges from NW
        is_behind_front = (d_lon_km < 40.0) and (hour >= 10)
        if is_behind_front:
            temp = 5.0 - 0.22 * (lat - 50.0) + p["temp_offset"] - (0.22 * max(0, hour - 10))
        else:
            temp = 12.5 - 0.28 * (lat - 50.0) + p["temp_offset"] - (0.12 * hour)
        temp = float(np.clip(temp, -1.0, 18.0))

        # 5. Heating Degree Days (HDD)
        # Hourly contribution to HDD = max(0, 15.5 - T) / 24
        hdd = float(max(0.0, (HDD_BASE_TEMP - temp) / 24.0))

        return {
            "pressure": round(pressure, 2),
            "wind_speed": round(wind_speed, 2),
            "swh": round(swh, 2),
            "temperature": round(temp, 2),
            "hdd": round(hdd, 4),
        }

    def generate_point_ensemble(self, lat: float, lon: float) -> EnsemblePointForecast:
        """Generate 24-hour 10-member ensemble forecast snapped to grid."""
        snapped_lat, snapped_lon, lat_idx, lon_idx = snap_to_grid(lat, lon)

        hours = list(range(FORECAST_HOURS))
        timestamps = [
            (self.base_time.replace(minute=0, second=0) + np.timedelta64(h, "h").astype(object)).isoformat()
            for h in hours
        ]

        # Allocate arrays: [FORECAST_HOURS, N_MEMBERS]
        pressures = np.zeros((FORECAST_HOURS, N_MEMBERS), dtype=float)
        wind_speeds = np.zeros((FORECAST_HOURS, N_MEMBERS), dtype=float)
        wave_heights = np.zeros((FORECAST_HOURS, N_MEMBERS), dtype=float)
        temperatures = np.zeros((FORECAST_HOURS, N_MEMBERS), dtype=float)
        hdds = np.zeros((FORECAST_HOURS, N_MEMBERS), dtype=float)

        for h in hours:
            for m in range(N_MEMBERS):
                res = self.simulate_point(snapped_lat, snapped_lon, m, h)
                pressures[h, m] = res["pressure"]
                wind_speeds[h, m] = res["wind_speed"]
                wave_heights[h, m] = res["swh"]
                temperatures[h, m] = res["temperature"]
                hdds[h, m] = res["hdd"]

        def _calc_quantiles(arr: np.ndarray, round_digits: int = 2) -> QuantileSeries:
            p10 = np.round(np.percentile(arr, 10, axis=1), round_digits).tolist()
            p50 = np.round(np.percentile(arr, 50, axis=1), round_digits).tolist()
            p90 = np.round(np.percentile(arr, 90, axis=1), round_digits).tolist()
            return QuantileSeries(p10=p10, p50=p50, p90=p90)

        return EnsemblePointForecast(
            requested_lat=lat,
            requested_lon=lon,
            snapped_lat=snapped_lat,
            snapped_lon=snapped_lon,
            grid_lat_idx=lat_idx,
            grid_lon_idx=lon_idx,
            hours=hours,
            timestamps=timestamps,
            temperature=_calc_quantiles(temperatures, 2),
            wind_speed=_calc_quantiles(wind_speeds, 2),
            wave_height=_calc_quantiles(wave_heights, 2),
            surface_pressure=_calc_quantiles(pressures, 2),
            hdd=_calc_quantiles(hdds, 4),
            members_count=N_MEMBERS,
            model_run=self.base_time.isoformat(),
        )

    def evaluate_marine_energy_alerts(self) -> RiskAdvisoryReport:
        """Scan critical marine energy and trading assets for threshold breaches.

        Alert triggers:
        1. Marine/Offshore: SWH >= 5.0m offshore (e.g. Forties Charlie in central North Sea)
        2. Gas Trading: HDD spike >= 0.30 over forecast window
        3. Offshore Wind Cutoff: 10m wind speed >= 25.0 m/s
        """
        alerts: List[AlertItem] = []

        for asset in KNOWN_ASSETS:
            forecast = self.generate_point_ensemble(asset.lat, asset.lon)

            # Check SWH threshold (specifically breaching 5.0m for offshore assets)
            if asset.critical_swh_threshold > 0.0 and asset.asset_type in ("oil_gas_platform", "offshore_wind", "gas_terminal"):
                swh_p50 = np.array(forecast.wave_height.p50)
                swh_p90 = np.array(forecast.wave_height.p90)
                swh_breach_indices = np.where(swh_p50 >= asset.critical_swh_threshold)[0]
                swh_p90_breach_indices = np.where(swh_p90 >= asset.critical_swh_threshold)[0]

                if len(swh_breach_indices) > 0 or len(swh_p90_breach_indices) > 0:
                    is_p50_breached = len(swh_breach_indices) > 0
                    active_hours = swh_breach_indices.tolist() if is_p50_breached else swh_p90_breach_indices.tolist()
                    peak_p50 = float(np.max(swh_p50))
                    peak_p90 = float(np.max(swh_p90))
                    severity = "CRITICAL" if peak_p50 >= 5.0 else "WARNING"

                    alerts.append(
                        AlertItem(
                            alert_id=f"SWH-{asset.id}-{severity.lower()}",
                            category="MARINE_OFFSHORE",
                            severity=severity,
                            asset_id=asset.id,
                            asset_name=asset.name,
                            lat=asset.lat,
                            lon=asset.lon,
                            trigger_variable="significant_wave_height",
                            threshold_value=asset.critical_swh_threshold,
                            observed_peak_p50=peak_p50,
                            observed_peak_p90=peak_p90,
                            breach_hours=active_hours,
                            message=(
                                f"Significant Wave Height (SWH) threshold breach at {asset.name}. "
                                f"Ensemble median reaches {peak_p50:.2f}m (P90: {peak_p90:.2f}m) "
                                f"exceeding critical operating threshold ({asset.critical_swh_threshold:.1f}m)."
                            ),
                            recommended_action=(
                                "Halt personnel transfers via crew transfer vessels (CTV), initiate platform "
                                "storm down-manning, and secure offshore crane operations."
                            ),
                        )
                    )

            # Check Wind Turbine Cutoff (>= 25 m/s) for offshore wind farms
            if asset.asset_type == "offshore_wind":
                wind_p50 = np.array(forecast.wind_speed.p50)
                wind_p90 = np.array(forecast.wind_speed.p90)
                cutoff_indices = np.where(wind_p50 >= 25.0)[0]
                if len(cutoff_indices) > 0:
                    peak_w_p50 = float(np.max(wind_p50))
                    peak_w_p90 = float(np.max(wind_p90))
                    alerts.append(
                        AlertItem(
                            alert_id=f"WIND-CUTOFF-{asset.id}",
                            category="WIND_POWER",
                            severity="CRITICAL",
                            asset_id=asset.id,
                            asset_name=asset.name,
                            lat=asset.lat,
                            lon=asset.lon,
                            trigger_variable="wind_speed_10m",
                            threshold_value=25.0,
                            observed_peak_p50=peak_w_p50,
                            observed_peak_p90=peak_w_p90,
                            breach_hours=cutoff_indices.tolist(),
                            message=(
                                f"Storm wind ramp exceeds 25 m/s storm shutdown threshold at {asset.name}. "
                                f"Ensemble median peak: {peak_w_p50:.1f} m/s (P90: {peak_w_p90:.1f} m/s)."
                            ),
                            recommended_action=(
                                "Prepare grid balancing reserves for sudden offshore wind power trip/curtailment."
                            ),
                        )
                    )

            # Check Heating Degree Days (HDD) spike (>= 0.30) for gas trading
            # Calculate cumulative HDD or 24-hour delta HDD spike
            hdd_p50 = np.array(forecast.hdd.p50)
            hdd_p90 = np.array(forecast.hdd.p90)
            # Delta between end of forecast (cold Arctic maritime air) and start of forecast
            hdd_spike_p50 = float(np.max(hdd_p50) - np.min(hdd_p50))
            hdd_spike_p90 = float(np.max(hdd_p90) - np.min(hdd_p90))

            if hdd_spike_p50 >= 0.30 or hdd_spike_p90 >= 0.30:
                severity = "CRITICAL" if hdd_spike_p50 >= 0.30 else "WARNING"
                spike_hours = np.where((hdd_p50 - np.min(hdd_p50)) >= 0.30)[0].tolist()
                alerts.append(
                    AlertItem(
                        alert_id=f"HDD-SPIKE-{asset.id}",
                        category="GAS_TRADING",
                        severity=severity,
                        asset_id=asset.id,
                        asset_name=asset.name,
                        lat=asset.lat,
                        lon=asset.lon,
                        trigger_variable="hdd_delta_spike",
                        threshold_value=0.30,
                        observed_peak_p50=round(hdd_spike_p50, 4),
                        observed_peak_p90=round(hdd_spike_p90, 4),
                        breach_hours=spike_hours or list(range(12, FORECAST_HOURS)),
                        message=(
                            f"Rapid Arctic cold plunge causing Heating Degree Day (HDD) spike "
                            f"(+{hdd_spike_p50:.3f} HDD) at {asset.name}. Drives sudden spike in UK gas heating demand."
                        ),
                        recommended_action=(
                            "Adjust UK NBP (National Balancing Point) gas contracts, hedge prompt power demand, "
                            "and alert terminal linepack storage operators."
                        ),
                    )
                )

        crit_count = sum(1 for a in alerts if a.severity == "CRITICAL")
        warn_count = sum(1 for a in alerts if a.severity == "WARNING")

        return RiskAdvisoryReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            domain=f"UK & North Sea [{LAT_MIN}N-{LAT_MAX}N, {LON_MIN}W-{LON_MAX}E, res {LAT_STEP}°]",
            total_alerts=len(alerts),
            critical_alerts=crit_count,
            warning_alerts=warn_count,
            alerts=alerts,
        )
