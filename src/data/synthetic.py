"""Synthetic data generator for the Air Quality Forecasting project.

Produces 3 stations × n_years of hourly PM2.5-driven air quality and
meteorological data that is physically plausible for Delhi.  The output
DataFrames match the standard schemas defined in docs/INTERFACES.md exactly.

All randomness is seeded so the same config always yields identical frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic(
    config: dict = CONFIG,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (air_quality_df, weather_df, events_df) in the standard schemas.

    Parameters
    ----------
    config:
        Project configuration dict (defaults to the module-level CONFIG).
    end_date:
        ISO-8601 date string such as "2024-12-31".  Defaults to today's date
        (``pandas.Timestamp.today()`` truncated to midnight).

    Returns
    -------
    aq_df : DataFrame
        Columns: station_id, station_name, city, lat, lon, timestamp,
        pm25, pm10, no2, so2, co, o3
    weather_df : DataFrame
        Columns: station_id, timestamp, temp_c, humidity_pct, wind_speed_ms,
        wind_dir_deg, pressure_hpa, precip_mm, cloud_pct, blh_m
    events_df : DataFrame
        Columns: date, event_name, flag
    """
    if end_date is None:
        end_dt = pd.Timestamp.today().normalize()
    else:
        end_dt = pd.Timestamp(end_date).normalize()

    n_years: int = config["synthetic"]["n_years"]
    seed: int = config["synthetic"]["seed"]
    stations: list[dict] = config["stations"]

    # Build the hourly date range (n_years ending at end_date)
    start_dt = end_dt - pd.DateOffset(years=n_years) + pd.Timedelta(hours=1)
    timestamps = pd.date_range(start=start_dt, end=end_dt, freq="h")
    n_hours = len(timestamps)

    rng = np.random.default_rng(seed)

    all_aq: list[pd.DataFrame] = []
    all_weather: list[pd.DataFrame] = []

    for station in stations:
        sid = station["station_id"]

        # ------------------------------------------------------------------
        # Time coordinates reused across all series
        # ------------------------------------------------------------------
        hour_of_day = timestamps.hour.to_numpy(dtype=float)          # 0-23
        day_of_year = timestamps.day_of_year.to_numpy(dtype=float)   # 1-365/366
        month = timestamps.month.to_numpy(dtype=float)                # 1-12

        # Fractional year angle (peaks ~Jul 1 = mid-summer)
        # season_angle=0 at Jan 1, π at Jul 1
        season_angle = 2 * np.pi * (day_of_year - 1) / 365.25
        daily_angle = 2 * np.pi * hour_of_day / 24.0

        # Station-specific noise offset (keeps stations distinct)
        s_idx = stations.index(station)

        # ------------------------------------------------------------------
        # WEATHER generation
        # ------------------------------------------------------------------

        # Temperature: annual cycle hot summer (~40°C Jul) / cold winter (~8°C Jan)
        # + diurnal cycle (~10°C swing)
        temp_annual = 24.0 + 16.0 * np.sin(season_angle - np.pi / 2)
        temp_diurnal = 5.0 * np.sin(daily_angle - np.pi / 2)
        temp_noise = rng.normal(0, 1.5, n_hours)
        temp_c = temp_annual + temp_diurnal + temp_noise + s_idx * 0.5

        # Humidity: higher in monsoon (Jul–Sep month ≈ 7-9)
        # monsoon_weight peaks at month 8 (Aug)
        monsoon_w = np.clip(np.exp(-0.5 * ((month - 8) / 1.5) ** 2), 0, 1)
        humidity_base = 45.0 + 35.0 * monsoon_w
        # Also slightly higher at night
        humidity_diurnal = -10.0 * np.sin(daily_angle - np.pi / 4)
        humidity_noise = rng.normal(0, 5.0, n_hours)
        humidity_pct = np.clip(humidity_base + humidity_diurnal + humidity_noise, 10, 100)

        # Wind speed: calmer in winter (Nov-Jan), slightly higher in pre-monsoon
        wind_annual = 2.5 + 1.5 * np.sin(season_angle)
        wind_noise = rng.exponential(0.8, n_hours)
        wind_speed_ms = np.clip(wind_annual + wind_noise, 0.1, 25.0)

        # Wind direction: 0-360, uniform with seasonal bias
        wind_dir_deg = rng.uniform(0, 360, n_hours)

        # Pressure: ~1000-1016 hPa, slight seasonal and diurnal variation
        pressure_base = 1008.0 + 4.0 * np.cos(season_angle)
        pressure_diurnal = 2.0 * np.sin(2 * daily_angle)
        pressure_noise = rng.normal(0, 1.0, n_hours)
        pressure_hpa = pressure_base + pressure_diurnal + pressure_noise

        # Precipitation: monsoon-concentrated (Jul–Sep), mostly 0
        precip_prob = 0.005 + 0.18 * monsoon_w  # max ~18% chance per hour in peak monsoon
        rain_flag = rng.uniform(0, 1, n_hours) < precip_prob
        precip_intensity = rng.exponential(2.0, n_hours)
        precip_mm = np.where(rain_flag, precip_intensity, 0.0)

        # Cloud cover: correlated with humidity
        cloud_base = 10.0 + 50.0 * monsoon_w
        cloud_noise = rng.normal(0, 15.0, n_hours)
        cloud_pct = np.clip(cloud_base + cloud_noise, 0, 100)

        # Boundary-layer height: lower at night/winter, higher daytime/summer
        blh_annual = 1200.0 + 400.0 * np.sin(season_angle - np.pi / 2)
        blh_diurnal = 600.0 * np.maximum(np.sin(daily_angle - np.pi / 4), 0)
        blh_noise = rng.normal(0, 100, n_hours)
        blh_m = np.clip(blh_annual + blh_diurnal + blh_noise, 100, 3000)

        # ------------------------------------------------------------------
        # PM2.5 generation (driven by weather + seasonality)
        # ------------------------------------------------------------------
        # Base: winter seasonality (high Nov–Feb)
        # season_angle ~-π/2 at Jan (minimum of sin) → flip sign for winter peak
        winter_signal = -np.sin(season_angle - np.pi / 2)  # peaks Jan, trough Jul
        pm25_base = 80.0 + 80.0 * winter_signal  # 0–160 range from seasonality

        # Daily cycle: rush-hour peaks 08–10h and 19–22h
        rush_am = np.exp(-0.5 * ((hour_of_day - 9.0) / 1.5) ** 2)
        rush_pm = np.exp(-0.5 * ((hour_of_day - 20.5) / 1.5) ** 2)
        daily_pm = 40.0 * (rush_am + 0.8 * rush_pm)

        # Negative wind-speed dependence (dispersion)
        wind_suppression = -18.0 * np.log1p(wind_speed_ms)  # negative contribution

        # Rain washout (each mm of precip reduces PM2.5)
        rain_washout = -20.0 * np.log1p(precip_mm)

        # Negative temperature dependence
        temp_suppression = -0.8 * (temp_c - 20.0)

        # Low BLH → higher PM2.5 (trapping)
        blh_effect = -0.03 * (blh_m - 800.0)  # negative: higher blh → less PM

        # Per-station offset (Anand Vihar hotspot, R K Puram medium, Punjabi Bagh lower)
        station_offsets = [30.0, 10.0, 0.0]
        station_offset = station_offsets[s_idx]

        # Combine deterministic components
        pm25_det = (
            pm25_base
            + daily_pm
            + wind_suppression
            + rain_washout
            + temp_suppression
            + blh_effect
            + station_offset
        )
        pm25_det = np.clip(pm25_det, 5.0, 400.0)

        # AR(1) autocorrelation
        ar_noise = np.zeros(n_hours)
        eps = rng.normal(0, 15.0, n_hours)
        for i in range(1, n_hours):
            ar_noise[i] = 0.7 * ar_noise[i - 1] + eps[i]

        pm25_raw = pm25_det + ar_noise

        # Random spikes: Diwali (late Oct) + stubble burning (Oct-Nov)
        is_diwali_season = ((month == 10) | (month == 11)).astype(float)
        spike_prob = 0.002 + 0.015 * is_diwali_season
        spike_flag = rng.uniform(0, 1, n_hours) < spike_prob
        spike_magnitude = rng.uniform(100, 400, n_hours)
        pm25_raw = pm25_raw + np.where(spike_flag, spike_magnitude, 0.0)

        pm25 = np.clip(pm25_raw, 1.0, 999.0)

        # PM10 ≈ 1.6 × PM2.5 + noise, always ≥ PM2.5
        pm10_noise = rng.normal(0, 15.0, n_hours)
        pm10 = np.maximum(1.6 * pm25 + pm10_noise, pm25 + 1.0)

        # NO2: traffic-driven, correlated with daily PM cycle, higher in winter
        no2 = np.clip(
            30.0 + 0.4 * pm25 + 20.0 * rush_am + rng.normal(0, 10.0, n_hours), 1.0, 400.0
        )

        # SO2: industrial, smoother
        so2 = np.clip(
            10.0 + 0.1 * pm25 + rng.normal(0, 8.0, n_hours), 0.5, 300.0
        )

        # CO (mg/m³): correlated with traffic/combustion
        co = np.clip(
            0.5 + 0.015 * pm25 + rng.normal(0, 0.3, n_hours), 0.1, 25.0
        )

        # O3: photochemical, peaks in afternoon, suppressed by NO2 / in winter
        o3_diurnal = np.maximum(np.sin(daily_angle - np.pi / 3), 0)
        o3_summer = 0.5 + 0.5 * np.sin(season_angle)  # peaks summer
        o3 = np.clip(
            60.0 * o3_diurnal * o3_summer - 0.1 * no2 + rng.normal(0, 8.0, n_hours),
            1.0, 300.0,
        )

        # ------------------------------------------------------------------
        # Build DataFrames
        # ------------------------------------------------------------------
        weather_df_station = pd.DataFrame({
            "station_id": sid,
            "timestamp": timestamps,
            "temp_c": temp_c,
            "humidity_pct": humidity_pct,
            "wind_speed_ms": wind_speed_ms,
            "wind_dir_deg": wind_dir_deg,
            "pressure_hpa": pressure_hpa,
            "precip_mm": precip_mm,
            "cloud_pct": cloud_pct,
            "blh_m": blh_m,
        })

        aq_df_station = pd.DataFrame({
            "station_id": sid,
            "station_name": station["station_name"],
            "city": station["city"],
            "lat": station["lat"],
            "lon": station["lon"],
            "timestamp": timestamps,
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "so2": so2,
            "co": co,
            "o3": o3,
        })

        all_aq.append(aq_df_station)
        all_weather.append(weather_df_station)

    aq_df = pd.concat(all_aq, ignore_index=True)
    weather_df = pd.concat(all_weather, ignore_index=True)

    # ------------------------------------------------------------------
    # Events DataFrame: Diwali + crop-burning flags per year
    # ------------------------------------------------------------------
    events_df = _build_events(timestamps, rng)

    return aq_df, weather_df, events_df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _diwali_date(year: int) -> pd.Timestamp:
    """Approximate Diwali date for a given year (falls Oct–Nov).

    Uses a fixed approximation based on historical data for Delhi:
    2020-11-14, 2021-11-04, 2022-10-24, 2023-11-12, 2024-11-01, etc.
    We approximate as the 2nd Saturday of October/November using a simple
    offset from the new moon cycle (synodic month = 29.53 days).
    Reference epoch: 2020-11-14 (confirmed Diwali).
    """
    epoch = pd.Timestamp("2020-11-14")
    synodic = 29.530588  # days
    lunar_cycles = 12.3685  # tropical months per year
    delta_years = year - 2020
    # Approximate: add delta_years × 365.25 days, then subtract full lunar cycles
    # to stay within the Oct-Nov window
    approx_days = delta_years * 365.25
    approx_cycles = approx_days / synodic
    # Round to nearest integer cycle
    n_cycles = round(approx_cycles * lunar_cycles) / lunar_cycles
    candidate = epoch + pd.Timedelta(days=n_cycles * synodic)
    # Clamp to Oct 15 – Nov 20
    year_start = pd.Timestamp(f"{year}-10-15")
    year_end = pd.Timestamp(f"{year}-11-20")
    candidate = max(min(candidate, year_end), year_start)
    return candidate.normalize()


def _build_events(
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build events DataFrame covering the timestamp range."""
    start_year = timestamps.min().year
    end_year = timestamps.max().year

    rows: list[dict] = []

    for year in range(start_year, end_year + 1):
        # Diwali
        diwali_dt = _diwali_date(year)
        rows.append({
            "date": diwali_dt.date(),
            "event_name": "Diwali",
            "flag": 1,
        })
        # Crop burning season: Oct 1 – Nov 30
        for month in [10, 11]:
            import calendar
            last_day = calendar.monthrange(year, month)[1]
            for day in range(1, last_day + 1):
                rows.append({
                    "date": pd.Timestamp(year=year, month=month, day=day).date(),
                    "event_name": "crop_burning",
                    "flag": 1,
                })

    events_df = pd.DataFrame(rows, columns=["date", "event_name", "flag"])
    # De-duplicate (Diwali day also falls in crop-burning season range)
    events_df = events_df.drop_duplicates(subset=["date", "event_name"]).reset_index(drop=True)
    events_df["flag"] = events_df["flag"].astype(int)
    return events_df
