"""Small, explained PV design study shared by the notebook and command line.

Run ``python pv_demo.py --help`` for options. Costs are illustrative USD.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


ROOT = Path(__file__).resolve().parent
WEATHER_COLUMNS = ["ghi", "dni", "dhi", "temp_air", "wind_speed"]
DESIGN_COLUMNS = ["tilt_deg", "azimuth_deg", "size_kwp"]
METRIC_COLUMNS = ["annual_energy_kwh", "investment_usd", "daily_cv"]
LOWER_BOUNDS = np.array([0.0, 90.0, 10.0])
UPPER_BOUNDS = np.array([60.0, 270.0, 100.0])
DC_AC_RATIO = 1.2
INVERTER_EFFICIENCY = 0.96
PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_3/"


@dataclass(frozen=True)
class Site:
    name: str = "Ho Chi Minh City"
    latitude: float = 10.8231
    longitude: float = 106.6297
    timezone: str = "Asia/Ho_Chi_Minh"
    utc_offset: int = 7
    altitude_m: float = 10.0  # Approximate, fixed teaching assumption.

    def location(self):
        return pvlib.location.Location(
            self.latitude, self.longitude, tz=self.timezone,
            altitude=self.altitude_m, name=self.name,
        )


@dataclass(frozen=True)
class Design:
    tilt_deg: float = 10.0
    azimuth_deg: float = 180.0
    size_kwp: float = 50.0

    def __post_init__(self):
        values = np.array([self.tilt_deg, self.azimuth_deg, self.size_kwp])
        if not np.isfinite(values).all():
            raise ValueError("Design variables must be finite.")
        if not (0 <= self.tilt_deg <= 90 and 0 <= self.azimuth_deg <= 360
                and self.size_kwp > 0):
            raise ValueError("Use tilt 0–90°, azimuth 0–360°, and positive kWp.")


def validate_weather(weather: pd.DataFrame, site: Site = Site()) -> pd.DataFrame:
    """Require a complete, finite, hourly non-leap TMY in local time.

    Each row represents one hour. The 2023 calendar is a convenient label;
    source months come from different historical years, recorded in metadata.
    """
    missing = set(WEATHER_COLUMNS) - set(weather.columns)
    if missing:
        raise ValueError(f"Missing weather columns: {sorted(missing)}")
    if not isinstance(weather.index, pd.DatetimeIndex) or weather.index.tz is None:
        raise ValueError("Weather needs a timezone-aware DatetimeIndex.")
    weather = weather.loc[:, WEATHER_COLUMNS].copy()
    weather.index = weather.index.tz_convert(site.timezone)
    expected = pd.date_range("2023-01-01", periods=8760, freq="h", tz=site.timezone)
    if not weather.index.equals(expected):
        raise ValueError("Expected all 8,760 hourly rows of the local 2023 TMY calendar.")
    if not np.isfinite(weather.to_numpy(dtype=float)).all():
        raise ValueError("Weather contains missing or non-finite values.")
    if (weather[["ghi", "dni", "dhi", "wind_speed"]] < 0).any().any():
        raise ValueError("Irradiance and wind speed must be nonnegative.")
    weather.index.name = "time"
    return weather


def load_weather(data_dir: Path = ROOT / "data", *, refresh: bool = False,
                 site: Site = Site()):
    """Load the checked local cache, or download PVGIS TMY once.

    No synthetic or clear-sky substitution is made if the API is unavailable.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "weather.csv"
    meta_path = data_dir / "weather_metadata.json"
    if not refresh and csv_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("site") != asdict(site):
            raise ValueError("Cache belongs to a different site. Use refresh=True.")
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if metadata.get("csv_sha256") != digest:
            raise ValueError("Weather cache checksum failed. Use refresh=True to replace it.")
        weather = pd.read_csv(csv_path, index_col="time")
        weather.index = pd.to_datetime(weather.index, utc=True)
        weather = validate_weather(weather, site)
        weather.attrs["irradiance_time_offset_hours"] = float(
            metadata["pvgis_metadata"]["inputs"]["location"]["irradiance_time_offset"]
        )
        return weather, metadata

    try:
        weather, source_metadata = pvlib.iotools.get_pvgis_tmy(
            site.latitude, site.longitude, map_variables=True,
            roll_utc_offset=site.utc_offset, coerce_year=2023,
            usehorizon=True, url=PVGIS_URL, timeout=60,
        )
    except Exception as exc:
        raise RuntimeError(
            "PVGIS download failed. Check internet access and the PVGIS service, "
            "or restore data/weather.csv and weather_metadata.json from this project. "
            "No synthetic weather has been substituted."
        ) from exc
    weather = validate_weather(weather, site)
    weather.attrs["irradiance_time_offset_hours"] = float(
        source_metadata["inputs"]["location"]["irradiance_time_offset"]
    )
    metadata = {
        "source": "European Commission JRC PVGIS 5.3 TMY",
        "source_url": PVGIS_URL + "tmy",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "site": asdict(site),
        "calendar_year_label": 2023,
        "description": "Representative TMY assembled from historical months; not observed 2023 weather.",
        "request": {"roll_utc_offset": site.utc_offset, "coerce_year": 2023,
                    "map_variables": True, "usehorizon": True},
        "pvlib_version": pvlib.__version__,
        "pvgis_metadata": source_metadata,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    weather.to_csv(csv_path)
    metadata["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return weather, metadata


def weather_for_model(weather: pd.DataFrame) -> pd.DataFrame:
    """Apply the PVGIS irradiance offset before calculating solar position.

    ERA5 values represent the hour midpoint (HH:30), although TMY CSV/JSON
    labels are HH:00. The unchanged hourly values are modeled at HH:30.
    See the JRC TMY generator documentation linked from the README.
    """
    offset = weather.attrs.get("irradiance_time_offset_hours")
    if offset is None or not np.isfinite(offset) or not 0 <= offset < 1:
        raise ValueError("Need the PVGIS irradiance time offset, in [0, 1) hours.")
    aligned = weather.copy()
    aligned.index = weather.index + pd.Timedelta(hours=offset)
    return aligned


def build_model(design: Design, site: Site = Site()):
    """Construct a generic aggregate array and correctly sized inverter.

    Module pdc0 is array DC nameplate power, in W. Inverter pdc0 is a DC
    input reference, NOT its AC rating: P_ac,rated = eta_nom * pdc0.
    """
    dc_w = design.size_kwp * 1000
    ac_rating_w = dc_w / DC_AC_RATIO
    system = pvlib.pvsystem.PVSystem(
        surface_tilt=design.tilt_deg,
        surface_azimuth=design.azimuth_deg,
        module_parameters={"pdc0": dc_w, "gamma_pdc": -0.004},
        inverter_parameters={"pdc0": ac_rating_w / INVERTER_EFFICIENCY,
                             "eta_inv_nom": INVERTER_EFFICIENCY},
        temperature_model_parameters=(
            pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
        ),
        # Percent losses combine multiplicatively, totaling approximately 14.08%.
        losses_parameters={"soiling": 2, "shading": 3, "snow": 0,
                           "mismatch": 2, "wiring": 2, "connections": 0.5,
                           "lid": 1.5, "nameplate_rating": 1, "age": 0,
                           "availability": 3},
        albedo=0.2,
    )
    return pvlib.modelchain.ModelChain.with_pvwatts(
        system, site.location(), transposition_model="haydavies",
        temperature_model="sapm", aoi_model="physical", spectral_model="no_loss",
    )


def investment_cost(design: Design) -> float:
    """Illustrative USD: fixed cost + kWp × (equipment + tilt allowance)."""
    return 5000.0 + design.size_kwp * (800.0 + 2.0 * design.tilt_deg)


def summarize_power(ac_kw: pd.Series) -> dict:
    """Integrate regular hourly power; population CV of 365 daily energies."""
    values = ac_kw.to_numpy(dtype=float)
    if (len(ac_kw) != 8760 or not np.isfinite(values).all()
            or (values < 0).any()):
        raise ValueError("Expected 8,760 finite, nonnegative AC power samples.")
    hourly_steps = ac_kw.index.to_series().diff().iloc[1:]
    if (ac_kw.index.tz is None or ac_kw.index[0].hour != 0
            or not hourly_steps.eq(pd.Timedelta(hours=1)).all()):
        raise ValueError("Expected consecutive hourly samples covering complete local days.")
    hourly_energy_kwh = ac_kw * 1.0  # kW × 1 hour = kWh
    daily = hourly_energy_kwh.resample("D").sum()
    mean_daily = float(daily.mean())
    if len(daily) != 365 or mean_daily <= 0:
        raise ValueError("Need 365 complete days with positive mean energy.")
    return {"annual_energy_kwh": float(hourly_energy_kwh.sum()),
            "daily_cv": float(daily.std(ddof=0) / mean_daily)}


class PVSimulator:
    """Evaluate designs with pvlib, caching repeated tilt/azimuth pairs.

    At fixed DC/AC ratio these models scale linearly with array size. Running
    a 1 kWp reference therefore preserves the chosen model while saving work.
    This shortcut would need revisiting for fixed inverter sizes or row shading.
    """

    def __init__(self, weather: pd.DataFrame, site: Site = Site()):
        self.site = site
        self.weather = validate_weather(weather, site)
        self.model_weather = weather_for_model(self.weather)
        self._power_cache = {}
        self.evaluations = []

    def power(self, design: Design) -> pd.Series:
        key = (float(design.tilt_deg), float(design.azimuth_deg))
        if key not in self._power_cache:
            reference = Design(design.tilt_deg, design.azimuth_deg, 1.0)
            model = build_model(reference, self.site)
            model.run_model(self.model_weather)
            power = model.results.ac / 1000  # pvlib gives W; analysis uses kW.
            if not np.isfinite(power.to_numpy()).all():
                raise ValueError("pvlib returned non-finite AC power.")
            self._power_cache[key] = power.clip(lower=0).rename("ac_kw")
        return self._power_cache[key] * design.size_kwp

    def evaluate(self, design: Design) -> dict:
        row = asdict(design) | summarize_power(self.power(design))
        row["investment_usd"] = investment_cost(design)
        self.evaluations.append(row)
        return row


def objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    """All minimization: negative energy, cost, dimensionless variability."""
    return frame[METRIC_COLUMNS].to_numpy(dtype=float) * np.array([-1, 1, 1])


class PVDesignProblem(ElementwiseProblem):
    def __init__(self, simulator: PVSimulator):
        super().__init__(n_var=3, n_obj=3, xl=LOWER_BOUNDS, xu=UPPER_BOUNDS)
        self.simulator = simulator

    def _evaluate(self, x, out, *args, **kwargs):
        row = self.simulator.evaluate(Design(*map(float, x)))
        out["F"] = [-row["annual_energy_kwh"], row["investment_usd"], row["daily_cv"]]


def nondominated(frame: pd.DataFrame) -> pd.DataFrame:
    """Approximate Pareto set among supplied candidates, including anchors."""
    frame = frame.drop_duplicates(subset=DESIGN_COLUMNS).reset_index(drop=True)
    indices = NonDominatedSorting().do(objective_matrix(frame), only_non_dominated_front=True)
    return frame.iloc[indices].sort_values("investment_usd").reset_index(drop=True)


def optimize_designs(simulator: PVSimulator, *, population: int = 40,
                     generations: int = 30, seed: int = 42, verbose: bool = False):
    if population < 4 or generations < 1:
        raise ValueError("Use population >= 4 and generations >= 1.")
    start = len(simulator.evaluations)
    # Include the stated baseline and boundary sizes at the outset. This makes
    # the teaching comparison useful even with a small evolutionary budget.
    rng = np.random.default_rng(seed)
    sampling = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(population, 3))
    sampling[:3] = [[10, 180, 50], [0, 180, 10], [10, 180, 100]]
    result = minimize(
        PVDesignProblem(simulator),
        NSGA2(pop_size=population, sampling=sampling, eliminate_duplicates=True),
        termination=("n_gen", generations), seed=seed, verbose=verbose,
    )
    evaluated = pd.DataFrame(simulator.evaluations[start:])
    # Use the nondominated archive of ALL evaluated points. A small final
    # population alone may discard useful extremes during crowding selection.
    pareto = nondominated(evaluated)
    return pareto, evaluated, result


def rank_compromises(pareto: pd.DataFrame, weights=(1 / 3, 1 / 3, 1 / 3)):
    """Weighted distance to the ideal point after per-objective min-max scaling.

    Weights are in energy, cost, variability order; lower score is preferred.
    The ideal can combine unattainable extremes from different designs.
    """
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (3,) or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError("Provide three finite nonnegative weights with positive sum.")
    if pareto.empty:
        raise ValueError("Cannot rank an empty Pareto set.")
    weights = weights / weights.sum()
    objectives = objective_matrix(pareto)
    span = np.ptp(objectives, axis=0)
    scaled = (objectives - objectives.min(axis=0)) / np.where(span > 0, span, 1)
    ranked = pareto.copy()
    ranked["compromise_score"] = np.sqrt((scaled**2 * weights).sum(axis=1))
    return ranked.sort_values("compromise_score", kind="stable").reset_index(drop=True)


def representative_designs(pareto: pd.DataFrame):
    ranked = rank_compromises(pareto)
    selections = {
        "Lowest cost": pareto.loc[pareto.investment_usd.idxmin()],
        "Highest energy": pareto.loc[pareto.annual_energy_kwh.idxmax()],
        "Lowest variability": pareto.loc[pareto.daily_cv.idxmin()],
        "Equal-weight compromise": ranked.iloc[0],
    }
    return pd.DataFrame(selections).T.loc[:, DESIGN_COLUMNS + METRIC_COLUMNS]


def plot_baseline(weather, ac_kw, output_dir=None):
    """Weather and power for one week, and monthly representative-year energy."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), layout="constrained")
    week = weather.index[:7 * 24]
    weather.loc[week, ["ghi", "dni", "dhi"]].plot(ax=axes[0], alpha=0.8)
    axes[0].set(title="First TMY week: irradiance", ylabel="Irradiance (W/m²)", xlabel="")
    ac_kw.iloc[:7 * 24].plot(ax=axes[1], color="tab:orange")
    axes[1].set(title="Baseline AC generation", ylabel="AC power (kW)", xlabel="Local time (UTC+7)")
    monthly = ac_kw.resample("MS").sum() / 1000  # hourly kW → MWh
    axes[2].bar(monthly.index.strftime("%b"), monthly.to_numpy(), color="tab:blue")
    axes[2].set(title="Baseline monthly energy — representative TMY", ylabel="Energy (MWh)")
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / "baseline.png", dpi=150)
    return fig


def plot_pareto(pareto, evaluated, baseline, output_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    pairs = [("annual_energy_kwh", "investment_usd", "daily_cv",
              "Annual energy (MWh)", "Illustrative investment (thousand USD)", "Daily energy CV"),
             ("annual_energy_kwh", "daily_cv", "size_kwp",
              "Annual energy (MWh)", "Daily energy CV", "System size (kWp)")]
    for ax, (x, y, color, xlabel, ylabel, clabel) in zip(axes, pairs):
        y_scale = 1000 if y == "investment_usd" else 1
        ax.scatter(evaluated[x] / 1000, evaluated[y] / y_scale,
                   s=8, c="lightgray", alpha=0.5, label="Evaluated designs")
        points = ax.scatter(pareto[x] / 1000, pareto[y] / y_scale,
                            c=pareto[color], cmap="viridis", s=24, label="Approximate Pareto set")
        ax.scatter(baseline[x] / 1000, baseline[y] / y_scale,
                   marker="*", s=160, c="red", edgecolors="black", label="Baseline")
        best = rank_compromises(pareto).iloc[0]
        ax.scatter(best[x] / 1000, best[y] / y_scale,
                   marker="D", s=80, facecolors="none", edgecolors="black", label="Compromise")
        fig.colorbar(points, ax=ax, label=clabel)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=7)
    fig.suptitle("NSGA-II: trade-offs in the representative weather year")
    fig3d = plt.figure(figsize=(9, 6), layout="constrained")
    ax3d = fig3d.add_subplot(111, projection="3d")
    points = ax3d.scatter(pareto.annual_energy_kwh / 1000,
                          pareto.investment_usd / 1000, pareto.daily_cv,
                          c=pareto.size_kwp, cmap="viridis", s=24)
    ax3d.set(xlabel="Annual energy (MWh)", ylabel="Investment (thousand USD)",
             zlabel="Daily energy CV", title="Approximate three-objective Pareto front")
    fig3d.colorbar(points, ax=ax3d, label="System size (kWp)", shrink=0.65)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / "pareto.png", dpi=150)
        fig3d.savefig(Path(output_dir) / "pareto_3d.png", dpi=150)
    return fig, fig3d


def export_results(pareto, evaluated, baseline, metadata, output_dir,
                   *, population, generations, seed):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_compromises(pareto).to_csv(output_dir / "pareto_solutions.csv", index=False)
    evaluated.to_csv(output_dir / "evaluated_designs.csv", index=False)
    representatives = representative_designs(pareto)
    representatives.to_csv(output_dir / "representative_designs.csv", index_label="selection")
    summary = {
        "baseline": baseline,
        "optimizer": {"algorithm": "NSGA-II", "population": population,
                      "generations": generations, "seed": seed},
        "evaluations": len(evaluated), "pareto_count": len(pareto),
        "weather_sha256": metadata["csv_sha256"],
        "weather_source": metadata["source"],
        "irradiance_time_offset_hours": metadata["pvgis_metadata"]["inputs"]["location"]["irradiance_time_offset"],
        "assumptions": {"dc_ac_ratio": DC_AC_RATIO, "nominal_inverter_efficiency": INVERTER_EFFICIENCY,
                        "cost": "5000 + size_kwp * (800 + 2 * tilt_deg); illustrative USD",
                        "variability": "population standard deviation / mean of 365 daily energies"},
        "versions": {name: version(name) for name in
                     ["pvlib", "pymoo", "numpy", "pandas", "matplotlib"]},
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return representatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-weather", action="store_true", help="Replace cached PVGIS data.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--show", action="store_true", help="Open plot windows after exporting.")
    args = parser.parse_args()
    if not args.show:
        plt.switch_backend("Agg")
    weather, metadata = load_weather(refresh=args.refresh_weather)
    print(f"Weather: {len(weather):,} hourly TMY rows; {metadata['source']}")
    simulator = PVSimulator(weather)
    baseline = simulator.evaluate(Design())
    print("Baseline:", json.dumps(baseline, indent=2))
    pareto, evaluated, _ = optimize_designs(
        simulator, population=args.population, generations=args.generations,
        seed=args.seed, verbose=True,
    )
    representatives = export_results(
        pareto, evaluated, baseline, metadata, args.output_dir,
        population=args.population, generations=args.generations, seed=args.seed,
    )
    plot_baseline(weather, simulator.power(Design()), args.output_dir / "figures")
    plot_pareto(pareto, evaluated, baseline, args.output_dir / "figures")
    print("\nRepresentative designs (illustrative USD; CV is dimensionless):")
    print(representatives.round(3).to_string())
    print(f"\nSaved {len(pareto)} nondominated designs to {args.output_dir.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
