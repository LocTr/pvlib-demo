# PV power plant design with pvlib and NSGA-II

A small educational project demonstrating how to simulate a photovoltaic (PV)
installation and compare designs with conflicting objectives. Start with basic
pvlib features in an explained Jupyter notebook, then use NSGA-II to explore
energy, investment cost, and production variability.

## 1. Project objectives and scenario

Design a hypothetical fixed-tilt installation in **Ho Chi Minh City, Vietnam**
(10.8231° N, 106.6297° E; UTC+7). The site elevation is assumed to be 10 m.

| Decision variable | Range | Baseline |
|---|---|---|
| Panel tilt | 0–60° above horizontal | 10° |
| Panel azimuth | 90–270°, clockwise from north | 180° (south) |
| Installed DC capacity | 10–100 kWp | 50 kWp |

The objectives are to **maximize annual AC energy**, **minimize illustrative
investment cost**, and **minimize relative variability of daily energy**.
The final example selects a compromise using explicit preference weights.

The scope is deliberately small: one location, a generic array, a fixed DC/AC
ratio, and hourly weather. Basic Python knowledge is enough to follow the
notebook; no previous pvlib or optimization experience is assumed.

## 2. What is pvlib?

[pvlib python](https://pvlib-python.readthedocs.io/en/stable/) is an open-source
Python library for modeling solar energy systems. It provides weather readers,
solar-position calculations, irradiance models, module-temperature models,
electrical performance models, and inverter models. You can call individual
functions or connect them with its higher-level `ModelChain` interface.

Here pvlib estimates **electrical performance**. Investment cost and variability
are calculations we add, and optimization is performed by **pymoo**.

### Main classes and functions used here

| API | What it does | How this project uses it |
|---|---|---|
| `pvlib.iotools.get_pvgis_tmy()` | Downloads a Typical Meteorological Year (TMY) from PVGIS | Retrieves hourly solar irradiance, air temperature, and wind; saves a local cache |
| `pvlib.location.Location` | Stores coordinates, elevation, and timezone | Describes the Ho Chi Minh City site |
| `Location.get_solarposition()` | Calculates solar zenith, elevation, and azimuth | Demonstrates the sun's position; ModelChain also performs this step internally |
| `pvlib.pvsystem.PVSystem` | Describes an array and its inverter/model parameters | Sets tilt, orientation, DC nameplate power, temperature coefficients, losses, and inverter size |
| `pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS` | Provides standard coefficients for supported thermal models | Selects SAPM open-rack glass/glass coefficients |
| `pvlib.modelchain.ModelChain.with_pvwatts()` | Configures a modeling chain with PVWatts electrical models | Connects the site, system, and selected physical models |
| `ModelChain.run_model(weather)` | Runs the chain from broadband irradiance | Produces the power time series for each orientation |
| `ModelChain.results.total_irrad` | Stores irradiance on the panels | Explains the difference between horizontal weather irradiance and panel irradiance |
| `ModelChain.results.cell_temperature` | Stores modeled cell temperature | Shows the effect of weather on module operating temperature |
| `ModelChain.results.dc` / `.ac` | Stores DC / AC power | Supplies generation for the energy and variability objectives |
| `PVSystem.pvwatts_losses()` | Combines generic loss percentages multiplicatively | Shows the approximately 14.08% generic system-loss allowance |

Other useful APIs for future exploration are `pvsystem.Array` and `FixedMount`
for multiple array groups, `SingleAxisTrackerMount` for tracking, and
`irradiance.get_total_irradiance()` for a standalone transposition calculation.
They are not required by this single-array example.

### How the simulation works

```text
Location + timestamps → solar position
Weather + tilt/azimuth → irradiance on the panel plane
Irradiance + air temperature + wind → cell temperature
Effective irradiance + cell temperature + DC capacity → DC power
DC power + inverter parameters → AC power
Hourly AC power → annual energy and daily energy variability
Design variables → illustrative investment cost
Three objective values → NSGA-II → approximate Pareto set → preference ranking
```

`ModelChain` performs the physical steps. This project selects Hay–Davies
irradiance transposition, physical incidence-angle losses, SAPM cell
temperature, PVWatts DC and inverter models, and no spectral correction.
The generic loss percentages in `build_model()` combine to approximately
14.08%; temperature, incidence-angle, and inverter effects are modeled separately.
The shading loss is a fixed allowance, not a geometric shading model.

The array has a generic temperature coefficient of −0.004/°C and ground albedo
of 0.2. There is no need to download a commercial module database.

**Two meanings of `pdc0`:** the module's `pdc0` is array DC nameplate power in W.
The inverter's `pdc0` is its DC input reference; its AC rating is
`eta_inv_nom * pdc0`. We use 96% nominal inverter efficiency and DC/AC ratio 1.2.
For the 50 kWp baseline, the inverter AC rating is 41.67 kW and its `pdc0` is
43.40 kW. See the [PVWatts inverter API](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.pvwatts.html).

## 3. Tools

| Tool | Role |
|---|---|
| Python 3.12 | Tested interpreter; use a separate virtual environment |
| JupyterLab and ipykernel | Interactive notebook execution |
| pvlib | PV performance simulation |
| pandas | Timezone-aware weather tables, daily/monthly aggregation, CSV export |
| NumPy | Objective arrays, numerical calculations, checks |
| Matplotlib | Weather, power, monthly energy, and Pareto plots |
| pymoo | NSGA-II and nondominated sorting |
| nbconvert / nbformat | Notebook execution, validation, and optional HTML export |
| unittest | Focused numerical and workflow checks; included in Python |

`requirements.txt` lists the project dependencies. `requirements-lock.txt`
records the exact environment used for validation; use it when reproducing
the supplied results on Python 3.12 / Windows.

## 4. Install

Install Python 3.12, open a terminal in this directory, and create a virtual
environment. No PVGIS API key is needed. Package installation needs internet;
the included weather cache supports offline execution afterward.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name pvlib-demo --display-name "Python (pvlib demo)"
.\.venv\Scripts\python.exe -m jupyterlab
```

If `python` is unavailable but the Python launcher is installed, use
`py -3.12 -m venv .venv` for the first command. No PowerShell activation or
execution-policy change is necessary with the explicit executable paths.

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name pvlib-demo --display-name "Python (pvlib demo)"
.venv/bin/python -m jupyterlab
```

In JupyterLab, open **pv_multiobjective_demo.ipynb** and select the
**Python (pvlib demo)** kernel. Starting JupyterLab from this project directory
lets the notebook import `pv_demo.py` and locate its data.

## 5. Run the demonstration

### Explained notebook

Use **Kernel → Restart Kernel and Run All Cells**. The notebook walks through:

1. pvlib overview, its main APIs, and the configuration.
2. Weather loading, source metadata, units, and data checks.
3. Solar-position calculations.
4. Explicit `PVSystem` and `ModelChain` construction and execution.
5. Intermediate irradiance, temperature, DC power, and AC power.
6. Energy, cost, and variability calculations.
7. Hand-picked examples changing orientation, tilt, and size.
8. NSGA-II and the approximate Pareto set.
9. 2D and 3D plots, representative designs, and preference weights.
10. Sanity checks, exports, and suggested experiments.

### Command-line script

The script uses the same physical model and objective functions as the notebook:

```powershell
.\.venv\Scripts\python.exe pv_demo.py
```

Optional examples:

```powershell
# Short run for exploring the workflow:
.\.venv\Scripts\python.exe pv_demo.py --population 12 --generations 3 --output-dir outputs/quick

# Larger search; compare results before making convergence claims:
.\.venv\Scripts\python.exe pv_demo.py --population 80 --generations 80 --seed 42 --output-dir outputs/larger

# Replace the included weather cache with a fresh PVGIS response:
.\.venv\Scripts\python.exe pv_demo.py --refresh-weather

# Display figures in windows as well as saving them:
.\.venv\Scripts\python.exe pv_demo.py --show
```

Use `.venv/bin/python` in place of the Windows executable on macOS/Linux.
The default run exports figures without opening windows.

### Execute or export the notebook from a terminal

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute pv_multiobjective_demo.ipynb --inplace --ExecutePreprocessor.kernel_name=pvlib-demo --ExecutePreprocessor.timeout=600
.\.venv\Scripts\python.exe -m nbconvert --to html pv_multiobjective_demo.ipynb --output-dir outputs
```

The first command stores computed tables and figures in the notebook. The
second creates a shareable HTML version without rerunning it.

## 6. Weather data and units

Source: **European Commission Joint Research Centre PVGIS 5.3 TMY**. Data are
obtained using `get_pvgis_tmy(map_variables=True, roll_utc_offset=7,
coerce_year=2023)`. PVGIS provides historical-derived radiation/weather;
these should not be described as measurements from this hypothetical plant.

A **Typical Meteorological Year** assembles representative months selected
from historical years. The 2023 timestamps are non-leap calendar labels, not
observed weather for the actual year 2023. Source years and the radiation
database are preserved inside `data/weather_metadata.json`.
The included cache uses **PVGIS-ERA5**, with source months selected from
**2005–2023**.
TMY is useful for this small demonstration; continuous, multiyear historical
data are more appropriate for studying ramps and interannual production risk.

**Time alignment matters:** PVGIS labels TMY rows at HH:00, while ERA5
irradiance represents the hour midpoint, HH:30. The metadata's
`irradiance_time_offset` is 0.5 hours. `weather_for_model()` applies this offset
before the standalone solar-position calculation and every ModelChain run.
The cache retains the original labels; simulated power is indexed at HH:30,
with one value representing each hour and 24 values per local day. See the
[JRC TMY timing documentation](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/pvgis-typical-meteorological-year-tmy-generator_en).

| Column | Meaning | Unit |
|---|---|---|
| `ghi` | Global horizontal irradiance | W/m² |
| `dni` | Direct normal irradiance | W/m² |
| `dhi` | Diffuse horizontal irradiance | W/m² |
| `temp_air` | Air temperature | °C |
| `wind_speed` | Wind speed | m/s |

The cache contains 8,760 consecutive hourly rows in local time, covering 365
complete days. Loading checks location metadata, a SHA-256 checksum, timestamps,
required columns, missing values, and nonnegative irradiance/wind speed.
An unavailable API produces an explicit error; synthetic weather is not used
as a fallback. Refreshing changes the cached source response and can affect
the numerical results.

## 7. Objective formulas

Let `S` be system size in kWp, `beta` be tilt in degrees, and `P_t` be AC power
in kW for hourly row `t`.

**Annual energy**, maximized:

```text
E_annual = sum(P_t × 1 hour)                  [kWh per representative year]
```

pvlib returns watts; divide by 1,000 before working in kW. Summing power is an
energy calculation here only because every timestep represents exactly one
hour. Different or irregular timesteps require explicit duration weights.

**Investment cost**, minimized:

```text
C = 5,000 + S × (800 + 2 × beta)              [illustrative USD]
```

The fixed term is USD, equipment allowance is USD/kWp, and the tilt allowance
is USD/(kWp·degree). These are editable teaching assumptions, not market prices.
The formula does not model financing, land, operation, taxes, or revenue.

**Daily energy variability**, minimized:

```text
E_day = sum(hourly energy within each local day)       [kWh/day]
daily_CV = std(E_day, ddof=0) / mean(E_day)             [dimensionless]
```

We use population standard deviation over all 365 days. A CV of 0.20 means
standard deviation equals 20% of mean daily energy. Normalizing avoids making
a small system appear intrinsically more stable just because it generates less.
With fixed orientation and DC/AC ratio, CV is independent of size. This metric
includes seasonal variation; it does not describe hourly ramps or the order
of high/low production days.

## 8. Optimization and decision making

`PVDesignProblem` subclasses pymoo's `ElementwiseProblem`. Each evaluation
simulates a design, calculates the metrics, and returns:

```python
[-annual_energy_kwh, investment_usd, daily_cv]
```

The negative sign turns energy maximization into minimization. Default NSGA-II
settings are population 40, 30 generations, and random seed 42. The initial
population contains the baseline and two designs at the boundary sizes.
The other initial candidates are uniformly sampled within the design bounds.

NSGA-II favors nondominated designs and maintains diversity using crowding
distance, then produces new designs using crossover and mutation. The script
keeps the search history and extracts the nondominated designs from **all
evaluations in that run**. This archive can contain more than 40 points.
It is an **approximate Pareto set**, not proof of the global optimum.

The simulator caches exact tilt/azimuth pairs. It runs each orientation at
1 kWp and scales power with size because the chosen electrical models are
homogeneous when inverter rating also scales. The checks compare this shortcut
against a full-size ModelChain run. Revisit this shortcut for a fixed inverter,
discrete equipment, changing losses, or layout-dependent shading.

For preference ranking, convert energy to minimization, normalize each
objective to 0–1 across the Pareto set, and calculate:

```text
score_i = sqrt(sum(weight_j × normalized_objective_ij²))
```

The lowest score is the example compromise. Weights follow **energy, cost,
variability** order and sum to one after normalization. The notebook compares
equal weights against energy-, cost-, and stability-priority weights.
The ideal point may be unattainable; rankings depend on weights and the
discovered set's ranges. A weighted preference is not a universally best design.

## 9. Files and outputs

```text
pvlib_research/
├── README.md                          # This guide
├── requirements.txt                   # Direct dependencies
├── requirements-lock.txt              # Exact validated environment
├── pv_multiobjective_demo.ipynb        # Explained, executable notebook
├── pv_demo.py                         # Shared model, objectives, optimization, CLI
├── tests/test_demo.py                 # Numerical and workflow checks
├── data/
│   ├── weather.csv                    # Cached PVGIS representative year
│   └── weather_metadata.json          # Provenance, source months, checksum
└── outputs/
    ├── pareto_solutions.csv           # Pareto designs with compromise score
    ├── evaluated_designs.csv          # All candidates from the optimization run
    ├── representative_designs.csv     # Extremes and equal-weight compromise
    ├── run_summary.json               # Baseline, settings, assumptions, versions
    └── figures/
        ├── baseline.png
        ├── pareto.png
        ├── pareto_3d.png
        └── selected_designs.png       # Additional notebook comparison
```

CSV tables retain full precision. `pareto.png` shows energy versus cost colored
by daily CV and energy versus CV colored by capacity. `pareto_3d.png` displays
all three objectives. Red stars mark the baseline in the 2D plots.

### Example results from the included execution

With the bundled weather, population 40, 30 generations, and seed 42, the
search evaluates **1,200 designs** and retains **455 nondominated designs**.

| Selection | Size (kWp) | Annual energy (MWh) | Illustrative cost (USD) | Daily CV |
|---|---:|---:|---:|---:|
| Baseline | 50.00 | 72.30 | 46,000 | 0.2047 |
| Lowest cost found | 10.00 | 14.22 | 13,000 | 0.1980 |
| Highest energy found | 99.99 | 148.61 | 88,718 | 0.2060 |
| Lowest variability found | 44.14 | 58.44 | 41,569 | 0.1968 |
| Equal-weight compromise | 56.21 | 80.44 | 50,125 | 0.1981 |

The compromise produces **11.3% more energy** than the baseline at **9.0% higher
illustrative cost**, with lower daily CV. Its added capacity explains much of
the energy increase: its energy per installed kWp is slightly lower than the
baseline's. The maximum-energy design costs much more and has higher daily CV
than the compromise. The minimum-CV design gives up total energy. These are
trade-offs among the evaluated alternatives under the stated assumptions.
Different weather, preferences, seeds, or search budgets can change the selection.

## 10. Check the implementation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

The checks cover constant-power integration, variability scale invariance,
full-size versus scaled pvlib power, inverter rating, invalid weather rejection,
independent dominance checks, optimizer bounds, preference-score unit
invariance, seeded repeatability, irradiance timing, and offline cache reading.
The notebook also contains visible sanity checks and can be executed with
nbconvert as shown above.

### Troubleshooting

- **Missing package / wrong kernel:** install requirements with the same Python
  executable used to launch JupyterLab; select `Python (pvlib demo)`.
- **Cannot import `pv_demo`:** launch JupyterLab from this project directory.
- **PVGIS network or service failure:** retain the included CSV and metadata to
  run offline, or retry `--refresh-weather` when the service is available.
- **Cache checksum / site mismatch:** restore the original paired cache files
  or refresh intentionally. Editing the site requires a new weather download.
  For another timezone, also update its integer UTC offset and validation rules;
  this small example assumes no daylight saving time.
- **Notebook takes longer than expected:** first try population 12 and 3
  generations. Increase the search budget when analyzing convergence.
- **Unexpectedly simple front:** size strongly couples cost and energy; relative
  daily CV does not change with size. Inspect the actual results before making
  claims about conflicts among the objectives.

## 11. Limitations and extension ideas

This demonstration uses generic equipment, continuous system size, simple costs,
one site, fixed inverter sizing rules, and a modest evolutionary search.
It does not model row spacing, explicit nearby obstacles, land constraints,
grid limits, battery operation, detailed economics, or long-term degradation.
TMY cannot quantify interannual uncertainty or establish plant investment risk.

Useful next steps are continuous multiyear weather, hourly ramp-rate metrics,
real component specifications and cost data, layout constraints, energy
requirements, comparison with a coarse design grid, and convergence comparisons
across seeds and search budgets.

## 12. References and attribution

- Anderson, K., Hansen, C., Holmgren, W., Jensen, A., Mikofski, M., and Driesse,
  A. (2023). [pvlib python: 2023 project update](https://joss.theoj.org/papers/10.21105/joss.05994).
  *Journal of Open Source Software*, 8(92), 5994. DOI: 10.21105/joss.05994.
- [pvlib ModelChain](https://pvlib-python.readthedocs.io/en/stable/reference/modelchain.html).
- [pvlib PVGIS TMY API](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.iotools.get_pvgis_tmy.html).
- [pvlib PVSystem API](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.PVSystem.html).
- [pymoo NSGA-II documentation](https://www.pymoo.org/algorithms/moo/nsga2.html).
- Weather data: European Commission, Joint Research Centre,
  [Photovoltaic Geographical Information System (PVGIS)](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en).
  The exact database, source months, and retrieval metadata accompany the cache.
