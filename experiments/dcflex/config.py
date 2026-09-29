"""All model parameters in one place. Values tagged ASSUMPTION are engineering assumptions; the others
carry their source here and in the paper, Appendix A (Table: tab:params)."""
from dataclasses import dataclass, field
import numpy as np

AED_PER_USD = 3.6725  # UAE dirham peg


@dataclass
class Site:
    name: str = "Abu Dhabi"
    lat: float = 24.45
    lon: float = 54.38
    tz: str = "Asia/Dubai"          # no daylight saving time


@dataclass
class Facility:
    it_mw: float = 100.0            # case design
    util_mean: float = 0.75         # ASSUMPTION
    idle_frac: float = 0.25         # ASSUMPTION
    # Shares of dynamic IT energy. Deferrable work (training + batch) = 0.30 x (0.75 - 0.25) x 100 MW
    # = 15 MW = 20% of mean IT load (base case; sensitivity 0-40%). ASSUMPTION
    shares: dict = field(default_factory=lambda: {"inf": 0.70, "trn": 0.22, "bat": 0.08})
    deadline_h: dict = field(default_factory=lambda: {"trn": 12, "bat": 24})               # ASSUMPTION
    loss_frac: float = 0.06         # ASSUMPTION
    fan_frac: float = 0.05          # ASSUMPTION
    aux_mw: float = 1.0             # ASSUMPTION
    # Chiller COP = cop_ref - cop_slope x (T_air - t_ref): fitted to Carrier AquaForce 30XA ratings (19 air-cooled
    # units, 30-45 C air, 10 C leaving chilled water; median fit), scripts/fit_chiller_cop.py. Cross-checks:
    # Daikin EWAD-MZ 2.45-2.65 at 46 C; ASHRAE 90.1-2022 minimum 2.96 at 35 C (full load, path A).
    cop_ref: float = 3.57
    t_ref: float = 29.0
    cop_slope: float = 0.069        # per K
    cop_min: float = 2.0            # guard only; not reached with 2025 weather
    tes_hours: float = 4.0          # design choice
    tes_rate_frac: float = 0.5
    tes_loss_h: float = 0.001       # 2.4%/day, within the 1-5%/day tank losses of Roth et al. (ASHRAE J. 2006)
    tes_eff: float = 0.90           # upper end of the 55-90% tank-TES efficiency for district cooling (IRENA 2020)
    bess_mw: float = 25.0           # design choice
    bess_mwh: float = 50.0
    bess_eta: float = 0.85 ** 0.5   # one-way; round trip 0.85 (NREL ATB 2024, utility-scale Li-ion)
    reserve_min: float = 10.0       # UPS ride-through at full IT load
    pv_mwp: float = 30.0            # design choice

    def cop(self, temp):
        return np.maximum(self.cop_min, self.cop_ref - self.cop_slope * (np.asarray(temp, float) - self.t_ref))

    @property
    def peak_heat_mw(self):
        return self.it_mw * (1 + self.loss_frac + self.fan_frac)


@dataclass
class System:
    """Synthetic 2030 Abu Dhabi (EWEC) system. Sources: the comments below and Appendix A of the paper."""
    # 2030 gross peak forecast for EWEC "Global Demand" (Abu Dhabi system incl. exports to the Northern
    # Emirates, which the EWEC fleet serves): EWEC Statistical Report 2024, table p. 96 (chart p. 97).
    peak_gw: float = 26.181
    # 2030 energy = peak x the 2024 "Global" load factor from the same report, same scope as the peak:
    # 107,729 GWh (p. 32, "Global includes exports to Northern Emirates"; the 111,005 GWh column also counts
    # exchanges with other GCC states) at an 18,623 MW Global peak (p. 21). Checked against the PDF, 27 Sep 2026.
    energy_2024_gwh: float = 107729.0
    peak_2024_mw: float = 18623.0
    pv_gw: float = 14.0        # EWEC 2030 solar target (The National, 21 Sep 2026; MEES, 25 Sep 2026)
    nuclear_gw: float = 5.348  # Barakah, 4 x 1,337 MWe net reference unit power (IAEA PRIS)
    # Open-cycle units are marginal when net load is within ocgt_gw of its annual peak: the 2.6 GW of OCGT
    # EWEC recommends by 2027 "for a small number of very high-demand days" (SFCR 2024-2037, Sec. 4.1).
    ocgt_gw: float = 2.6
    t_base: float = 18.0       # ASSUMPTION: cooling-degree base temperature
    # Marginal fuel cost and CO2 of gas plants: gas price x heat rate, and IPCC default factor / efficiency.
    gas_usd_mmbtu: float = 3.0     # UAE gas price assumed in a peer-reviewed UAE CCGT study (Mondol & Carr 2017)
    # Merit order within the combined-cycle band: the marginal unit's net (LHV) efficiency falls linearly from
    # eta_ccgt_max at zero net load to eta_ccgt_min at the open-cycle threshold (mean 0.55). ASSUMPTION
    eta_ccgt_max: float = 0.58
    eta_ccgt_min: float = 0.52
    eta_ocgt: float = 0.34         # ASSUMPTION: representative net (LHV) efficiency, open cycle
    ef_gas_t_mwh_th: float = 56100 * 3.6e-6   # IPCC 2006 Vol. 2 Table 2.2: 56,100 kg CO2/TJ = 0.20196 t/MWh_th
    ef_avg: float = 0.19       # EWEC power emissions intensity projected for 2030, Base Case (SFCR 2024-2037, Sec. 5)
    # Peaking capacity value: EIA AEO2025 capital cost study, Case 4 (H-class simple cycle, 2023 USD),
    # annualized at the 7% discount rate and 25-year life of UAE IPP models (Mondol & Carr 2017).
    capex_ocgt_usd_kw: float = 835.5
    fom_ocgt_usd_kw_yr: float = 6.87
    discount_rate: float = 0.07
    life_yr: int = 25
    peak_hours: int = 100      # the net-load peak set K (Sec. III.E)

    @property
    def energy_twh(self):
        load_factor = self.energy_2024_gwh / (self.peak_2024_mw / 1000 * 8784)   # 2024 is a leap year
        return self.peak_gw * load_factor * 8760 / 1000

    def fuel_aed_kwh(self, eta):
        return self.gas_usd_mmbtu * 3.412 / np.asarray(eta, float) / 1000 * AED_PER_USD     # 3.412 MMBtu per MWh

    def eta_ccgt(self, frac):
        """Marginal combined-cycle efficiency at net load = frac x the open-cycle threshold (0 <= frac <= 1)."""
        return self.eta_ccgt_max - (self.eta_ccgt_max - self.eta_ccgt_min) * np.clip(frac, 0.0, 1.0)

    @property
    def cost_ccgt(self):          # AED/kWh, band mean
        return float(self.fuel_aed_kwh(0.5 * (self.eta_ccgt_max + self.eta_ccgt_min)))

    @property
    def cost_ocgt(self):
        return float(self.fuel_aed_kwh(self.eta_ocgt))

    @property
    def ef_ccgt(self):            # tCO2/MWh, band mean
        return self.ef_gas_t_mwh_th / (0.5 * (self.eta_ccgt_max + self.eta_ccgt_min))

    @property
    def ef_ocgt(self):
        return self.ef_gas_t_mwh_th / self.eta_ocgt

    @property
    def cap_value(self):          # AED/kW-yr
        r, n = self.discount_rate, self.life_yr
        crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
        return (self.capex_ocgt_usd_kw * crf + self.fom_ocgt_usd_kw_yr) * AED_PER_USD


@dataclass
class Tariffs:
    # TAQA Distribution (ADDC) Utility Tariff 2025, effective 1 Jan 2025 (unchanged from 2023)
    addc_commercial: float = 0.200   # AED/kWh flat
    addc_ind_peak: float = 0.366     # industrial > 1 MW, 10:00-22:00, 1 Jun-30 Sep
    addc_ind_off: float = 0.270
    addc_transmission: float = 0.231 # transmission-connected customers, fixed 1 Jan 2023-31 Dec 2027
    # DEWA slab tariff page (industrial: 0.23 up to 10,000 kWh/month, 0.38 above), fuel surcharge Sep 2026
    dewa_top_slab: float = 0.38      # AED/kWh
    dewa_fuel: float = 0.06          # AED/kWh; set monthly by DEWA


@dataclass
class Study:
    year: int = 2025
    weather_model: str = "era5"     # Open-Meteo dataset (ERA5 reanalysis, Hersbach et al. 2020)
    events: int = 10                 # mirrors the ten events of the 2024 Abu Dhabi pilot
    event_hours: int = 4
    summer_months: tuple = (6, 7, 8, 9)
    flex_days: int = 30
    reliability_q: float = 0.20      # 20th percentile = 80% reliability
    cap_pay_sweep: tuple = (0, 5, 10, 20, 35, 50, 100, 200, 350, 500)   # AED/kW-yr; System.cap_value is added
    breakeven_iter: int = 6          # bisection steps for the 95%-of-maximum payment
    gaming_inflation: float = 0.10
    gaming_days: int = 10
    placebo_days: int = 30
    # One-at-a-time sensitivity ranges for F(4), in the units of Appendix A (Table tab:params).
    sens: dict = field(default_factory=lambda: {
        "cop_slope": (0.05, 0.08),    # K^-1; spread across the 19 Carrier units (0.049-0.077)
        "cop_ref": (3.2, 4.2),        # COP at t_ref; low = weakest Carrier unit, high = warmer chilled water (assumption)
        "tes_hours": (0.0, 8.0),      # h of peak heat
        "bess_mw": (12.5, 50.0),      # MW; energy scaled with power (2 h); 12.5 MW keeps the UPS reserve feasible
        "deadline": (0.5, 2.0),       # multiplier on both deadlines (6/12 h to 24/48 h)
        "share_pct": (0.0, 40.0),     # deferrable work, % of mean IT load
        "pv_gw": (10.0, 35.0)})       # system PV in 2030, GW
    ercot_hub: str = "HB_WEST"
    ercot_tz: str = "America/Chicago"   # Texas (Central) time for the workload shapes in S4
    ercot_lat: float = 32.45         # West Texas site
    ercot_lon: float = -99.73
    seed: int = 7
