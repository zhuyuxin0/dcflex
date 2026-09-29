"""Synthetic 2030 Abu Dhabi system signal (Eq. net load). Calibrated to annual peak and energy."""
import numpy as np


def system_signal(temp, pv_norm, s, calib=None):
    """calib: the calibration returned by an earlier call (demand slope and intercept, combined-cycle band, peak
    threshold). A forecast signal reuses the calibration of the actual year, so that only its inputs differ."""
    c = np.maximum(np.asarray(temp, float) - s.t_base, 0.0)
    n = len(c)
    if calib is None:
        E, P = s.energy_twh * 1000.0, s.peak_gw                 # GWh, GW
        den = n * c.max() - c.sum()
        if den <= 0:
            raise ValueError("calibration infeasible: no variation in cooling degrees (check System.t_base)")
        k = (P * n - E) / den
        b = P - k * c.max()
        if not (np.isfinite(k) and np.isfinite(b)) or k <= 0 or b <= 0:
            raise ValueError("calibration infeasible: check System.peak_gw and System.energy_twh")
    else:
        k, b = calib["k"], calib["b"]
    demand = b + k * c
    pv = s.pv_gw * np.asarray(pv_norm, float)
    net = demand - pv - s.nuclear_gw
    # open-cycle units are marginal within ocgt_gw of the annual net-load peak (merit-order proxy)
    n_cc = max(float(net.max()) - s.ocgt_gw, 0.0) if calib is None else calib["n_cc"]
    if calib is None:
        peak = np.zeros(n, bool)
        peak[np.argsort(net)[-s.peak_hours:]] = True
        thr = float(net[peak].min())
    else:
        thr = calib["peak_thr"]
        peak = net >= thr
    mc, ef, price = _costs(net, s, n_cc, peak)
    return dict(demand=demand, pv=pv, net=net, mc=mc, ef=ef, surplus=net <= 0, peak=peak, n_cc=n_cc, price=price,
                calib=dict(k=float(k), b=float(b), n_cc=float(n_cc), peak_thr=thr))


def _costs(net, s, n_cc, peak):
    # within the combined-cycle band the marginal unit is less efficient the higher the net load
    eta = s.eta_ccgt(net / n_cc if n_cc > 0 else np.ones(len(net)))
    mc = np.where(net <= 0, 0.0, np.where(net <= n_cc, s.fuel_aed_kwh(eta), s.cost_ocgt))   # AED/kWh
    ef = np.where(net <= 0, 0.0, np.where(net <= n_cc, s.ef_gas_t_mwh_th / eta, s.ef_ocgt))  # tCO2/MWh
    # S2 price (AED/MWh): marginal energy cost plus the annualized cost of peaking capacity spread over the
    # net-load peak set K, so that S2 minimizes the numerator of the system value (Eq. vsys)
    price = 1000.0 * mc + np.where(peak, 1000.0 * s.cap_value / s.peak_hours, 0.0)
    return mc, ef, price


def price_from_net(net, s, calib):
    """S2 price (AED/MWh) of any net-load values (GW) under a given calibration; vectorized over any shape."""
    net = np.asarray(net, float)
    flat = net.ravel()
    return _costs(flat, s, calib["n_cc"], flat >= calib["peak_thr"])[2].reshape(net.shape)


def daily_peak_hour(net):
    return np.asarray(net).reshape(-1, 24).argmax(axis=1)
