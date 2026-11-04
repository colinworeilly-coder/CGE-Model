"""Calibration: benchmark extraction from the SAM and all structural parameters.

The calibration reproduces the SAM exactly at benchmark prices of one (except
factor user costs, which include factor taxes).  All functional forms use the
*calibrated share form* (Rutherford 1998): every CES/CET/Cobb-Douglas function
is written relative to its benchmark point, so share parameters are simply
benchmark value shares and zero flows are handled gracefully.

Sections follow the account types of the SAM:

1. canonicalisation (factor taxes paid from factor/enterprise columns are
   re-attributed to the employing sectors; income taxes paid from factor
   columns are re-attributed to households),
2. sectors / commodities and their taxes,
3. factors and capital (capital stock K0 = I0 / (g + delta) so that the
   baseline is a balanced-growth path),
4. households (income components, direct taxes with marginal/average rates and
   deductibility, ELES demand with leisure; labor-supply elasticity targets),
5. enterprise, governments, rest of world, investment.

Everything the model needs is stored on a :class:`Calibration` object.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.sets import Set
from ..data.balance import ras
from ..data.dataset import Dataset
from ..data.sam import check_balance


class CalibrationError(ValueError):
    pass


INCOME_COMPONENTS = ("labor", "capital", "transfers")
PURCHASERS = ("household", "intermediate", "investment", "government")


def canonicalize_sam(sam: pd.DataFrame, acc: dict) -> pd.DataFrame:
    """Re-attribute taxes so that every factor tax is paid by an employer
    column (sector or government) and every income tax by a household column."""
    S = sam.copy()
    sectors = list(acc["sectors"])
    govs = list(acc["governments"])
    hh = list(acc["households"])
    ents = list(acc.get("enterprises") or [])
    for k, spec in (acc.get("taxes") or {}).items():
        base = spec["base"]
        if base == "factor":
            f = spec["factor"]
            employers = [e for e in sectors + govs if S.loc[f, e] > 0]
            w = S.loc[f, employers] / S.loc[f, employers].sum()
            for payer in [f] + ents:
                R = S.loc[k, payer]
                if abs(R) < 1e-14:
                    continue
                for e in employers:
                    S.loc[k, e] += R * w[e]
                    S.loc[f, e] -= R * w[e]
                S.loc[k, payer] -= R
                if payer in ents:
                    S.loc[payer, f] -= R
        elif base == "income":
            for f in acc["factors"]:
                R = S.loc[k, f]
                if abs(R) < 1e-14:
                    continue
                recv = [h for h in hh if S.loc[h, f] > 0]
                w = S.loc[recv, f] / S.loc[recv, f].sum()
                for h in recv:
                    S.loc[k, h] += R * w[h]
                    S.loc[h, f] += R * w[h]
                S.loc[k, f] -= R
    check_balance(S, rel_tol=1e-8)
    return S


def _ces_ok(sigma: float) -> float:
    if not np.isfinite(sigma) or sigma < 0:
        raise CalibrationError(f"Invalid elasticity {sigma}")
    return float(sigma)


@dataclass
class Calibration:
    ds: Dataset
    spec: dict
    sets: dict = field(default_factory=dict)
    bench: dict = field(default_factory=dict)   # benchmark levels of model variables
    par: dict = field(default_factory=dict)     # structural constants
    p0: dict = field(default_factory=dict)      # base values of time-varying parameters
    info: dict = field(default_factory=dict)    # bookkeeping (jobs, labels, ...)
    warnings: list = field(default_factory=list)

    # ================================================================== main
    @classmethod
    def from_dataset(cls, ds: Dataset, spec: dict | None = None) -> "Calibration":
        cal = cls(ds, spec or ds.model)
        cal.S = canonicalize_sam(ds.sam, ds.accounts)
        cal._sets()
        cal._growth()
        cal._check_cells()
        cal._sectors()
        cal._investment()
        cal._factors()
        cal._sales_taxes()
        cal._households()
        cal._enterprise()
        cal._governments()
        cal._row()
        cal._income_taxes_and_labor()
        return cal

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        warnings.warn(msg, stacklevel=3)

    # ================================================================== sets
    def _sets(self):
        acc = self.ds.accounts
        A = Set("A", acc["sectors"])
        fac = acc["factors"]
        F = Set("F", list(fac.keys()))
        FL = Set("FL", [f for f, t in fac.items() if t == "labor"])
        FK = [f for f, t in fac.items() if t == "capital"]
        FX = Set("FX", [f for f, t in fac.items() if t == "fixed"])
        H = Set("H", acc["households"])
        G = Set("G", acc["governments"])
        ents = list(acc.get("enterprises") or [])
        if len(ents) > 1:
            raise CalibrationError("Only one enterprise account is supported; aggregate them in the SAM")
        taxes = acc.get("taxes") or {}
        K = Set("K", list(taxes.keys()))
        by_base = {b: Set("K_" + b, [k for k, s in taxes.items() if s["base"] == b])
                   for b in ("sales", "output", "factor", "property", "income")}
        if not FL.labels:
            raise CalibrationError("At least one labor factor is required")
        if not FK:
            raise CalibrationError("A capital factor is required")
        self.sets = dict(A=A, F=F, FL=FL, FX=FX, H=H, G=G, K=K, **{f"K{b[0].upper()}": s for b, s in by_base.items()})
        # KS sales, KO output, KF factor, KP property, KI income
        self.sets["KS"], self.sets["KO"], self.sets["KF"], self.sets["KP"], self.sets["KI"] = (
            by_base["sales"], by_base["output"], by_base["factor"], by_base["property"], by_base["income"])
        self.cap = FK[0]
        self.ent = ents[0] if ents else None
        self.inv = acc["investment"]
        self.row = acc["row"]
        self.taxes = taxes

    def _growth(self):
        gr = self.spec.get("growth", {})
        n = float(gr.get("population", 0.0))
        a = float(gr.get("productivity", 0.0))
        self.par["n"] = n
        self.par["a"] = a
        self.par["g"] = (1 + n) * (1 + a) - 1

    # ----------------------------------------------------- structural checks
    def _check_cells(self):
        """Raise on SAM cells the model has no interpretation for (shared rules
        in regcge.data.rules; the validator reports the same issues without raising)."""
        from ..data.rules import disallowed_nonzero, tax_payer_issues
        issues = disallowed_nonzero(self.S, self.ds.accounts, raw=False) + tax_payer_issues(self.S, self.ds.accounts)
        if issues:
            raise CalibrationError("SAM cells without a model interpretation (row <- column):\n  "
                                   + "\n  ".join(f"{i.row} <- {i.col} ({i.value:.4g}): {i.message}"
                                                  for i in issues[:30]))

    # ================================================================ sectors
    def _sectors(self):
        S = self.S
        A = self.sets["A"].labels
        sec = self.ds.sectors
        Z = S.loc[A, A].to_numpy()
        QM0 = S.loc[self.row, A].to_numpy()
        QE0 = S.loc[A, self.row].to_numpy()
        QX0 = S[A].sum(axis=0).to_numpy() - QM0
        if (QX0 <= 0).any():
            raise CalibrationError(f"Sectors with zero output are not supported: {[a for a, x in zip(A, QX0) if x <= 0]}")
        QD0 = QX0 - QE0
        if (QD0 < -1e-9).any():
            raise CalibrationError("Exports exceed output for some sector")
        QD0 = np.maximum(QD0, 0.0)
        QQ0 = QD0 + QM0
        self.bench.update(QX=QX0, QM=QM0, QE=QE0, QD=QD0, QQ=QQ0, QINT=Z)
        tout = {}
        for k in self.sets["KO"]:
            tout[k] = S.loc[k, A].to_numpy() / QX0
        self.par["tout"] = tout
        # elasticities
        self.par["sigma_va"] = sec["sigma_va"].to_numpy(float)
        self.par["sigma_top"] = sec["sigma_top"].to_numpy(float)
        self.par["sigma_arm"] = sec["sigma_armington"].to_numpy(float)
        cet = pd.to_numeric(sec["cet_elasticity"], errors="coerce").fillna(np.inf).to_numpy(float)
        self.par["cet"] = cet
        self.par["eta_exp"] = pd.to_numeric(sec["export_elasticity"], errors="coerce").fillna(np.inf).to_numpy(float)
        self.par["delta"] = sec["depreciation"].to_numpy(float)
        self.par["eta_inv"] = sec["inv_elasticity"].to_numpy(float)
        for nm in ("sigma_va", "sigma_top", "sigma_arm"):
            for s in self.par[nm]:
                _ces_ok(s)

    # ============================================================= investment
    def _investment(self):
        S = self.S
        A = self.sets["A"].labels
        inv = self.inv
        QINV0 = S.loc[A, inv].to_numpy()
        TINV0 = S[inv].sum()
        self.bench["QINV"] = QINV0
        self.par["TINV0"] = TINV0
        cap_inc = S.loc[self.cap, A].to_numpy()
        share = self.ds.sectors["inv_share"].to_numpy(float)
        if np.all(np.isnan(share)):
            share = cap_inc / cap_inc.sum()
        else:
            share = np.nan_to_num(share, nan=0.0)
            share = share / share.sum()
        if ((share <= 0) & (cap_inc > 0)).any():
            raise CalibrationError("Every sector with capital income needs a positive investment share")
        self.info["inv_share"] = share
        # capital-coefficient matrix B[c, a]: quantity of commodity c per unit of capital good a
        cc = self.ds.capital_coefficients
        if cc is None:
            comp = QINV0 / QINV0.sum() if QINV0.sum() > 0 else np.zeros(len(A))
            V = np.outer(comp, share) * QINV0.sum()
        else:
            M = cc.reindex(index=A, columns=A).fillna(0.0).to_numpy(float)
            M = M / np.where(M.sum(axis=0) > 0, M.sum(axis=0), 1.0)
            prior = M * share[None, :] * QINV0.sum() + 1e-12 * (QINV0[:, None] > 0)
            V = ras(prior, QINV0, share * QINV0.sum())
        self.info["V_inv"] = V  # basic value of commodity c bought for destination a
        # sales taxes on investment purchases: one rate vector by commodity
        tsv = {}
        for k in self.sets["KS"]:
            st = self._structure(k)
            tsv[k] = self._scale_rates(k, S.loc[k, inv], st["investment"].to_numpy(), V.sum(axis=1), inv)
        self.par["tsv"] = tsv
        tot_v = sum(tsv.values()) if tsv else np.zeros(len(A))
        IND_val = share * TINV0                                  # purchaser value by destination
        PK0 = np.where(IND_val > 0, (V * (1 + tot_v)[:, None]).sum(axis=0) / np.where(IND_val > 0, IND_val, 1), 1.0)
        IND0 = IND_val / PK0                                     # real units
        self.par["B"] = V / np.where(IND0 > 0, IND0, 1)[None, :]   # commodity per unit of capital good
        self.par["PK0"] = PK0
        self.par["wsav"] = IND_val / IND_val.sum()               # weights of the saving price index
        self.bench["IND"] = IND0
        tinv = ((self.par["B"] @ IND0) * (1 + tot_v)).sum()
        if not np.isclose(tinv, TINV0, rtol=1e-8):
            raise CalibrationError("Investment value does not reproduce the SAM")

    # =============================================================== factors
    def _factors(self):
        S = self.S
        A = self.sets["A"].labels
        G = self.sets["G"].labels
        H = self.sets["H"].labels
        g, delta = self.par["g"], self.par["delta"]
        fac = self.ds.accounts["factors"]
        # labor
        FL = self.sets["FL"].labels
        QF0_L = S.loc[FL, A].to_numpy()                  # (FL x A) net wage payments
        QFG0 = S.loc[FL, G].to_numpy()                   # government employment
        YLROW0 = S.loc[FL, self.row].to_numpy()          # residents' out-of-state earnings
        LSNR0 = S.loc[self.row, FL].to_numpy()           # non-resident labor income
        hh_lab = S.loc[H, FL].to_numpy().T               # (FL x H) gross receipts
        tot = hh_lab.sum(axis=1)
        LS0 = hh_lab * (1 - np.where(tot > 0, YLROW0 / np.where(tot > 0, tot, 1), 0))[:, None]
        demand = QF0_L.sum(axis=1) + QFG0.sum(axis=1)
        if not np.allclose(LS0.sum(axis=1) + LSNR0, demand, rtol=1e-8, atol=1e-8):
            raise CalibrationError("Labor supply and demand do not match in the SAM")
        self.bench.update(QF_L=QF0_L, QFG=QFG0, LSl=LS0)
        self.par["YLROW0"] = YLROW0
        self.par["nrsh"] = LSNR0 / demand
        self.par["rowlab"] = hh_lab / np.where(tot > 0, tot, 1)[:, None]
        # capital
        cap = self.cap
        KINC0 = S.loc[cap, A].to_numpy()
        IND0 = self.bench["IND"]
        PK0 = self.par["PK0"]
        K0 = IND0 / (g + delta)
        hasK = KINC0 > 0
        RK0 = np.where(hasK, KINC0 / np.where(K0 > 0, K0, 1), 1.0)
        self.bench.update(K=K0, RK=RK0, KINC=KINC0)
        self.par["hasK"] = hasK
        self.par["YKG0"] = S.loc[cap, G].to_numpy()      # government capital rentals (exogenous)
        self.par["YKROW0"] = S.loc[cap, self.row]
        recips = [r for r in S.index if abs(S.loc[r, cap]) > 1e-12]
        totc = S.loc[recips, cap].sum()
        self.par["thK"] = {r: S.loc[r, cap] / totc for r in recips}
        # fixed factors (land, natural resources): sector-specific, grow with the economy
        FX = self.sets["FX"].labels
        self.bench["QF_X"] = S.loc[FX, A].to_numpy() if FX else np.zeros((0, len(A)))
        self.par["thX"] = {}
        self.par["YXROW0"] = {}
        for x in FX:
            rec = [r for r in S.index if abs(S.loc[r, x]) > 1e-12]
            t = S.loc[rec, x].sum()
            self.par["thX"][x] = {r: S.loc[r, x] / t for r in rec}
            self.par["YXROW0"][x] = S.loc[x, self.row]
        # factor taxes by instrument: rates on sector payments and on government employment
        tf, tfg, tprop = {}, {}, {}
        for k in self.sets["KF"]:
            f = self.taxes[k]["factor"]
            pay = S.loc[f, A].to_numpy()
            tf[k] = np.where(pay > 0, S.loc[k, A].to_numpy() / np.where(pay > 0, pay, 1), 0.0)
            payg = S.loc[f, G].to_numpy()
            tfg[k] = np.where(payg > 0, S.loc[k, G].to_numpy() / np.where(payg > 0, payg, 1), 0.0)
        for k in self.sets["KP"]:
            tprop[k] = np.where(K0 > 0, S.loc[k, A].to_numpy() / np.where(K0 > 0, PK0 * K0, 1), 0.0)
            if ((S.loc[k, A].to_numpy() > 0) & (K0 <= 0)).any():
                raise CalibrationError(f"Property tax {k} paid by a sector without capital")
        self.par["tf"], self.par["tfg"], self.par["tprop"] = tf, tfg, tprop
        self.par["tax_factor"] = {k: self.taxes[k]["factor"] for k in self.sets["KF"]}
        # user costs and value added
        F = self.sets["F"].labels
        nF, nA = len(F), len(A)
        QF0 = np.zeros((nF, nA))
        UC0 = np.ones((nF, nA))
        for i, f in enumerate(F):
            ftaxes = [k for k in self.sets["KF"] if self.taxes[k]["factor"] == f]
            ad = 1 + sum(tf[k] for k in ftaxes) if ftaxes else np.ones(nA)
            if fac[f] == "labor":
                QF0[i] = S.loc[f, A].to_numpy()
                UC0[i] = ad
            elif fac[f] == "capital":
                QF0[i] = K0
                UC0[i] = RK0 * ad + (sum(tprop[k] for k in self.sets["KP"]) * PK0 if self.sets["KP"].labels else 0.0)
            else:
                QF0[i] = S.loc[f, A].to_numpy()
                UC0[i] = ad
        self.bench["QF"] = QF0
        self.par["UC0"] = UC0
        QVA0 = (UC0 * QF0).sum(axis=0)
        self.bench["QVA"] = QVA0
        self.par["thva"] = (UC0 * QF0) / QVA0[None, :]

    # ============================================================ sales taxes
    def _structure(self, k: str) -> pd.DataFrame:
        A = self.sets["A"].labels
        ts = self.ds.tax_structure.get(k)
        if ts is None:
            ts = pd.DataFrame(1.0, index=A, columns=list(PURCHASERS))
        ts = ts.reindex(index=A).fillna(0.0)
        for p in PURCHASERS:
            if p not in ts.columns:
                ts[p] = 1.0 if self.ds.tax_structure.get(k) is None else 0.0
        return ts[list(PURCHASERS)].astype(float)

    def _scale_rates(self, k: str, R: float, w: np.ndarray, base: np.ndarray, who: str) -> np.ndarray:
        if abs(R) < 1e-14:
            return np.zeros_like(w)
        B = float((w * base).sum())
        if B <= 1e-14:
            self.warn(f"Sales tax {k}: {who} pays {R:.4g} but has no taxable purchases under the "
                      f"structure file; using a uniform rate on all its purchases instead.")
            w = np.ones_like(w)
            B = float(base.sum())
        return (R / B) * w

    def _sales_taxes(self):
        S = self.S
        A, H, G = self.sets["A"].labels, self.sets["H"].labels, self.sets["G"].labels
        Z = self.bench["QINT"]
        QH0 = S.loc[A, H].to_numpy()
        QG0 = S.loc[A, G].to_numpy()
        tsh, tsi, tsg = {}, {}, {}
        self.info["structure"] = {}
        for k in self.sets["KS"]:
            st = self._structure(k)
            self.info["structure"][k] = st
            tsh[k] = np.column_stack([self._scale_rates(k, S.loc[k, h], st["household"].to_numpy(), QH0[:, j], h)
                                      for j, h in enumerate(H)])
            tsi[k] = np.column_stack([self._scale_rates(k, S.loc[k, a], st["intermediate"].to_numpy(), Z[:, j], a)
                                      for j, a in enumerate(A)])
            tsg[k] = np.column_stack([self._scale_rates(k, S.loc[k, g], st["government"].to_numpy(), QG0[:, j], g)
                                      for j, g in enumerate(G)])
        self.par.update(tsh=tsh, tsi=tsi, tsg=tsg)
        nA = len(A)
        tot_i = sum(tsi.values()) if tsi else np.zeros((nA, nA))
        # intermediate bundle and top nest
        QINTA0 = (Z * (1 + tot_i)).sum(axis=0)
        self.bench["QINTA"] = QINTA0
        self.par["icoef"] = Z / np.where(QINTA0 > 0, QINTA0, 1)[None, :]
        # benchmark top nest of production
        tot_o = sum(self.par["tout"].values()) if self.par["tout"] else np.zeros(nA)
        QVA0 = self.bench["QVA"]
        lhs = self.bench["QX"] * (1 - tot_o)
        if not np.allclose(lhs, QVA0 + QINTA0, rtol=1e-8):
            bad = [a for a, l, r in zip(A, lhs, QVA0 + QINTA0) if not np.isclose(l, r, rtol=1e-8)]
            raise CalibrationError(f"Zero-profit check fails at benchmark for sectors {bad}")
        self.par["c0"] = 1 - tot_o
        self.par["thv_top"] = QVA0 / (QVA0 + QINTA0)

    # ============================================================= households
    def _households(self):
        S = self.S
        A, H, G = self.sets["A"].labels, self.sets["H"].labels, self.sets["G"].labels
        hh = self.ds.households
        nA, nH = len(A), len(H)
        YH0 = S.loc[H].sum(axis=1).to_numpy()
        QH0 = S.loc[A, H].to_numpy()
        tot_h = sum(self.par["tsh"].values()) if self.par["tsh"] else np.zeros((nA, nH))
        PH0 = 1 + tot_h
        QROWC0 = S.loc[self.row, H].to_numpy()
        SAVH0 = S.loc[self.inv, H].to_numpy()
        KI = self.sets["KI"].labels
        TAXH0 = S.loc[KI, H].to_numpy() if KI else np.zeros((0, nH))
        YD0 = YH0 - TAXH0.sum(axis=0)
        spend = (PH0 * QH0).sum(axis=0) + QROWC0 + SAVH0
        if not np.allclose(spend, YD0, rtol=1e-8):
            raise CalibrationError("Household budgets do not close at benchmark")
        if (YD0 <= 0).any():
            raise CalibrationError("Non-positive disposable income for some household")
        N0 = hh["households"].to_numpy(float)
        if np.isnan(N0).any():
            N0 = np.where(np.isnan(N0), YH0 * float(self.ds.accounts.get("dollar_scale", 1e9)) / 75_000.0, N0)
            self.warn("households.csv lacks household counts; assuming $75,000 of income per household")
        self.bench.update(YH=YH0, YD=YD0, QH=QH0, QROWC=QROWC0, SAVH=SAVH0, TAXH=TAXH0, N=N0, CPI=np.ones(nH))
        self.par["PH0"] = PH0
        # income components
        FL = self.sets["FL"].labels
        YL0 = S.loc[H, FL].sum(axis=1).to_numpy()
        cap_like = [self.cap] + self.sets["FX"].labels + ([self.ent] if self.ent else [])
        YK0 = S.loc[H, cap_like].sum(axis=1).to_numpy()
        TR0 = S.loc[H, G + [self.row]].sum(axis=1).to_numpy()
        self.bench.update(YL=YL0, YK=YK0, TR=TR0)
        self.par["trpc"] = S.loc[H, G].to_numpy() / N0[:, None]      # (H x G) per household
        self.par["TRROW0"] = S.loc[H, self.row].to_numpy()
        self.par["div"] = None  # filled in _enterprise
        cons = (PH0 * QH0).sum(axis=0) + QROWC0
        self.par["wc0"] = (PH0 * QH0) / cons[None, :]
        self.par["wr0"] = QROWC0 / cons
        self.par["wcpi"] = cons / cons.sum()

    # ============================================================ enterprise
    def _enterprise(self):
        S = self.S
        H, G = self.sets["H"].labels, self.sets["G"].labels
        if not self.ent:
            self.par.update(div=np.zeros(len(H)), divROW=0.0, divG=np.zeros(len(G)),
                            ENTG0=np.zeros(len(G)), ENTROW0=0.0)
            self.bench.update(YENT=0.0, SENT=0.0)
            return
        e = self.ent
        YENT0 = S.loc[e].sum()
        self.bench["YENT"] = YENT0
        self.bench["SENT"] = S.loc[self.inv, e]
        self.par["div"] = S.loc[H, e].to_numpy() / YENT0
        self.par["divROW"] = S.loc[self.row, e] / YENT0
        self.par["divG"] = S.loc[G, e].to_numpy() / YENT0
        self.par["ENTG0"] = S.loc[e, G].to_numpy()       # government transfers to enterprises (exog)
        self.par["ENTROW0"] = S.loc[e, self.row]

    # ============================================================ governments
    def _governments(self):
        S = self.S
        A, H, G = self.sets["A"].labels, self.sets["H"].labels, self.sets["G"].labels
        K = self.sets["K"].labels
        FL = self.sets["FL"].labels
        nA, nG = len(A), len(G)
        YG0 = S.loc[G].sum(axis=1).to_numpy()
        self.bench["YG"] = YG0
        # tax revenue shares
        shTG = np.zeros((len(K), nG))
        for i, k in enumerate(K):
            col = S.loc[G, k].to_numpy()
            if col.sum() > 1e-12:
                shTG[i] = col / col.sum()
            else:
                gv = self.taxes[k].get("government")
                if gv is None:
                    raise CalibrationError(f"Tax {k} has no revenue in the SAM; declare its 'government' in accounts.yaml")
                if isinstance(gv, dict):
                    for gg, v in gv.items():
                        shTG[i, G.index(gg)] = v
                else:
                    shTG[i, G.index(gv)] = 1.0
        self.par["shTG"] = shTG
        self.bench["TAXR"] = S.loc[K].sum(axis=1).to_numpy()
        # spending bundle
        QG0 = S.loc[A, G].to_numpy()
        tot_g = sum(self.par["tsg"].values()) if self.par["tsg"] else np.zeros((nA, nG))
        QFG0 = self.bench["QFG"]
        tfg_tot = np.zeros((len(FL), nG))
        for k in self.sets["KF"]:
            f = self.taxes[k]["factor"]
            if f in FL:
                tfg_tot[FL.index(f)] += self.par["tfg"][k]
        QGL0 = (QG0 * (1 + tot_g)).sum(axis=0) + (QFG0 * (1 + tfg_tot)).sum(axis=0)
        if (QGL0 <= 0).any():
            self.warn("A government has no purchases of goods or labor; its spending closure is degenerate")
        QGL0 = np.where(QGL0 > 0, QGL0, 1.0)
        self.bench["QGL"] = QGL0
        self.bench["QG"] = QG0
        self.par["qgc"] = QG0 / QGL0[None, :]
        self.par["qfg"] = QFG0 / QGL0[None, :]
        # intergovernmental transfers TRGG[to, from]
        TRGG0 = S.loc[G, G].to_numpy()
        mode = self.ds.governments["intergov_mode"].astype(str).to_numpy()
        self.par["intergov_share"] = np.array([m == "share" for m in mode])
        self.par["mu"] = TRGG0 / YG0[None, :]
        self.par["TRGG0"] = TRGG0
        self.par["GENT0"] = S.loc[self.ent, G].to_numpy() if self.ent else np.zeros(nG)
        self.par["GROW0"] = S.loc[self.row, G].to_numpy()
        self.par["ROWG0"] = S.loc[G, self.row].to_numpy()
        self.bench["SG"] = S.loc[self.inv, G].to_numpy()
        self.info["gov_jobs"] = self.ds.governments["jobs"].to_numpy(float)

    # ============================================================ rest of world
    def _row(self):
        S = self.S
        self.bench["SROW"] = S.loc[self.inv, self.row]

    # ========================================== income taxes, labor & leisure
    def _income_taxes_and_labor(self):
        S = self.S
        H = self.sets["H"].labels
        KI = self.sets["KI"].labels
        KS = self.sets["KS"].labels
        hh = self.ds.households
        nH = len(H)
        b = self.bench
        comp = {"labor": b["YL"], "capital": b["YK"], "transfers": b["TR"]}
        ded = hh["salt_deduction_share"].to_numpy(float)
        self.par["ded"] = ded
        # household-paid amounts of every potentially deductible tax
        A = self.sets["A"].labels
        paid0 = {}
        for k in KS:
            paid0[k] = (self.par["tsh"][k] * b["QH"]).sum(axis=0)
        for i, k in enumerate(KI):
            paid0[k] = b["TAXH"][i]
        wcomp, dlist, BASE0, MTR, AV, A0 = {}, {}, {}, {}, {}, {}
        for i, k in enumerate(KI):
            spec = self.taxes[k]
            cs = spec.get("components", list(INCOME_COMPONENTS))
            if isinstance(cs, dict):
                w = {c: float(cs.get(c, 0.0)) for c in INCOME_COMPONENTS}
            else:
                w = {c: (1.0 if c in cs else 0.0) for c in INCOME_COMPONENTS}
            wcomp[k] = w
            dl = list(spec.get("deductible", []) or [])
            for d in dl:
                if d not in KI and d not in KS:
                    raise CalibrationError(f"Tax {k}: deductible tax {d} must be an income or sales tax")
            dlist[k] = dl
        # bases need all paid amounts (deductions refer to other taxes' payments)
        for i, k in enumerate(KI):
            base = sum(wcomp[k][c] * comp[c] for c in INCOME_COMPONENTS)
            base = base - ded * sum(paid0[d] for d in dlist[k]) if dlist[k] else base
            BASE0[k] = base
            R = b["TAXH"][i]
            avg = np.where(base > 0, R / np.where(base > 0, base, 1), 0.0)
            col = f"mtr_{k}"
            m = hh[col].to_numpy(float) if col in hh.columns else avg
            m = np.where(np.isnan(m), avg, m)
            MTR[k], AV[k] = m, avg
            A0[k] = m * base - R
        self.par.update(wcomp=wcomp, dlist=dlist, BASE0=BASE0, mtr0=MTR, avg0=AV, A0=A0)
        # effective marginal tax rate on labor income (with deductibility interactions)
        mtr_lab = effective_labor_mtr(KI, wcomp, dlist, {k: MTR[k] for k in KI}, ded, nH)
        self.par["MTRL0"] = mtr_lab
        # ------------------------------------------------------- labor supply
        LS0 = b["LSl"].sum(axis=0)                        # resident in-state labor (efficiency units)
        self.bench["LS"] = LS0
        self.par["lshare"] = b["LSl"] / np.where(LS0 > 0, LS0, 1)[None, :]
        exo = self.spec.get("closures", {}).get("labor_supply", "endogenous") == "exogenous"
        psi = np.ones(nH) if exo else hh["time_endowment_ratio"].to_numpy(float)
        if (psi < 1).any():
            raise CalibrationError("time_endowment_ratio must be >= 1")
        LEIS0 = (psi - 1) * LS0
        E0 = LS0 + LEIS0
        PL0 = 1.0 - mtr_lab
        if (PL0 <= 0).any():
            raise CalibrationError("Effective marginal tax rate on labor >= 100% for some household")
        self.bench["LEIS"] = LEIS0
        self.par["epc"] = E0 / b["N"]
        self.par["PL0"] = PL0
        # ---------------------------------------------------------------- ELES
        nA = len(A)
        items_P0 = np.vstack([self.par["PH0"], np.ones((1, nH)), np.ones((1, nH))])     # (nA+2) x H
        items_X0 = np.vstack([b["QH"], b["QROWC"][None, :], b["SAVH"][None, :]])
        YD0 = b["YD"]
        wshare = items_P0 * items_X0 / YD0[None, :]
        labels = A + ["ROW", "SAV"]
        ie = np.ones((nA + 2, nH))
        if self.ds.income_elasticities is not None:
            tab = self.ds.income_elasticities
            for i, lab in enumerate(labels):
                if lab in tab.index:
                    row = tab.loc[lab]
                    for j, h in enumerate(H):
                        if h in tab.columns:
                            ie[i, j] = float(row[h])
                        elif "all" in tab.columns:
                            ie[i, j] = float(row["all"])
        beta = ie * wshare
        nonpos = items_X0 <= 0
        beta = np.where(nonpos, 0.0, beta)
        # households with non-positive saving get a small positive marginal saving share
        beta = np.where(nonpos & (np.arange(nA + 2)[:, None] == nA + 1), 0.01, beta)
        beta = beta / beta.sum(axis=0, keepdims=True)
        frisch = hh["frisch"].to_numpy(float)
        if (frisch >= 0).any():
            raise CalibrationError("Frisch parameters must be negative")
        SUP0 = YD0 / (-frisch)
        gamma = items_X0 - beta * SUP0[None, :] / items_P0
        # consistency: committed expenditure + supernumerary = YD
        if not np.allclose((items_P0 * gamma).sum(axis=0) + SUP0, YD0, rtol=1e-8):
            raise CalibrationError("ELES calibration does not close")
        if (SUP0 <= 0).any():
            raise CalibrationError("Non-positive supernumerary income; adjust Frisch parameters")
        self.par["beta"] = beta
        self.par["gamma_pc"] = gamma / b["N"][None, :]
        self.par["items_P0"] = items_P0
        self.bench["Z"] = SUP0  # supernumerary bundle (PZ0 = 1)
        FS0 = SUP0 + PL0 * LEIS0
        thL = PL0 * LEIS0 / FS0
        self.par["thL"] = thL
        self.par["thZ"] = SUP0 / FS0
        self.bench["U"] = FS0
        # elasticity of substitution between leisure and goods from target labor-supply elasticity
        eps = hh["labor_supply_elasticity"].to_numpy(float)
        sig = np.ones(nH)
        for j in range(nH):
            if LEIS0[j] <= 0:
                sig[j] = 1.0
                continue
            wE_FS = PL0[j] * E0[j] / FS0[j]
            s = 1 - (1 - wE_FS - eps[j] * LS0[j] / LEIS0[j]) / (1 - thL[j])
            if s < 0.05:
                self.warn(f"Labor-supply elasticity target {eps[j]} for {H[j]} implies sigma {s:.3f}; "
                          f"using 0.05 (lower the target or raise the time endowment)")
                s = 0.05
            sig[j] = s
        self.par["sigma_ls"] = sig
        self.par["eps_ls"] = eps
        # migration
        self.par["mig_eta"] = hh["migration_elasticity"].to_numpy(float)
        self.par["mig_speed"] = hh["migration_speed"].to_numpy(float)
        self.bench["XBAR"] = YD0 / b["N"]        # real disposable income per household (CPI0 = 1)


def effective_labor_mtr(KI, wcomp, dlist, mtr, ded, nH, xp=np):
    """Total marginal tax rate on labor income, accounting for deductibility.

    With linear schedules ``T_k = m_k (sum_c w_kc I_c - ded * sum_{d in D_k} T_d) - A_k``,
    the response of the vector of taxes to labor income solves
    ``(I + diag(m) D) dT = diag(m) w_L``.  Works on numpy arrays (calibration)
    and on CasADi expressions (model).
    """
    import casadi as ca
    K = list(KI)
    nK = len(K)
    if nK == 0:
        return np.zeros(nH) if xp is np else ca.SX.zeros(nH, 1)
    out = []
    for j in range(nH):
        if xp is np:
            M = np.eye(nK)
            rhs = np.zeros(nK)
        else:
            M = ca.SX.eye(nK)
            rhs = ca.SX.zeros(nK, 1)
        for i, k in enumerate(K):
            mk = mtr[k][j]
            rhs[i] = mk * wcomp[k]["labor"]
            for d in dlist[k]:
                if d in K:
                    M[i, K.index(d)] = M[i, K.index(d)] + mk * ded[j]
        if xp is np:
            out.append(np.linalg.solve(M, rhs).sum())
        else:
            out.append(ca.sum1(ca.solve(M, rhs)))
    return np.array(out) if xp is np else ca.vertcat(*out)
