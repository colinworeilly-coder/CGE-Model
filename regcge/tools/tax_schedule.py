"""Convert a statutory income-tax schedule into effective average and marginal
rates for each household class.

The model's income taxes are linear in a broad income concept for each class
(``T = mtr * BASE - A``).  To simulate a statutory schedule (brackets,
standard deductions by filer type), each class is represented by its average
household; its adjusted gross income is ``agi_ratio`` times model income per
household (model income is a personal-income concept that includes items not
in AGI).  Tax and marginal rate are computed for each filer type and averaged
with the class's filer mix.  Rates are returned relative to *model* income, so
they can be plugged into the model directly.

STAMP made the same representative-taxpayer assumption ("households do not
move from one tax bracket to the next").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FILERS = ("single", "head_of_household", "married")


@dataclass
class Schedule:
    brackets: list                      # [[lower_threshold, rate], ...] on taxable income, ascending
    standard_deduction: dict = field(default_factory=lambda: {f: 0.0 for f in FILERS})
    personal_exemption: float = 0.0
    threshold_scale: float = 1.0        # multiply thresholds and deductions (e.g. to inflate old dollars)
    married_bracket_factor: float = 1.0  # married thresholds = factor x single thresholds

    @classmethod
    def from_dict(cls, d: dict) -> "Schedule":
        sd = d.get("standard_deduction", {})
        if isinstance(sd, (int, float)):
            sd = {f: float(sd) for f in FILERS}
        sd = {f: float(sd.get(f, sd.get("single", 0.0))) for f in FILERS}
        return cls(brackets=[[float(a), float(b)] for a, b in d["brackets"]],
                   standard_deduction=sd,
                   personal_exemption=float(d.get("personal_exemption", 0.0)),
                   threshold_scale=float(d.get("threshold_scale", 1.0)),
                   married_bracket_factor=float(d.get("married_bracket_factor", 1.0)))

    def tax(self, taxable: float, filer: str = "single") -> tuple[float, float]:
        """Tax liability and marginal rate at a taxable income."""
        f = self.threshold_scale * (self.married_bracket_factor if filer == "married" else 1.0)
        th = [b[0] * f for b in self.brackets]
        rt = [b[1] for b in self.brackets]
        t = 0.0
        mrate = 0.0
        for i, (lo, r) in enumerate(zip(th, rt)):
            hi = th[i + 1] if i + 1 < len(th) else np.inf
            if taxable > lo:
                t += (min(taxable, hi) - lo) * r
                mrate = r
        return t, (mrate if taxable > th[0] else 0.0)

    def household(self, agi: float, filer: str) -> tuple[float, float]:
        ded = (self.standard_deduction[filer] + self.personal_exemption) * self.threshold_scale
        taxable = max(agi - ded, 0.0)
        return self.tax(taxable, filer)


def effective_rates(schedule: Schedule | dict, income_per_household: np.ndarray, agi_ratio: np.ndarray,
                    share_married: np.ndarray, share_hoh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average and marginal tax rates relative to model income, by class.

    ``income_per_household`` is the model tax base per household in dollars.
    """
    sch = schedule if isinstance(schedule, Schedule) else Schedule.from_dict(schedule)
    inc = np.asarray(income_per_household, float)
    avg = np.zeros_like(inc)
    mtr = np.zeros_like(inc)
    for j, y in enumerate(inc):
        agi = agi_ratio[j] * y
        shares = {"married": share_married[j], "head_of_household": share_hoh[j]}
        shares["single"] = max(0.0, 1 - shares["married"] - shares["head_of_household"])
        t = m = 0.0
        for f, s in shares.items():
            tf, mf = sch.household(agi, f)
            t += s * tf
            m += s * mf
        avg[j] = t / y if y > 0 else 0.0
        mtr[j] = m * agi_ratio[j]
    return avg, mtr
