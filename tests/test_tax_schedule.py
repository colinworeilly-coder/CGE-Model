import numpy as np

from regcge.tools.tax_schedule import Schedule, effective_rates

SCH = {"brackets": [[0, 0.02], [10000, 0.03], [75000, 0.04]],
       "standard_deduction": {"single": 4750, "head_of_household": 7000, "married": 9500}}


def test_liability_hand_computed():
    s = Schedule.from_dict(SCH)
    # single, AGI 50,000: taxable 45,250 -> 2% x 10,000 + 3% x 35,250
    t, m = s.household(50_000, "single")
    assert np.isclose(t, 200 + 0.03 * 35_250) and m == 0.03
    # married, AGI 100,000: taxable 90,500 -> 200 + 3% x 65,000 + 4% x 15,500
    t, m = s.household(100_000, "married")
    assert np.isclose(t, 200 + 1950 + 620) and m == 0.04
    # below the deduction: no tax
    assert s.household(4000, "single") == (0.0, 0.0)


def test_effective_rates_relative_to_model_income():
    avg, mtr = effective_rates(SCH, np.array([100_000.0]), np.array([0.5]), np.array([0.0]), np.array([0.0]))
    t, m = Schedule.from_dict(SCH).household(50_000, "single")
    assert np.isclose(avg[0], t / 100_000) and np.isclose(mtr[0], m * 0.5)
