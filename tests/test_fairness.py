import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))

import fairness_sim  # noqa: E402


def test_per_tenant_isolation_fully_serves_polite():
    offered, admitted, _ = fairness_sim.run_per_tenant(seed=1)
    polite = admitted["polite"] / offered["polite"]
    assert polite >= 0.95          # within-quota tenant is unaffected by the flood


def test_per_tenant_beats_shared_for_polite():
    o_pt, a_pt, _ = fairness_sim.run_per_tenant(seed=1)
    o_sh, a_sh, _ = fairness_sim.run_shared(seed=1)
    polite_pt = a_pt["polite"] / o_pt["polite"]
    polite_sh = a_sh["polite"] / o_sh["polite"]
    assert polite_pt > polite_sh + 0.3    # isolation clearly fairer than a shared bucket


def test_greedy_is_throttled_under_both():
    _, a_pt, t_pt = fairness_sim.run_per_tenant(seed=1)
    assert t_pt["greedy"] > a_pt["greedy"]   # most of the greedy flood is rejected
