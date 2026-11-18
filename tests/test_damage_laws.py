"""The damage laws added to :mod:`morphogen.homogenization.laws`, against their closed forms
at a material point: Lemaitre's damage coupled to plasticity (a perfectly plastic and a
hardening bar in tension, the triaxiality, the closure of the microcracks in compression,
the cycles to failure of a strain range, a point that breaks), Kachanov's damage of creep
(the time to rupture, the damage and the tertiary creep under a constant stress, Hayhurst's
stress), Mazars' law of concrete (tension and compression, the shares of a biaxial state),
the unilateral effect of the spectral split, the tangents of all of them by differences,
their configurations and errors, and a composite whose matrix is damaged, by finite
elements and by FFT."""

import numpy as np
import pytest

from morphogen.homogenization import homogenize
from morphogen.homogenization.laws import (
    DUCTILE_LAWS,
    SPLITS,
    Damage,
    DuctileDamage,
    Mazars,
    analytic_model,
    damage_from_config,
)
from morphogen.homogenization.materials import Material, materials_from_config
from morphogen.homogenization.nonlinear import simulate_point
from morphogen.image import VoxelImage

E, NU, R0 = 200e3, 0.3, 400.0
FULL = {c: 0.0 for c in ("11", "22", "33", "23", "31", "12")}


def steel(damage, **plasticity):
    return Material(young=E, poisson=NU, plasticity={"yield": R0, **plasticity}, damage=damage)


def tension(end, increments=100):
    return {"times": [0, 1.0], "increments": increments, "strain": {"11": [0, end]}}


# -- Lemaitre ------------------------------------------------------------------------------


@pytest.mark.parametrize("s", [1.0, 2.0])
def test_lemaitre_damage_of_a_perfectly_plastic_bar(s):
    """A perfectly plastic bar in tension: the effective stress stays at the yield stress,
    the energy release rate at ``R0² / (2 E)`` and the damage grows as ``(Y / S)^s (p -
    p_D)`` beyond the threshold; the stress is ``(1 - D) R0``."""
    strength, threshold = 2.0, 0.01
    m = steel({"law": "lemaitre", "S": strength, "s": s, "threshold": threshold})
    r = simulate_point(m, tension(0.3), tangent="full")
    e, d = r.strain[:, 0], r.plastic
    p = np.maximum(e - R0 / E, 0.0)  # of the effective stress
    exact = (R0**2 / (2 * E * strength)) ** s * np.maximum(p - threshold, 0.0)
    np.testing.assert_allclose(d, exact, atol=1e-12)
    np.testing.assert_allclose(r.stress[:, 0], (1 - exact) * np.minimum(E * e, R0), rtol=1e-9)
    assert np.abs(r.stress[:, 1:]).max() < 1e-8 * R0 and d[-1] > 0
    # unloaded elastically along the damaged modulus (1 - D) E
    cycle = {"times": [0, 1, 2], "increments": [50, 10], "strain": {"11": [0, 0.3, 0.299]}}
    back = simulate_point(m, cycle, tangent="full")
    slope = (back.stress[-1, 0] - back.stress[50, 0]) / (back.strain[-1, 0] - back.strain[50, 0])
    assert slope == pytest.approx((1 - back.plastic[50]) * E, rel=1e-9)
    assert back.plastic[-1] == back.plastic[50]


HILL = {"11": 1.5, "22": 0.8, "33": 1.2, "12": 1.3, "23": 0.9, "31": 1.1}


@pytest.mark.parametrize("s", [1.0, 2.0])
def test_lemaitre_damage_with_hill_criterion(s):
    """A perfectly plastic bar of Hill's criterion pulled along its first axis: it yields
    at ``R0 / c``, ``c = sqrt((4 h11 + h22 + h33) / 6)``, its cumulated plastic strain is
    its plastic strain along the axis over ``c`` (the norm dual to the criterion), and the
    damage grows as ``(Y / S)^s (p - p_D)`` with ``Y`` that of its yield stress."""
    strength, threshold = 2.0, 0.01
    m = steel({"law": "lemaitre", "S": strength, "s": s, "threshold": threshold}, hill=HILL)
    r = simulate_point(m, tension(0.3), tangent="full")
    c = np.sqrt((4 * HILL["11"] + HILL["22"] + HILL["33"]) / 6)
    limit = R0 / c
    assert m.plasticity.yield_stresses()["11"] == pytest.approx(limit)
    e = r.strain[:, 0]
    p = np.maximum(e - limit / E, 0.0) / c
    exact = (limit**2 / (2 * E * strength)) ** s * np.maximum(p - threshold, 0.0)
    np.testing.assert_allclose(r.plastic, exact, atol=1e-11)
    np.testing.assert_allclose(r.stress[:, 0], (1 - exact) * np.minimum(E * e, limit), rtol=1e-9)
    assert np.abs(r.stress[:, 1:]).max() < 1e-8 * R0 and r.plastic[-1] > 0
    # the contraction of the bar is not that of von Mises: more along 3 than along 2
    assert r.strain[-1, 1] != pytest.approx(r.strain[-1, 2], rel=0.05)
    # coefficients of 1: von Mises' criterion
    path = {"times": [0, 1.0], "increments": 40, "strain": {"11": [0, 0.1], "12": [0, 0.05]}}
    law = {"law": "lemaitre", "S": strength, "s": s, "threshold": threshold}
    one = simulate_point(steel(law, hill=dict.fromkeys(HILL, 1.0)), path, tangent="full")
    mises = simulate_point(steel(law), path, tangent="full")
    np.testing.assert_allclose(one.stress, mises.stress, atol=1e-7)
    np.testing.assert_allclose(one.plastic, mises.plastic, atol=1e-12)


def test_lemaitre_damage_of_a_bar_that_hardens():
    """Linear hardening ``R0 + H p``: the damage is ``[(R0 + H p)^(2s+1) - (R0 + H
    p_D)^(2s+1)] / (H (2s + 1) (2 E S)^s)``, to the order of the steps."""
    hardening, strength, s, threshold = 2000.0, 0.6, 1.0, 0.005
    m = steel({"law": "lemaitre", "S": strength, "s": s, "threshold": threshold},
              hardening=hardening)  # fmt: skip
    errors = []
    for increments in (100, 400):
        r = simulate_point(m, tension(0.15, increments), tangent="full")
        e = r.strain[:, 0]
        p = np.maximum((e - R0 / E) / (1 + hardening / E), 0.0)

        def integral(q):
            return (R0 + hardening * q) ** (2 * s + 1)

        exact = np.where(p > threshold, (integral(p) - integral(threshold))
                         / (hardening * (2 * s + 1) * (2 * E * strength) ** s), 0.0)  # fmt: skip
        errors.append(np.abs(r.plastic - exact).max())
        assert r.plastic[-1] == pytest.approx(exact[-1], rel=0.02) and exact[-1] > 0.1
        effective = R0 + hardening * p
        on = p > 0
        np.testing.assert_allclose(r.stress[on, 0], ((1 - r.plastic) * effective)[on], rtol=1e-8)
    assert errors[1] < 0.3 * errors[0]  # first order in the step


def test_lemaitre_triaxiality_and_closure():
    """The energy release rate grows with the triaxiality by ``R_ν = 2/3 (1 + ν) + 3 (1 -
    2ν) η²``: a plastic flow under a mean tension damages more than the same in shear;
    with the closure ``h`` of the microcracks, a compression damages ``h`` times as a
    tension."""
    law = DuctileDamage("lemaitre", S=2.0)
    shear = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, R0 / np.sqrt(3.0)]])  # von Mises' stress R0
    pull = np.array([[R0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    for stress, eta in ((shear, 0.0), (pull, 1.0 / 3.0), (pull + 2 * R0 * np.eye(6)[:1] * 0
                                                           + R0 * np.array([[1, 1, 1, 0, 0, 0]]),
                                                           4.0 / 3.0)):  # fmt: skip
        y, _ = law.release_rate(stress, E, NU)
        r_nu = 2 / 3 * (1 + NU) + 3 * (1 - 2 * NU) * eta**2
        assert y[0] == pytest.approx(R0**2 * r_nu / (2 * E), rel=1e-12)
    closed = DuctileDamage("lemaitre", S=2.0, closure=0.2)
    assert closed.release_rate(-pull, E, NU)[0][0] == pytest.approx(
        0.2 * law.release_rate(pull, E, NU)[0][0], rel=1e-12
    )
    assert closed.release_rate(pull, E, NU)[0][0] == law.release_rate(pull, E, NU)[0][0]
    # a bar compressed: a fifth of the damage of the bar pulled
    m = steel({"law": "lemaitre", "S": 2.0, "closure": 0.2})
    pulled = simulate_point(m, tension(0.2), tangent="full").plastic[-1]
    pushed = simulate_point(m, tension(-0.2), tangent="full").plastic[-1]
    assert pushed == pytest.approx(0.2 * pulled, rel=1e-9)
    # held on its sides (uniaxial strain), the bar damages more than free: a mean tension
    held = simulate_point(m, {"times": [0, 1.0], "increments": 100,
                              "strain": {**FULL, "11": [0, 0.02]}}, tangent="full")  # fmt: skip
    free = simulate_point(m, tension(0.02), tangent="full")
    assert held.plastic[-1] > 2.0 * free.plastic[-1] > 0


def test_lemaitre_damage_of_a_few_cycles():
    """Cycles of a strain range, the hardening kinematic and linear: each cycle adds the
    damage of its plastic strain, ``N_R = D_c / (2 Δε_p (Y / S)^s)`` cycles to a damage
    ``D_c`` for a perfectly plastic bar (Manson and Coffin's law with the exponent -1)."""
    strength, amplitude, cycles = 0.25, 0.006, 5
    m = steel({"law": "lemaitre", "S": strength})
    values = [0.0] + [amplitude * (-1) ** k for k in range(2 * cycles + 1)]
    path = {"times": list(range(len(values))), "increments": 40, "strain": {"11": values}}
    r = simulate_point(m, path, tangent="full")
    d = r.plastic[::40]
    rate = R0**2 / (2 * E * strength)
    plastic_range = 2 * amplitude - 2 * R0 / E
    per_reversal = np.diff(d)[2:]  # once the loop is closed
    np.testing.assert_allclose(per_reversal, rate * plastic_range, rtol=0.03)
    cycles_to = 0.5 / (2 * plastic_range * rate)  # to a damage of 0.5
    assert d[-1] == pytest.approx(0.5 * (cycles + 0.25) / cycles_to, rel=0.06)


def test_a_point_breaks_at_its_critical_damage():
    m = steel({"law": "lemaitre", "S": 0.5, "critical": 0.3, "residual": 1e-3})
    r = simulate_point(m, tension(0.5, 200), tangent="full")
    d = r.plastic
    broken = np.flatnonzero(d > 0.3)
    assert broken.size and d[broken[0] - 1] <= 0.3
    np.testing.assert_allclose(d[broken], 1 - 1e-3)
    np.testing.assert_allclose(r.stress[broken, 0], 1e-3 * R0, rtol=1e-6)
    assert m.damage.top == 0.3 and DuctileDamage("lemaitre", S=1.0).top == 1 - 1e-6


def test_delayed_damage_of_a_plastic_bar():
    """With a viscosity the damage that degrades the stress follows that of the law with a
    delay: less of it at a faster rate, the same once the bar is held; the law itself does
    not depend on the delay."""
    law = {"law": "lemaitre", "S": 0.5}
    path = {"times": [0, 1.0, 400.0], "increments": [40, 40], "strain": {"11": [0, 0.2, 0.2]}}
    plain = simulate_point(steel(law), path, tangent="full")
    slow = simulate_point(steel({**law, "viscosity": 1e-3}), path, tangent="full")
    fast = simulate_point(steel({**law, "viscosity": 20.0}), path, tangent="full")
    d = plain.plastic[40]
    assert 0.05 < d < 0.9 and np.allclose(plain.plastic[40:], d)
    assert slow.plastic[40] == pytest.approx(d, rel=2e-2)
    assert fast.plastic[40] < 0.1 * d and fast.stress[40, 0] > plain.stress[40, 0]
    assert fast.plastic[-1] == pytest.approx(d, rel=1e-6)  # the bar held: the damage of the law
    assert steel({**law, "viscosity": 20.0}).damage.to_config()["viscosity"] == 20.0
    with pytest.raises(ValueError, match="viscosity"):
        steel({**law, "viscosity": -1.0})


# -- Kachanov --------------------------------------------------------------------------------


def creeping(**damage):
    return Material(young=150e3, poisson=0.3,
                    plasticity={"yield": 0.0, "norton": {"K": 800.0, "n": 5.0}},
                    damage={"law": "kachanov", "A": 500.0, "r": 4.0, **damage})  # fmt: skip


def test_kachanov_creep_to_rupture():
    """A constant uniaxial stress: the time to rupture ``1 / ((k + 1) (σ / A)^r)``, the
    damage ``1 - (1 - t / t_R)^(1 / (k + 1))`` and the tertiary creep of the effective
    stress, ``ε = σ / ((1 - D) E) + (σ / K)^N t_R / (1 - q) (1 - (1 - t / t_R)^(1 - q))``
    with ``q = N / (k + 1)``, to the order of the steps."""
    stress, k = 200.0, 6.0
    m = creeping(k=k)
    t_r = m.damage.rupture_time(stress)
    assert t_r == pytest.approx(1 / ((k + 1) * (stress / 500.0) ** 4.0))
    errors = []
    for increments in (100, 400):
        path = {"times": [0, 1e-6 * t_r, 0.9 * t_r], "increments": [1, increments],
                "stress": {"11": [0, stress, stress]}}  # fmt: skip
        r = simulate_point(m, path, tangent="full")
        t, d = r.times[1:], r.plastic[1:]
        exact = 1 - (1 - t / t_r) ** (1 / (k + 1))
        q = 5.0 / (k + 1)
        creep = (stress / 800.0) ** 5.0 * t_r / (1 - q) * (1 - (1 - t / t_r) ** (1 - q))
        strain = stress / ((1 - exact) * 150e3) + creep
        errors.append(np.abs(d - exact).max())
        assert d[-1] == pytest.approx(exact[-1], rel=0.05)
        np.testing.assert_allclose(r.strain[1:, 0], strain, rtol=0.04)
        assert np.all(np.diff(np.diff(r.strain[2:, 0])) > 0)  # the creep speeds up
    assert errors[1] < 0.3 * errors[0]
    # by default the exponent of 1 - D is that of the stress
    assert creeping().damage.k == 4.0 and creeping().damage.rupture_time(stress) == (
        pytest.approx(1 / (5.0 * 0.4**4))
    )
    assert creeping(critical=0.4).damage.rupture_time(stress) == pytest.approx(
        (1 - 0.6**5) / (5.0 * 0.4**4)
    )
    # with Hill's criterion the bar creeps at another rate, and breaks at the same time:
    # the damage is that of the stress
    hill = Material(young=150e3, poisson=0.3,
                    plasticity={"yield": 0.0, "norton": {"K": 800.0, "n": 5.0}, "hill": HILL},
                    damage={"law": "kachanov", "A": 500.0, "r": 4.0, "k": k})  # fmt: skip
    path = {"times": [0, 1e-6 * t_r, 0.9 * t_r], "increments": [1, 400],
            "stress": {"11": [0, stress, stress]}}  # fmt: skip
    r = simulate_point(hill, path, tangent="full")
    assert r.plastic[-1] == pytest.approx(1 - 0.1 ** (1 / (k + 1)), rel=0.02)
    assert r.strain[-1, 0] != pytest.approx(strain[-1], rel=0.05)


def test_hayhurst_stress():
    """``χ = α σ_I + 3 β σ_m + (1 - α - β) σ_eq``: the same in a uniaxial tension whatever
    the weights; in a shear, von Mises' stress times ``α / sqrt(3) + (1 - α - β)``."""
    pull = np.array([[300.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    shear = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 100.0]])
    for alpha, beta in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.3, 0.2)):
        law = DuctileDamage("kachanov", A=500.0, r=4.0, alpha=alpha, beta=beta)
        assert law.creep_stress(pull)[0][0] == pytest.approx(300.0)
        twisted = alpha * 100.0 + (1 - alpha - beta) * np.sqrt(3.0) * 100.0
        assert law.creep_stress(shear)[0][0] == pytest.approx(twisted)
        pushed = max(300.0 * (1 - alpha - 2 * beta), 0.0)  # no principal tension
        assert law.creep_stress(-pull)[0][0] == pytest.approx(pushed)
    # a hydrostatic compression does not damage; a tension does, by its mean stress
    law = DuctileDamage("kachanov", A=500.0, r=4.0, beta=0.5)
    press = np.array([[-100.0, -100.0, -100.0, 0.0, 0.0, 0.0]])
    assert law.creep_stress(press)[0][0] == 0.0 and law.creep_stress(-press)[0][0] == 150.0


# -- Mazars -------------------------------------------------------------------------------------

CONCRETE = {"law": "mazars", "threshold": 1e-4, "tension": [1.0, 1.5e4],
            "compression": [1.2, 1500.0]}  # fmt: skip


def test_mazars_law_in_tension_and_in_compression():
    """A uniaxial tension sees the damage of tension alone, ``σ = E κ_0 (1 - A_t) + A_t E
    ε exp(-B_t (ε - κ_0))``; a compression that of compression, with the equivalent strain
    ``sqrt(2) ν |ε|`` of its lateral extensions: a strength some ten times larger."""
    m = Material(young=30e3, poisson=0.2, damage=CONCRETE)
    law = m.damage
    assert isinstance(law, Mazars) and law.strains(30e3)[0] == 1e-4
    peaks = {}
    for sign, end in ((1.0, 6e-4), (-1.0, 6e-3)):
        r = simulate_point(m, tension(sign * end, 200), tangent="full")
        e = r.strain[:, 0]
        np.testing.assert_allclose(r.stress[:, 0], law.uniaxial(e, 30e3, 0.2), rtol=1e-9,
                                   atol=1e-9)  # fmt: skip
        assert np.abs(r.stress[:, 1:]).max() < 1e-9
        peaks[sign] = np.abs(r.stress[:, 0]).max()
        kappa = np.where(e >= 0, e, -np.sqrt(2.0) * 0.2 * e)
        pair = law.tension if sign > 0 else law.compression
        exact = np.where(kappa > 1e-4, law.branch(kappa, pair)[0], 0.0)
        np.testing.assert_allclose(r.plastic, exact, atol=1e-12)
    assert peaks[1.0] == pytest.approx(3.0, rel=0.02)  # E κ_0: the strength in tension
    assert 8 < peaks[-1.0] / peaks[1.0] < 15
    # the tension with A_t = 1: σ = E ε exp(-B_t (ε - κ_0)) beyond the threshold
    e = np.linspace(1e-4, 5e-4, 9)
    np.testing.assert_allclose(law.uniaxial(e, 30e3, 0.2), 30e3 * e * np.exp(-1.5e4 * (e - 1e-4)))
    # irreversible: unloaded along the damaged secant
    cycle = {"times": [0, 1, 2], "increments": [40, 10], "strain": {"11": [0, 3e-4, 1e-4]}}
    r = simulate_point(m, cycle, tangent="full")
    assert np.allclose(r.plastic[40:], r.plastic[40])
    np.testing.assert_allclose(r.stress[40:, 0], (1 - r.plastic[40]) * 30e3 * r.strain[40:, 0])


def test_shares_of_tension_and_compression():
    """The share of the tension is 1 where every principal stress pulls, 0 where every
    one pushes, and between them in a shear (a tension and a compression): the damage of a
    shear is that of ``α_t^β d_t + α_c^β d_c``, less with ``β`` above 1."""
    m = Material(young=30e3, poisson=0.2, damage=CONCRETE)
    law = analytic_model(m, finite=False)
    w = np.array([[2e-4, 1e-4, 1e-4], [-3e-4, -1e-4, -1e-4], [3e-4, -3e-4, 0.0]])
    alpha, _ = law._shares(w)
    assert alpha[0] == pytest.approx(1.0) and alpha[1] == 0.0 and 0.0 < alpha[2] < 1.0
    shear = {"times": [0, 1.0], "increments": 50, "strain": {**FULL, "12": [0, 8e-4]}}
    plain = simulate_point(m, shear, tangent="full").plastic[-1]
    less = Material(young=30e3, poisson=0.2, damage={**CONCRETE, "beta": 1.06})
    assert 0 < simulate_point(less, shear, tangent="full").plastic[-1] < plain


# -- the unilateral effect ---------------------------------------------------------------------


def test_spectral_split_keeps_the_stiffness_of_closed_cracks():
    """A bar damaged in tension, then compressed, held on its sides (uniaxial strain: no
    positive strain): with the spectral split its stiffness in compression is that of the
    sound material; in tension again, the damaged one."""
    assert SPLITS == ("none", "volumetric", "spectral")
    young, nu = 30e3, 0.2
    oed = young * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    path = {"times": [0, 1, 2, 3], "increments": [40, 20, 20],
            "strain": {**FULL, "11": [0, 4e-4, -4e-4, 2e-4]}}  # fmt: skip
    results = {}
    for split in ("none", "spectral"):
        law = {"threshold": 1e-4, "ultimate": 8e-4, "softening": "linear", "split": split}
        m = Material(young=young, poisson=nu, damage=law)
        results[split] = simulate_point(m, path, tangent="full")
    plain, split = results["none"], results["spectral"]
    d = split.plastic[40]
    assert 0.5 < d < 1 and np.allclose(split.plastic[40:], d)  # no damage in compression
    e, s = split.strain[:, 0], split.stress[:, 0]
    pushed = e < -1e-12
    np.testing.assert_allclose(s[pushed], oed * e[pushed], rtol=1e-9)  # the sound stiffness
    pulled = (split.times > 1.0) & (e > 1e-12)
    np.testing.assert_allclose(s[pulled], (1 - d) * oed * e[pulled], rtol=1e-9)
    # without the split the compression is as soft as the tension
    np.testing.assert_allclose(plain.stress[60, 0], (1 - plain.plastic[60]) * oed * -4e-4)
    assert plain.plastic[60] >= plain.plastic[40]


# -- tangents ----------------------------------------------------------------------------------


def _tangent_error(law, e, old, dt, e_old, scale):
    _, c, new = law.step(e, e_old, old, dt)
    h = 1e-8 * np.abs(e).max()
    worst = 0.0
    for k in range(6):
        d = np.zeros(6)
        d[k] = h
        fd = (law.step(e + d, e_old, old, dt)[0] - law.step(e - d, e_old, old, dt)[0]) / (2 * h)
        worst = max(worst, np.abs(c[:, :, k] - fd).max() / scale)
    return worst, new


def test_tangents_of_the_damage_laws():
    rng = np.random.default_rng(1)
    # Mazars: points pulled, pushed and sheared, their damage growing
    m = Material(young=30e3, poisson=0.2, damage=CONCRETE)
    e = 3e-4 * rng.standard_normal((12, 6))
    e[:4, :3] += 2e-4
    e[4:8, :3] -= 5e-4
    old = np.column_stack([np.full(12, 1.2e-4), np.full(12, 0.05)])
    error, new = _tangent_error(analytic_model(m, finite=False), e, old, 1.0, e, 30e3)
    assert error < 1e-6 and np.all(new[:, 1] >= 0.05) and new[:, 1].max() > 0.2
    # Lemaitre: isotropic and kinematic hardenings, the closure of the microcracks
    m = steel({"law": "lemaitre", "S": 1.5, "s": 1.5, "threshold": 0.002, "closure": 0.3},
              hardening=1500.0, isotropic=[{"Q": 80.0, "b": 12.0}],
              kinematic=[{"C": 20000.0, "D": 150.0}])  # fmt: skip
    law = analytic_model(m, finite=False)
    zero = np.zeros((6, 6))
    e1 = 6e-3 * rng.standard_normal((6, 6))
    state = law.step(e1, zero, np.zeros((6, law.n_state)), 1.0)[2]
    e2 = e1 + 3e-3 * rng.standard_normal((6, 6))
    error, new = _tangent_error(law, e2, state, 1.0, e1, E)
    assert error < 1e-6 and np.all(new[:, -1] > state[:, -1]) and not law.symmetric
    # the same with Hill's criterion
    m = steel({"law": "lemaitre", "S": 1.5, "s": 1.5, "threshold": 0.002, "closure": 0.3},
              hardening=1500.0, isotropic=[{"Q": 80.0, "b": 12.0}],
              kinematic=[{"C": 20000.0, "D": 150.0}], hill=HILL)  # fmt: skip
    law = analytic_model(m, finite=False)
    state = law.step(e1, zero, np.zeros((6, law.n_state)), 1.0)[2]
    error, new = _tangent_error(law, e2, state, 1.0, e1, E)
    assert error < 1e-6 and np.all(new[:, -1] > state[:, -1])
    # Kachanov, with Hayhurst's stress
    for weights in ({}, {"alpha": 0.4, "beta": 0.2}):
        law = analytic_model(creeping(**weights), finite=False)
        e1 = 2e-3 * rng.standard_normal((6, 6))
        state = law.step(e1, zero, np.zeros((6, law.n_state)), 5.0)[2]
        e2 = e1 + 1e-3 * rng.standard_normal((6, 6))
        error, new = _tangent_error(law, e2, state, 20.0, e1, 150e3)
        assert error < 1e-6 and np.all(new[:, -1] >= state[:, -1])
        assert new[:, -1].max() > 0.01 and (weights or np.all(new[:, -1] > state[:, -1]))


# -- configurations ------------------------------------------------------------------------------


def test_damage_laws_of_configurations():
    assert DUCTILE_LAWS == ("lemaitre", "kachanov")
    plain = damage_from_config({"threshold": 1e-4, "ultimate": 1e-3})
    assert isinstance(plain, Damage) and damage_from_config(plain) is plain
    mazars = damage_from_config(CONCRETE)
    assert mazars == Mazars(1e-4, (1.0, 1.5e4), (1.2, 1500.0))
    assert damage_from_config(mazars.to_config()) == mazars
    for spec in (
        {"law": "lemaitre", "S": 2.0, "s": 1.5, "threshold": 0.01, "closure": 0.2,
         "critical": 0.4},
        {"law": "kachanov", "A": 500.0, "r": 4.0, "k": 6.0, "alpha": 0.2, "beta": 0.1,
         "residual": 1e-4},
    ):  # fmt: skip
        law = damage_from_config(spec)
        assert law.to_config() == spec and damage_from_config(law.to_config()) == law
    materials = materials_from_config({
        0: {"young": E, "poisson": NU, "plasticity": {"yield": R0},
            "damage": {"law": "lemaitre", "S": 2.0}},
        1: {"young": 30e3, "poisson": 0.2, "damage": CONCRETE},
    })  # fmt: skip
    assert materials[0].analytic and materials[0].plastic and materials[0].nonlinear
    assert Material.from_config(materials[0].to_config()) == materials[0]
    assert Material.from_config(materials[1].to_config()) == materials[1]
    plastic = {"young": E, "poisson": NU, "plasticity": {"yield": R0}}
    for spec, match in (
        ({"young": E, "poisson": NU, "damage": {"law": "lemaitre", "S": 2.0}}, "plastic material"),
        ({**plastic, "damage": {"law": "kachanov", "A": 500.0, "r": 4.0}}, "viscoplastic"),
        ({**plastic, "damage": {"threshold": 1e-4, "ultimate": 1e-3}}, "one behaviour"),
        ({**plastic, "damage": CONCRETE}, "one behaviour"),
        ({**plastic, "damage": {"law": "lemaitre"}}, "damage.S"),
        ({**plastic, "damage": {"law": "lemaitre", "S": 2.0, "A": 3.0}}, "law kachanov"),
        ({**plastic, "damage": {"law": "lemaitre", "S": 2.0, "closure": 1.5}}, "closure"),
        ({**plastic, "damage": {"law": "lemaitre", "S": 2.0, "critical": 1.0}}, "critical"),
        ({**plastic, "damage": {"law": "lemaitre", "S": 2.0, "length": 1.0}}, "unknown keys"),
        ({**plastic, "damage": {"law": "gurson"}}, "damage.law"),
        ({"young": E, "poisson": NU, "damage": {"law": "mazars", "threshold": 1e-4}},
         "Mazars' law needs"),
        ({"young": E, "poisson": NU, "damage": {**CONCRETE, "tension": [1.0]}}, "pair"),
        ({"young": E, "poisson": NU, "damage": {**CONCRETE, "beta": 0.5}}, "beta"),
        ({"young": E, "poisson": NU, "plasticity": {"yield": 0.0, "norton": {"K": 1.0, "n": 3.0}},
          "damage": {"law": "kachanov", "A": 500.0, "r": 4.0, "alpha": 0.8, "beta": 0.5}},
         "Hayhurst"),
    ):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            Material.from_config(spec)
    with pytest.raises(ValueError, match="of the law of Kachanov"):
        DuctileDamage("lemaitre", S=1.0).rupture_time(100.0)


# -- in the solvers ------------------------------------------------------------------------------


def test_composite_whose_matrix_is_damaged():
    """Stiff particles in a plastic matrix that Lemaitre's law damages, by the FFT solver in
    small and finite strains: softer than the same matrix without damage, the damage of
    the matrix as the variable p of its phase."""
    data = np.zeros((8, 8, 8), np.uint8)
    data[2:6, 2:6, 2:6] = 1
    image = VoxelImage(data, periodic=True)
    matrix = {"young": E, "poisson": NU, "plasticity": {"yield": R0, "hardening": 2000.0}}
    particle = {"young": 2 * E, "poisson": 0.25}
    path = {"pull": {"times": [0, 1.0], "increments": 8, "strain": {"11": [0, 0.03]}}}
    sound = homogenize(image, materials_from_config({0: matrix, 1: particle}), properties=[],
                       paths=path)["pull.periodic"]  # fmt: skip
    damaged = {**matrix, "damage": {"law": "lemaitre", "S": 0.3}}
    for kinematics in ("small", "finite"):
        materials = materials_from_config({0: damaged, 1: particle})
        res = homogenize(image, materials, properties=[], paths=path,
                         nonlinear={"kinematics": kinematics})["pull.periodic"]  # fmt: skip
        d = res.phases[0]["plastic"]
        assert 0.05 < d[-1] < 0.9 and np.all(np.diff(d) >= -1e-12)
        assert res.phases[1]["plastic"][-1] == 0.0
        if kinematics == "small":
            assert 0.3 * sound.stress[-1, 0] < res.stress[-1, 0] < 0.97 * sound.stress[-1, 0]
            np.testing.assert_allclose(res.stress[1, 0], sound.stress[1, 0], rtol=0.02)
