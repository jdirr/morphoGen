"""An alloy from a tool of alloy design to its microstructure and its elasticity.

The phases of an alloy (from CALPHAD equilibria, Scheil solidification, the KWN kinetics
of its precipitates: Leil-Mat, Thermo-Calc, pycalphad...) are written as a description
(morphogen.interop.alloys), turned into the microstructures of its two scales — the
precipitates in the matrix, the grains with their particles — and homogenized by FFT; the
effective Young's modulus is compared with the one the tool predicted from the phases.
The values below are illustrative, not data of a real alloy.
"""

from morphogen.interop import alloys

alloy = alloys.read_alloy(
    {
        "alloy": "AA7075-T6 (illustrative)",
        "source": "values typed from the canvas of a tool of alloy design",
        "composition": {"Al": 90.0, "Zn": 5.6, "Mg": 2.5, "Cu": 1.6, "Cr": 0.23},
        "basis": "wt%",
        "temperature": 393.15,  # K: the ageing
        "unit": "um",
        "stress_unit": "GPa",
        "grain_size": 50.0,
        "phases": [
            {"name": "FCC_A1", "role": "matrix", "young": 70.0, "poisson": 0.33},
            {"name": "MgZn2 (eta')", "role": "precipitates", "fraction": 0.05,
             "radius": {"mean": 0.003, "std": 0.001}, "young": 77.0, "poisson": 0.3},
            {"name": "Al7Cu2Fe", "role": "particles", "fraction": 0.02, "radius": 5.0,
             "young": 150.0, "poisson": 0.3},
        ],
        "properties": {"young": 71.7},  # the tool's estimate (a mixture of the phases)
    }
)  # fmt: skip
alloys.write_alloy("alloy_7075.json", alloy)  # the description, for morphogen alloy

for scale in alloys.alloy_scales(alloy):
    print(f"{scale['name']}: a box of {scale['domain']['size']:.4g} {alloy.unit}")

result = alloys.homogenize_alloy(alloy, shape=32)
for name, entry in result["scales"].items():
    print(f"{name}: E = {entry['young']:.2f} GPa, nu = {entry['poisson']:.3f}")
print(f"predicted {result['predicted_young']:.2f} GPa: {100 * result['departure']:+.1f} %")
