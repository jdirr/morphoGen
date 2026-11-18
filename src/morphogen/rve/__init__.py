"""Statistical size of the representative volume element (RVE).

* :mod:`morphogen.rve.scaling`: point variance, ensemble statistics, scaling law
  ``D_Z^2(V) = K V^-gamma``, RVE volume and number of realizations, window method.
* :mod:`morphogen.rve.study`: analysis of several properties and boundary conditions
  following Dirrenberger et al. (2014): intervals, relative errors, fitted and imposed
  exponents, RVE sizes with bootstrap intervals, isotropy, discrepancy between
  boundary conditions; the mean stiffness and compliance per volume of each boundary
  condition (:class:`TensorStatistics`: ``Cbar``, ``S = Cbar^-1``, the standard deviation
  of each component, their RVE sizes).
* :mod:`morphogen.rve.windows`: the window method for tensors, the stiffness and the
  conductivity of the windows of one periodic realization solved once per unit load by the
  FFT solver (:class:`WindowTensors`: their mean mutual energies), and their statistics.
* :mod:`morphogen.rve.report`: Markdown, LaTeX and JSON reports, figures, the table of the
  mean tensors per volume (``rve_stiffness.csv``).
"""

from morphogen.rve.scaling import (
    ScalingLaw,
    SizeStatistics,
    ensemble_statistics,
    fit_scaling_law,
    fit_statistics,
    point_variance,
    point_variance_multiphase,
    rve_table,
    rve_volume_classic,
    window_statistics,
)
from morphogen.rve.study import (
    Series,
    SeriesAnalysis,
    StudyOptions,
    StudyReport,
    TensorStatistics,
    analyze_series,
    analyze_statistics,
    analyze_study,
    discrepancy,
    is_tensor,
    series_from_rows,
    tensor_statistics,
    tensors_from_rows,
)
from morphogen.rve.windows import WindowTensors, window_tensors

__all__ = [
    "ScalingLaw",
    "Series",
    "SeriesAnalysis",
    "SizeStatistics",
    "StudyOptions",
    "StudyReport",
    "TensorStatistics",
    "WindowTensors",
    "analyze_series",
    "analyze_statistics",
    "analyze_study",
    "discrepancy",
    "ensemble_statistics",
    "fit_scaling_law",
    "fit_statistics",
    "is_tensor",
    "point_variance",
    "point_variance_multiphase",
    "rve_table",
    "rve_volume_classic",
    "series_from_rows",
    "tensor_statistics",
    "tensors_from_rows",
    "window_statistics",
    "window_tensors",
]
