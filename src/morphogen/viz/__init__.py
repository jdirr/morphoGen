"""Visualization: images without dependencies, figures with matplotlib (``[plot]``).

* :mod:`morphogen.viz.png`: colour maps, sections and PNG encoding (standard library);
* :mod:`morphogen.viz.figures`: sections, periodic tilings, 3D surfaces, histories,
  transformation kinetics, animated GIFs of the formation of structures, the beams of
  frames coloured by their forces or energies and the replicas of networks of
  non-identical beams.
"""

from morphogen.viz.png import colormap, decode_png, encode_png, slice_of, to_rgb, upscale

__all__ = ["colormap", "decode_png", "encode_png", "slice_of", "to_rgb", "upscale"]
