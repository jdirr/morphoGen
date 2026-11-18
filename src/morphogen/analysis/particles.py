"""The particles of a phase in a volume: its connected parts, those that touch split by
the watershed of their distance map, with their volumes, their centres and their number
per unit volume — measured in three dimensions, where the stereology of
:mod:`morphogen.analysis.stereology` recovers them from sections.

The distance of each voxel of the phase to the other phase is highest at the heart of
the particles and low in the necks where two of them touch or overlap. Its maxima of a
prominence at least ``prominence`` (the descent from a maximum to the saddle towards a
higher one: a neck, not the roughness of a surface) are the markers of the particles, and
the watershed of the opposite of the distance (:mod:`morphogen.kernels.watershed`: the
voxels conquered from the markers in decreasing order of their distance) shares the
phase between them: two overlapping balls are cut by the plane of their neck. Voxels are
26-connected, through the faces of a periodic image.

In an image that is not periodic, the particles cut by its frame are known in part only:
they are left out, and each of the others counts for ``Π n_i / Π (n_i - 1 - w_i)`` over
the axes (``n_i`` voxels of the image, ``w_i`` voxels of the particle: the inverse of the
fraction of its positions that keep it whole in the frame, Miles 1974 and Lantuéjoul
1980), so that the number per unit volume and the distribution of sizes are not biased
towards the small particles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from morphogen.analysis.basic import indicator
from morphogen.analysis.morphology import _distance, _periodic, _spacing


@dataclass
class Particles:
    """The particles of a phase: ``labels`` (0 outside the phase, else the number of the
    particle from 1), their ``volumes`` and ``centroids`` (n x 3, in the box), their
    ``weights`` (1; in an image that is not periodic, 0 for a particle cut by the frame
    and the factor of Miles and Lantuéjoul for the others) and the ``volume`` of the
    image."""

    labels: np.ndarray
    volumes: np.ndarray
    centroids: np.ndarray
    weights: np.ndarray
    volume: float

    @property
    def count(self) -> int:
        """The number of particles (those cut by the frame included)."""
        return len(self.volumes)

    @property
    def radii(self) -> np.ndarray:
        """The radii of the balls of the volumes of the particles."""
        return np.cbrt(0.75 * self.volumes / np.pi)

    def density(self, edges=None):
        """The number of particles per unit volume; in the classes of radii between
        ``edges``, an array."""
        if edges is None:
            return float(self.weights.sum() / self.volume)
        counts, _ = np.histogram(self.radii, bins=edges, weights=self.weights)
        return counts / self.volume

    def mean_radius(self) -> float:
        """Number mean of the radii of the particles."""
        return float(np.average(self.radii, weights=self.weights))

    def radius_cv(self) -> float:
        """Coefficient of variation of the radii of the particles."""
        mean = self.mean_radius()
        spread = np.average((self.radii - mean) ** 2, weights=self.weights)
        return float(np.sqrt(spread) / mean)


def _centroids(labels: np.ndarray, count: int, spacing, origin, periodic: bool) -> np.ndarray:
    """The centres of mass of the particles; through the faces of a periodic image, from
    the positions of their voxels nearest their first one."""
    index = np.nonzero(labels)
    which = labels[index] - 1
    position = np.stack(index, axis=1).astype(float)
    if periodic:
        _, first = np.unique(which, return_index=True)
        shape = np.asarray(labels.shape, dtype=float)
        reference = position[first][which]
        position = reference + (position - reference + 0.5 * shape) % shape - 0.5 * shape
    sums = np.stack([np.bincount(which, position[:, a], minlength=count) for a in range(3)], 1)
    centres = sums / np.bincount(which, minlength=count)[:, None]
    if periodic:
        centres %= np.asarray(labels.shape, dtype=float)
    return np.asarray(origin, dtype=float) + (centres + 0.5) * np.asarray(spacing, dtype=float)


def particle_sizes(
    image,
    phase: int = 1,
    *,
    separate: bool = True,
    prominence: float | None = None,
    min_radius: float = 0.0,
    periodic=None,
) -> Particles:
    """The particles of ``phase`` in the volume of an image (see the module): its
    26-connected parts and, with ``separate``, those that touch split by the watershed
    of their distance map from its maxima of prominence at least ``prominence`` (physical
    units; default half a voxel). The particles of equivalent radius below ``min_radius``
    are left out. On periodic images each particle counts once, whole across the faces;
    otherwise those cut by the frame have a zero weight and the others the factor of
    Miles and Lantuéjoul."""
    from morphogen.kernels.watershed import flood_volume, prominent_maxima_volume

    mask = np.ascontiguousarray(indicator(image, phase).astype(bool))
    h = _spacing(image)
    per = _periodic(image, periodic)
    origin = getattr(image, "origin", (0.0, 0.0, 0.0))
    volume = float(mask.size * np.prod(h))
    empty = Particles(np.zeros(mask.shape, np.int32), np.zeros(0), np.zeros((0, 3)),
                      np.zeros(0), volume)  # fmt: skip
    if not mask.any():
        return empty
    pad = 0
    if per:  # the distance through the faces, beyond the largest inscribed ball
        rough = ndimage.distance_transform_edt(mask, sampling=h)
        pad = min(int(np.ceil(rough.max() / float(h.min()))) + 2, max(mask.shape))
    distance = _distance(mask, h, per, pad)
    if separate:
        prominence = 0.5 * float(h.min()) if prominence is None else float(prominence)
    else:
        prominence = np.inf  # the highest point of each connected part alone
    peaks = prominent_maxima_volume(distance, mask, prominence, per)
    labels, count = ndimage.label(peaks, structure=np.ones((3, 3, 3), dtype=bool))
    labels = labels.astype(np.int32)
    labels[~mask] = -1
    flood_volume(-distance, labels, per)
    labels[~mask] = 0
    volumes = np.bincount(labels.ravel(), minlength=count + 1)[1:] * float(np.prod(h))
    keep = np.cbrt(0.75 * volumes / np.pi) >= float(min_radius)
    if not keep.all():
        renumber = np.zeros(count + 1, dtype=np.int32)
        renumber[1:][keep] = np.arange(1, int(keep.sum()) + 1)
        labels, volumes, count = renumber[labels], volumes[keep], int(keep.sum())
    if count == 0:
        return empty
    weights = np.ones(count)
    if not per:
        n = np.asarray(mask.shape)
        for k, box in enumerate(ndimage.find_objects(labels)):
            free = 1.0
            for axis in range(3):
                if n[axis] == 1:
                    continue  # a planar image: no frame across its plane
                low, high = box[axis].start, box[axis].stop
                if low == 0 or high == n[axis]:
                    free = 0.0  # cut by the frame
                    break
                free *= (n[axis] - 1 - (high - low)) / n[axis]
            weights[k] = 1.0 / free if free > 0.0 else 0.0
    centroids = _centroids(labels, count, h, origin, per)
    return Particles(labels, volumes, centroids, weights, volume)
