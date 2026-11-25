"""morphoGen with other programs and libraries.

=======================  ===========================================================================
:mod:`.arrays`           voxel images as the arrays of JAX, PyTorch and CuPy without copies
                         (DLPack, buffers aligned on 64 bytes), diffuse-interface phase fields
                         and ensembles of realizations for spectral solvers of phase-field
                         equations (Exponax: Allen–Cahn, Cahn–Hilliard) and differentiable
                         pipelines
:mod:`.microgen`         the shapes of Microgen (3MAH) as microstructures (the model
                         ``microgen``), the phases of morphoGen as shapes of Microgen, the
                         tetrahedral meshes of both through PyVista
:mod:`.blender`          a microstructure as a scene of Blender (the surfaces of its phases, a
                         fluid domain of Mantaflow around them, the flow baked read back); the
                         particles of a phase as its rigid bodies, settled under gravity
:mod:`.alloys`           the phases of an alloy from a tool of alloy design (CALPHAD, KWN:
                         Leil-Mat, Thermo-Calc, pycalphad) as the microstructures of its scales,
                         their effective elasticity against the moduli it predicts
=======================  ===========================================================================
"""
