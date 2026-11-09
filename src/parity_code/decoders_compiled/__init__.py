"""
Compiled versions of the decoders in `parity_code.decoders`.

They use the matching graphs of the original decoders (same edges and weights; ISM numbers the
nodes of every symmetry's graph anew, so that it only gets that symmetry's part of a syndrome).
Without post-processing, they differ only in how the result is read out: instead of returning the
matched edges to Python and finding the encircled qubits there, the base-qubit flips of every edge
are attached to the edges as PyMatching fault IDs, so PyMatching returns the predicted logical
flips directly, for whole batches of shots.

With post-processing, the matched edges are post-processed in compiled code by connected components
in space-time (`ComponentPostProcessing`). This differs from the post-processing of the original
decoders in how the matched edges are grouped and assigned to time slices; the post-processing
within a time slice is the same.
"""

from .global_symmetry_matching import CompiledGlobalSymmetryMatchingDecoder
from .independent_symmetry_matching import CompiledIndependentSymmetryMatchingDecoder
from .post_processing import ComponentPostProcessing

__all__ = [
    "CompiledGlobalSymmetryMatchingDecoder",
    "CompiledIndependentSymmetryMatchingDecoder",
    "ComponentPostProcessing",
]
