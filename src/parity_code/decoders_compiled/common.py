# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
from collections.abc import Callable

from parity_code.decoders.base import SymmetryMatchingDecoderBase
from parity_code.decoders.common import Node


def make_base_qubit_flips(
    decoder: SymmetryMatchingDecoderBase, base_qubit_coords: list[tuple[int, int]]
) -> Callable[[Node, Node], set[int]]:
    """
    Returns a function that maps a single edge of the matching graph, given by the coordinates
    (x, y, t) of its two nodes, to the indices of the base qubits (= logical observables) that this
    edge flips.

    The repo's decoders find the correction by ray casting: the correction of a union of closed
    loops is the XOR of fixed per-edge sets of data qubits
    (`SymmetryMatchingDecoderBase.flipped_data_qubits_of_edge`). PyMatching can XOR such per-edge
    sets ("fault IDs") along the edges of the matching solution itself, which is what the compiled
    decoders without post-processing use; this function gives the base-qubit part of the sets.

    Args:
        decoder: A decoder of `parity_code.decoders`, whose geometry (symmetries and ray casting)
            defines the flips.
        base_qubit_coords: Coordinates of the base qubits; base qubit i is logical observable i.
    """
    base_index_of_data_qubit = {
        decoder.data_qubit_coords.index(coord): i for i, coord in enumerate(base_qubit_coords)
    }

    def flips(coord_a: Node, coord_b: Node) -> set[int]:
        return {
            base_index_of_data_qubit[q]
            for q in decoder.flipped_data_qubits_of_edge(coord_a[:2], coord_b[:2])
            if q in base_index_of_data_qubit
        }

    return flips
