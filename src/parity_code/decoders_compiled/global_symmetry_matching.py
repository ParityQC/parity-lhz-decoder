# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
import numpy as np
import pymatching
import stim

from parity_code.decoders import GlobalSymmetryMatchingDecoder
from parity_code.decoders_compiled.common import make_base_qubit_flips
from parity_code.decoders_compiled.post_processing import ComponentPostProcessing, concatenate_edges


class CompiledGlobalSymmetryMatchingDecoder:
    """
    Global Symmetry Matching (GSM) in compiled code, with or without post-processing.

    The matching graph is exactly the one of `parity_code.decoders.GlobalSymmetryMatchingDecoder`
    (same nodes, edges and weights; it is built by that class).

    Without post-processing, the logical flips are computed inside PyMatching: every edge carries as
    PyMatching fault IDs the base qubits that the original ray casting flips for this edge (see
    `make_base_qubit_flips`). PyMatching XORs the fault IDs along the edges of the minimum-weight
    solution, which gives the base-qubit part of the original correction, i.e. the predicted flip of
    every logical observable, for a whole batch of shots in compiled code. No matched edges,
    coordinates or encircled qubits are produced in Python.

    With post-processing, the matched edges of every shot are obtained from the original global
    matching graph (`decode_to_edges_array`, as in the original decoder) and post-processed in
    compiled code by connected components in space-time, see `ComponentPostProcessing`. This differs
    from the original post-processing only in how the matched edges are grouped and assigned to time
    slices.

    Args:
        dem, data_qubit_coords, measurement_qubits_of_symmetry, boundary_edges_of_real_symmetry,
        one_lines: As for `parity_code.decoders.GlobalSymmetryMatchingDecoder`.

        base_qubit_coords (list[tuple[int, int]]): Coordinates of the base qubits; base qubit i is
            logical observable i of the circuit.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        data_qubit_coords: list[tuple[int, int]],
        measurement_qubits_of_symmetry: dict[int, list[tuple[int, int]]],
        boundary_edges_of_real_symmetry: dict[int, dict[tuple, tuple]],
        one_lines: dict[int, list[tuple[int, int]]],
        base_qubit_coords: list[tuple[int, int]],
    ):
        gsm = GlobalSymmetryMatchingDecoder(
            dem,
            data_qubit_coords,
            measurement_qubits_of_symmetry,
            boundary_edges_of_real_symmetry,
            one_lines,
        )
        flips = make_base_qubit_flips(gsm, base_qubit_coords)
        self.num_observables = len(base_qubit_coords)
        self.num_detectors = len(gsm.detector_coords)

        # Same edges and weights as the original global matching graph, plus the base-qubit flips of
        # every edge as fault IDs. (Boundary edges were already replaced by edges to virtual nodes.)
        self.matching = pymatching.Matching()
        for a, b, info in gsm.global_matching.edges():
            self.matching.add_edge(
                a,
                b,
                fault_ids=flips(gsm.global_detector_coords[a], gsm.global_detector_coords[b]),
                weight=info["weight"],
                merge_strategy="disallow",
            )
        self.matching.ensure_num_fault_ids(self.num_observables)

        # every detector of the original DEM has a node in each of the two symmetries it lies on
        self._global_det_ids = gsm.det_id_to_global_det_ids

        # For post-processing: the original global matching graph gives the matched edges, and every
        # node of it (both copies of a detector, and the virtual nodes) is a space-time point.
        self._edge_matching = gsm.global_matching
        self.component_post_processing = ComponentPostProcessing(gsm)
        self._point_of_node = np.array(
            [
                self.component_post_processing.point(gsm.global_detector_coords[node])
                for node in range(self._edge_matching.num_detectors)
            ],
            dtype=np.int32,
        )
        self._base_data_qubit_indices = [data_qubit_coords.index(c) for c in base_qubit_coords]

    def _global_syndromes(self, syndromes: np.ndarray, num_nodes: int) -> np.ndarray:
        global_syndromes = np.zeros((len(syndromes), num_nodes), dtype=np.uint8)
        global_syndromes[:, self._global_det_ids[:, 0]] = syndromes
        global_syndromes[:, self._global_det_ids[:, 1]] = syndromes
        return global_syndromes

    def _correction_masks(
        self, syndromes: np.ndarray, chunk_size: int, post_processing: bool
    ) -> np.ndarray:
        """Bit masks (shots, num_words) of the corrections, with or without post-processing."""
        num_words = self.component_post_processing.num_words
        masks = np.zeros((len(syndromes), num_words), dtype=np.uint64)
        for start in range(0, len(syndromes), chunk_size):
            chunk = syndromes[start : start + chunk_size]
            global_syndromes = self._global_syndromes(chunk, self._edge_matching.num_detectors)
            edges, shot_offsets = concatenate_edges(
                [self._edge_matching.decode_to_edges_array(s) for s in global_syndromes]
            )
            if (edges < 0).any():
                raise RuntimeError("The global matching graph has no boundary edges.")
            masks[start : start + len(chunk)] = self.component_post_processing(
                self._point_of_node[edges[:, 0]],
                self._point_of_node[edges[:, 1]],
                shot_offsets,
                post_process=post_processing,
            )
        return masks

    def decode_batch(
        self, syndromes: np.ndarray, post_processing: bool = False, chunk_size: int = 1_000
    ) -> np.ndarray:
        """
        Decode a batch of syndromes.

        Args:
            syndromes: Boolean array of shape (shots, num_detectors), as returned by a stim detector
                sampler.
            post_processing: Whether to post-process the correction (see `ComponentPostProcessing`).
            chunk_size: Number of shots processed at once (bounds memory).

        Returns:
            Boolean array of shape (shots, num_observables) with the predicted flip of every logical
            observable. Without post-processing, it is identical to the base-qubit part of the
            correction returned by `GlobalSymmetryMatchingDecoder.decode(syndrome,
            post_processing=False)`.
        """
        syndromes = np.asarray(syndromes, dtype=bool)
        if post_processing:
            corrections = self.decode_batch_to_corrections(syndromes, True, chunk_size)
            return corrections[:, self._base_data_qubit_indices]

        predictions = np.zeros((len(syndromes), self.num_observables), dtype=bool)
        for start in range(0, len(syndromes), chunk_size):
            chunk = syndromes[start : start + chunk_size]
            global_syndromes = self._global_syndromes(chunk, self.matching.num_detectors)
            predictions[start : start + len(chunk)] = self.matching.decode_batch(global_syndromes)
        return predictions

    def decode(self, syndrome: np.ndarray, post_processing: bool = False) -> np.ndarray:
        """Decode a single syndrome; returns the predicted flip of every logical observable."""
        syndromes = np.asarray(syndrome, dtype=bool)[np.newaxis, :]
        return self.decode_batch(syndromes, post_processing=post_processing)[0]

    def decode_batch_to_corrections(
        self, syndromes: np.ndarray, post_processing: bool = True, chunk_size: int = 1_000
    ) -> np.ndarray:
        """
        Full corrections (all data qubits) of a batch of syndromes.

        Args:
            syndromes: Boolean array of shape (shots, num_detectors).
            post_processing: Whether to post-process the corrections (see
                `ComponentPostProcessing`). Without post-processing, the corrections are those of
                the original decoder without post-processing.
            chunk_size: Number of shots processed at once (bounds memory).

        Returns:
            Boolean array of shape (shots, num_data_qubits); entry (i, j) says whether data qubit j
            (in the order of `data_qubit_coords`) is flipped by the correction of shot i.
        """
        syndromes = np.asarray(syndromes, dtype=bool)
        masks = self._correction_masks(syndromes, chunk_size, post_processing)
        return self.component_post_processing.unpack(masks)
