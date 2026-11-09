# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
import numpy as np
import pymatching
import stim

from parity_code.decoders import IndependentSymmetryMatchingDecoder
from parity_code.decoders_compiled.common import make_base_qubit_flips
from parity_code.decoders_compiled.post_processing import ComponentPostProcessing, concatenate_edges


class CompiledIndependentSymmetryMatchingDecoder:
    """
    Independent Symmetry Matching (ISM) in compiled code, with or without post-processing.

    The per-symmetry matching graphs have the same edges and weights as those of
    `parity_code.decoders.IndependentSymmetryMatchingDecoder`. Their nodes are the detectors of the
    symmetry only, numbered 0, 1, ... in the order of their IDs in the DEM, so that PyMatching only
    gets the symmetry's own part of a syndrome.

    Without post-processing, the logical flips are computed inside PyMatching: every edge, including
    the boundary edges, carries as PyMatching fault IDs the base qubits that the original ray
    casting flips for this edge (see `make_base_qubit_flips`; a boundary edge flips what the edge to
    its virtual stabilizer flips). The predicted logical flips are the XOR of the d per-symmetry
    predictions. The matching along the virtual symmetry is skipped: its edges join two virtual
    stabilizers or are time-like and flip nothing, so it cannot change the logical flips.

    With post-processing, the matched edges of every shot are obtained as in the original decoder:
    from the per-symmetry matchings (`decode_to_edges_array`), with the boundary matches
    replaced by edges to the corresponding virtual stabilizers, and from the matching along the
    virtual symmetry, whose detection events are the virtual stabilizers reached an odd number of
    times. They are post-processed in compiled code by connected components in space-time, see
    `ComponentPostProcessing`. This differs from the original post-processing only in how the
    matched edges are grouped and assigned to time slices.

    Args:
        dem, data_qubit_coords, measurement_qubits_of_symmetry, boundary_edges_of_real_symmetry,
        one_lines: As for `parity_code.decoders.IndependentSymmetryMatchingDecoder`.

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
        ism = IndependentSymmetryMatchingDecoder(
            dem,
            data_qubit_coords,
            measurement_qubits_of_symmetry,
            boundary_edges_of_real_symmetry,
            one_lines,
        )
        flips = make_base_qubit_flips(ism, base_qubit_coords)
        self.num_observables = len(base_qubit_coords)
        self.num_detectors = len(ism.detector_coords)
        # for post-processing, the ends of the matched edges are space-time points
        self.component_post_processing = ComponentPostProcessing(ism)
        point = self.component_post_processing.point

        # One matching per real symmetry with the same edges and weights as the original ISM, plus
        # fault IDs, on the detectors of the symmetry only: node i is detector detectors[i] of the
        # DEM. It gives the logical flips without post-processing and the matched edges with
        # post-processing. For post-processing also, per real symmetry: the space-time point of
        # every node, and for every node at the boundary of the symmetry the point of the virtual
        # stabilizer it is matched to via the boundary (-1 for the other nodes).
        self.matchings_per_symmetry: dict[int, tuple[pymatching.Matching, np.ndarray]] = {}
        self._points_per_symmetry: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for s, original in ism.matchings_per_symmetry.items():
            detectors = np.array(sorted(ism.detectors_of_real_symmetries[s]), dtype=np.int64)
            node = {det: i for i, det in enumerate(detectors.tolist())}
            matching = pymatching.Matching()
            for a, b, info in original.edges():
                coord_a = ism.detector_coords[a]
                if b is None:
                    # boundary edge: flips what the edge to its virtual stabilizer flips
                    fault_ids = flips(coord_a, ism.get_matched_virtual_det_coord(s, coord_a))
                    matching.add_boundary_edge(
                        node[a], fault_ids=fault_ids, weight=info["weight"], merge_strategy="disallow"
                    )
                else:
                    fault_ids = flips(coord_a, ism.detector_coords[b])
                    matching.add_edge(
                        node[a],
                        node[b],
                        fault_ids=fault_ids,
                        weight=info["weight"],
                        merge_strategy="disallow",
                    )
            matching.ensure_num_fault_ids(self.num_observables)
            # the detectors after the last one with an edge can never be flipped
            detectors = detectors[: matching.num_detectors]
            self.matchings_per_symmetry[s] = (matching, detectors)

            coords = [ism.detector_coords[det] for det in detectors.tolist()]
            point_of_node = np.array([point(coord) for coord in coords], dtype=np.int32)
            boundary_point = np.full(len(detectors), -1, dtype=np.int32)
            for i, (x, y, t) in enumerate(coords):
                if (x, y) in ism.boundary_edges_of_symmetry[s]:
                    boundary_point[i] = point(ism.get_matched_virtual_det_coord(s, (x, y, t)))
            self._points_per_symmetry[s] = (point_of_node, boundary_point)

        self._virtual_matching = ism.virtual_matching
        num_virtual_nodes = len(ism.virtual_idx_to_node)
        self._point_of_virtual_node = np.array(
            [point(ism.virtual_idx_to_node[i]) for i in range(num_virtual_nodes)], dtype=np.int32
        )
        self._virtual_node_of_point = np.full(
            self.component_post_processing.num_points, -1, dtype=np.int64
        )
        self._virtual_node_of_point[self._point_of_virtual_node] = np.arange(num_virtual_nodes)
        self._base_data_qubit_indices = [data_qubit_coords.index(c) for c in base_qubit_coords]

    def _correction_masks(
        self, syndromes: np.ndarray, chunk_size: int, post_processing: bool
    ) -> np.ndarray:
        """Bit masks (shots, num_words) of the corrections, with or without post-processing."""
        num_words = self.component_post_processing.num_words
        masks = np.zeros((len(syndromes), num_words), dtype=np.uint64)
        for start in range(0, len(syndromes), chunk_size):
            chunk = syndromes[start : start + chunk_size]
            num_shots = len(chunk)
            # the matched edges of every matching, grouped by shot, and the offsets of the shots in
            # the concatenation of all of them
            points_a, points_b, shot_offsets = [], [], []
            num_edges = 0
            virtual_syndromes = np.zeros((num_shots, len(self._point_of_virtual_node)), np.uint8)

            for s, (matching, detectors) in self.matchings_per_symmetry.items():
                point_of_node, boundary_point = self._points_per_symmetry[s]
                edges, offsets = concatenate_edges(
                    [matching.decode_to_edges_array(x) for x in chunk[:, detectors]]
                )
                to_boundary = edges[:, 1] < 0
                point_a = point_of_node[edges[:, 0]]
                point_b = point_of_node[edges[:, 1]]
                point_b[to_boundary] = boundary_point[edges[to_boundary, 0]]
                if (point_b < 0).any():
                    raise RuntimeError(
                        "A detector that is not at the boundary was matched to the boundary."
                    )
                # the ends of the boundary matches are the detection events of the virtual matching
                shot_of_edge = np.repeat(np.arange(num_shots), np.diff(offsets))
                np.bitwise_xor.at(
                    virtual_syndromes,
                    (shot_of_edge[to_boundary], self._virtual_node_of_point[point_b[to_boundary]]),
                    1,
                )
                points_a.append(point_a)
                points_b.append(point_b)
                shot_offsets.append(num_edges + offsets)
                num_edges += len(edges)

            # matching of the virtual stabilizers along the virtual symmetry
            edges, offsets = concatenate_edges(
                [self._virtual_matching.decode_to_edges_array(x) for x in virtual_syndromes]
            )
            points_a.append(self._point_of_virtual_node[edges[:, 0]])
            points_b.append(self._point_of_virtual_node[edges[:, 1]])
            shot_offsets.append(num_edges + offsets)

            # the edges of a shot are spread over the d + 1 matchings; the post-processing does not
            # depend on their order, so they are passed as d + 1 segments without sorting
            masks[start : start + num_shots] = self.component_post_processing(
                np.concatenate(points_a),
                np.concatenate(points_b),
                np.stack(shot_offsets),
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
            correction returned by `IndependentSymmetryMatchingDecoder.decode(syndrome,
            post_processing=False)`.
        """
        syndromes = np.asarray(syndromes, dtype=bool)
        if post_processing:
            corrections = self.decode_batch_to_corrections(syndromes, True, chunk_size)
            return corrections[:, self._base_data_qubit_indices]

        predictions = np.zeros((len(syndromes), self.num_observables), dtype=np.uint8)
        for start in range(0, len(syndromes), chunk_size):
            chunk = syndromes[start : start + chunk_size]
            for matching, detectors in self.matchings_per_symmetry.values():
                chunk_s = chunk[:, detectors].view(np.uint8)
                predictions[start : start + len(chunk)] ^= matching.decode_batch(chunk_s)
        return predictions.astype(bool)

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
