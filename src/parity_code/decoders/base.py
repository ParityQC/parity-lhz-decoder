# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
import numpy as np
import stim
from encircled_nodes_finder import EncircledNodesFinder
from numpy.typing import NDArray

from parity_code.decoders.common import (
    Edge,
    Node,
    divide_into_clusters,
    remove_even_occurrences,
    remove_pure_measurement_errors,
)
from parity_code.decoders.common import (
    project_cluster_to_majority_voted_plane as _project_majority,
)
from parity_code.decoders.common import (
    project_cluster_to_plane as _project_plane,
)


class SymmetryMatchingDecoderBase:
    """
    Shared machinery for decoding the parity code by matching along the symmetries of the code (see
    https://arxiv.org/pdf/2505.05210): building the per-symmetry detector error models, finding the
    data qubits enclosed by a set of matched detectors, and post-processing corrections against the
    code's 1-lines.

    Subclasses differ only in how they obtain `matches` from a syndrome (`_match`): the Independent
    Symmetry Matching (ISM) decoder matches each symmetry independently, while the Global Symmetry
    Matching (GSM) decoder matches all symmetries together in a single matching graph.

    Args:
        dem (stim.DetectorErrorModel): The full detector error model including hyperedges.

        data_qubit_coords (list[tuple[int, int]]):
            A list of data qubit coordinates of the form [(x,y), ...].

        measurement_qubits_of_symmetry (dict): A dictionary where the keys are symmetry indices
            (0 - num_symmetries) and the values are lists of measurement qubit coordinates of the
            form [(x,y), ...]. Including virtual measurement qubits, where 0 indicates the virtual
            symmetry.
            Note: the order of measurement qubit coordinates along each symmetry should be defined
                in a way that the first and last qubit are at the boundary of the symmetry and the
                rest are in order along the symmetry.

        boundary_edges_of_real_symmetry (dict): A dictionary where the keys are real symmetry
            indices (1 - num_symmetries) and the values are dictionaries giving for each real
            measurement qubit at a boundary the corresponding virtual measurement qubit.
            The dictionary is of the form
                {symmetry_index: {real_measurement_qubit: virtual_measurement_qubit}}.

        one_lines (dict): A dictionary defining the one lines of the parity code, see definition of
            k-lines in the paper: https://arxiv.org/pdf/2505.05210
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        data_qubit_coords: list[tuple[int, int]],
        measurement_qubits_of_symmetry: dict[int, list[tuple[int, int]]],
        boundary_edges_of_real_symmetry: dict[int, dict[tuple, tuple]],
        one_lines: dict[int, list[tuple[int, int]]],
    ):
        self.dem = dem
        self.data_qubit_coords = data_qubit_coords
        self._data_qubit_coords_np = np.array(data_qubit_coords, dtype=np.int32)

        self.measurement_qubits_of_symmetry = measurement_qubits_of_symmetry
        self.boundary_edges_of_symmetry = boundary_edges_of_real_symmetry
        self.one_lines = one_lines  # used for post-processing

        # detector coordinates of the original (unfiltered) DEM
        self.detector_coords = {
            d: (int(coord[0]), int(coord[1]), int(coord[2]))
            for d, coord in dem.get_detector_coordinates().items()
        }
        self.detectors_of_real_symmetries = self._get_detectors_of_real_symmetries()

        # Precompute a dictionary for finding the flipped qubits from edges found in the matching
        self._flipped_data_qubit_indices_of_edge = {
            edge: np.nonzero(mask)[0].astype(np.intp)  # make sure it's native int type
            for edge, mask in self._get_flipped_data_qubits_of_edge().items()
        }
        self._encircled_nodes_finder = EncircledNodesFinder(
            self._data_qubit_coords_np, self._flipped_data_qubit_indices_of_edge
        )

    def _get_detectors_of_real_symmetries(self) -> dict[int, list[int]]:
        detectors_of_symmetry = {}
        for s, qbts in self.measurement_qubits_of_symmetry.items():
            if s == 0:
                continue
            detectors_of_symmetry[s] = []
            for d, coord in self.detector_coords.items():
                x, y, t = coord
                if (x, y) in qbts:
                    detectors_of_symmetry[s].append(d)
        return detectors_of_symmetry

    def _get_flipped_data_qubits_of_edge(
        self,
    ) -> dict[tuple[tuple[int, int], tuple[int, int]], NDArray[np.bool_]]:
        """Determine which data qubits each edge that could be found in the matching flips. The
        ray casting finds the qubits in the interior of closed loops of matched edges, i.e. the
        qubits that have to be corrected, as the XOR of the flips of their edges.

        A horizontal ray from a data qubit to the right crosses an odd number of edges of closed
        loops if and only if the qubit is enclosed. An edge between consecutive stabilizers of a
        symmetry on different rows therefore flips the data qubits on the row it crosses, up to and
        including its crossing point: a qubit that a diagonal edge passes through lies on the
        boundary of the lattice and counts as inside it. Horizontal and time-like edges as well as
        edges between two virtual stabilizers (which lie to the left of and above the data qubits)
        flip nothing.

        Returns:
            flipped_data_qubits_of_edge: maps an edge (tuple of 2 points, in both orders) to a
                boolean mask of the data qubits flipped by it; edges that flip nothing are missing.
        """
        virtual_stabilizers = set(self.measurement_qubits_of_symmetry[0])
        flipped_data_qubits_of_edge: dict[
            tuple[tuple[int, int], tuple[int, int]], NDArray[np.bool_]
        ] = {}

        for path in self.measurement_qubits_of_symmetry.values():
            for p1, p2 in zip(path[:-1], path[1:]):
                if p1[1] == p2[1] or (p1 in virtual_stabilizers and p2 in virtual_stabilizers):
                    continue  # horizontal edge, or edge between two virtual stabilizers

                # all data qubits on the row of the edge midpoint, up to and including it
                x_mid = (p1[0] + p2[0]) / 2
                y_mid = (p1[1] + p2[1]) / 2
                mask = np.array(
                    [x <= x_mid and y == y_mid for x, y in self.data_qubit_coords], dtype=bool
                )
                flipped_data_qubits_of_edge[(p1, p2)] = mask
                flipped_data_qubits_of_edge[(p2, p1)] = mask  # symmetric

        return flipped_data_qubits_of_edge

    def flipped_data_qubits_of_edge(self, a: tuple[int, int], b: tuple[int, int]) -> set[int]:
        """Indices (into `data_qubit_coords`) of the data qubits flipped by an edge between the
        (real or virtual) stabilizers at a and b in the ray casting; empty for an edge that flips
        nothing, see `_get_flipped_data_qubits_of_edge`."""
        indices = self._flipped_data_qubit_indices_of_edge.get((a, b))
        return set() if indices is None else set(indices.tolist())

    def get_matched_virtual_det_coord(self, symmetry: int, real_det_coord: Node) -> Node:
        """Returns the corresponding virtual detector for a real detector at the boundary in the
        given symmetry."""
        a_x, a_y, a_t = real_det_coord
        try:
            b_x, b_y = self.boundary_edges_of_symmetry[symmetry][(a_x, a_y)]
        except KeyError:
            raise KeyError(
                f"No virtual pair found for symmetry {symmetry} and coordinates of the real "
                f"detector: ({a_x}, {a_y}). Define the virtual detector coordinate that "
                f"({a_x}, {a_y}) should be matched to in case this detector was matched to the "
                "boundary indicated by -1."
            )
        # The time coordinate is assumed to be the same as the real detector's time coordinate
        return (b_x, b_y, a_t)

    def find_encircled_nodes_from_closed_path(
        self, closed_path: list[tuple[int, int]]
    ) -> np.ndarray:
        return self._encircled_nodes_finder.find_encircled_nodes_from_closed_path(closed_path)

    def correction_postprocessing(self, correction):
        """
        Post-processing of the correction operator within one time slice (Algorithm 1 of
        https://arxiv.org/pdf/2505.05210):
        1. Find the 1-line with the largest overlap with the correction C (on ties, the first one
           in self.one_lines)
        2. If C covers at least half of this 1-line, multiply C by it and go back to 1.

        :param correction: indices of qubits to be X-flipped for correction
        :type correction: list of 2-tuples of int
        :return: correction with weight equal or smaller than weight of correction, covering less
            than half of every 1-line
        :rtype: list of 2-tuples of int
        """
        one_lines = [set(qbt_line) for qbt_line in self.one_lines.values()]
        final_correction = set(correction)
        while True:
            overlaps = [len(final_correction & qbt_line) for qbt_line in one_lines]
            # max() returns the first 1-line with the largest overlap
            m = max(range(len(one_lines)), key=lambda i: overlaps[i])
            if overlaps[m] < (len(one_lines[m]) + 1) / 2:
                return list(final_correction)
            # every multiplication shortens the correction, so the loop terminates
            final_correction ^= one_lines[m]

    def correction_postprocessing_ftp(
        self, corrections: dict[int, list[tuple[int, int]]]
    ) -> list[tuple[int, int]]:
        # post-processing is done within each time slice independently
        correction_postprocess = []
        for flips in corrections.values():
            # within each time slice remove even occurrences due to overlap between clusters
            correction = remove_even_occurrences(flips)
            correction_postprocess.extend(self.correction_postprocessing(correction))
        return correction_postprocess

    def find_interior_ftp_pp(self, clusters):
        """
        Find the correction operator within each time slice:
        * Project each cluster to one time slice by majority voting
        * Find the correction operator as the interior of the cluster within that time slice
        """
        corrections: dict[int, list[tuple[int, int]]] = {}
        for cluster in clusters:
            # project cluster to one plane by majority voting
            projected_cluster = _project_majority(cluster)
            time_plane = projected_cluster[0][-1]  # time plane to which we projected the cluster
            # remove time index from cluster indices
            spatial_cluster = _project_plane(projected_cluster)

            # find correction corresponding to the cluster projected to one plane
            corr = self.find_encircled_nodes_from_closed_path(spatial_cluster)
            corrections.setdefault(time_plane, []).extend(corr)
        return corrections

    def _match(self, syndrome: list | np.ndarray) -> list[Edge]:
        """Decode a syndrome into a list of matched detector-coordinate pairs. Implemented by each
        subclass, since this is the step that differs between ISM and GSM."""
        raise NotImplementedError

    def decode(
        self, syndrome: list | np.ndarray, post_processing: bool = False
    ) -> list[tuple[int, int]]:
        """Decode a single syndrome (binary array of length num_detectors indicating which detectors
        have been flipped)."""
        matches = self._match(syndrome)

        if not post_processing:
            corrections = self._encircled_nodes_finder.find_encircled_nodes_from_matches(matches)
        else:
            clusters = divide_into_clusters(matches)
            clusters_clean = remove_pure_measurement_errors(clusters)
            corrections = self.find_interior_ftp_pp(clusters_clean)
            corrections = self.correction_postprocessing_ftp(corrections)

        return remove_even_occurrences(corrections)
