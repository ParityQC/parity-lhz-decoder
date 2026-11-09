# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
import numpy as np
import pymatching
import stim

from parity_code.decoders.base import SymmetryMatchingDecoderBase
from parity_code.decoders.common import Edge, Node, filter_dem


class IndependentSymmetryMatchingDecoder(SymmetryMatchingDecoderBase):
    """
    Decodes the parity code under circuit-level noise using Independent Symmetry Matching (ISM), see
    https://arxiv.org/pdf/2505.05210: each symmetry of the code is matched independently with its own
    minimum-weight perfect matching, and the detectors matched to the boundary ("-1") are then matched
    to each other along the virtual symmetry.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        data_qubit_coords: list[tuple[int, int]],
        measurement_qubits_of_symmetry: dict[int, list[tuple[int, int]]],
        boundary_edges_of_real_symmetry: dict[int, dict[tuple, tuple]],
        one_lines: dict[int, list[tuple[int, int]]],
    ):
        super().__init__(
            dem,
            data_qubit_coords,
            measurement_qubits_of_symmetry,
            boundary_edges_of_real_symmetry,
            one_lines,
        )

        # One pymatching.Matching per real symmetry, decoding that symmetry's detectors on their own
        self.dems_per_symmetry = self._get_dems_per_real_symmetry()
        self.matchings_per_symmetry = {
            s: pymatching.Matching.from_detector_error_model(dem)
            for s, dem in self.dems_per_symmetry.items()
        }

        # Precompute a mask that, given a syndrome, selects the detectors of a given symmetry
        self.syndrome_masks_per_symmetry: dict[int, np.ndarray] = {}
        for s, detectors in self.detectors_of_real_symmetries.items():
            mask = np.zeros(len(self.detector_coords), dtype=bool)
            mask[detectors] = True
            self.syndrome_masks_per_symmetry[s] = mask

        # A separate matching for pairing up the virtual detectors along the virtual symmetry
        self.virtual_matching, self.virtual_idx_to_node, self.virtual_node_to_idx = (
            self._get_virtual_matching()
        )

    def _get_dems_per_real_symmetry(self) -> dict[int, stim.DetectorErrorModel]:
        return {
            s: filter_dem(self.dem, detectors, reindex=False)
            for s, detectors in self.detectors_of_real_symmetries.items()
        }

    def _get_virtual_matching(self) -> tuple[pymatching.Matching, dict, dict]:
        # this assumes that the time coordinate is at index 2
        max_t = max(coord[2] for coord in self.detector_coords.values())

        # this assumes that the virtual measurement qubits are ordered along the symmetry
        virtual_measure_qubit_coords = self.measurement_qubits_of_symmetry[0]

        node_to_index = {}
        index_to_node = {}
        index = 0
        for t in range(max_t + 1):
            for x, y in virtual_measure_qubit_coords:
                node_to_index[(x, y, t)] = index
                index_to_node[index] = (x, y, t)
                index += 1

        virtual_matching = pymatching.Matching()
        # Add spatial edges
        for t in range(max_t + 1):
            for i in range(len(virtual_measure_qubit_coords) - 1):
                a_x, a_y = virtual_measure_qubit_coords[i]
                b_x, b_y = virtual_measure_qubit_coords[i + 1]
                virtual_matching.add_edge(
                    node_to_index[(a_x, a_y, t)], node_to_index[(b_x, b_y, t)], weight=1
                )

        # Add temporal edges
        for x, y in virtual_measure_qubit_coords:
            for t in range(max_t):
                virtual_matching.add_edge(
                    node_to_index[(x, y, t)], node_to_index[(x, y, t + 1)], weight=1
                )

        return virtual_matching, index_to_node, node_to_index

    def match_virtual_detectors(self, added_virtual_detectors: list[Node]) -> list[Edge]:
        """Match the virtual detectors along the virtual symmetry."""
        syndrome = np.zeros(len(self.virtual_node_to_idx), dtype=bool)
        for node in added_virtual_detectors:
            syndrome[self.virtual_node_to_idx[node]] = True

        edges = self.virtual_matching.decode_to_edges_array(syndrome)
        return [(self.virtual_idx_to_node[a], self.virtual_idx_to_node[b]) for a, b in edges]

    def _match(self, syndrome: list | np.ndarray) -> list[Edge]:
        added_virtual_detectors: list[Node] = []

        # Matched detector coordinates, with "-1" (boundary) matches replaced by the corresponding
        # virtual detector coordinate.
        matched_det_coords_including_virtual: list[Edge] = []

        for s, matching in self.matchings_per_symmetry.items():
            syndrome_s = syndrome & self.syndrome_masks_per_symmetry[s]
            edges = matching.decode_to_edges_array(syndrome_s)
            # edges is a list of tuples (a, b) where a and b are the indices of the matched detectors

            for a, b in edges:
                coord_a: Node = self.detector_coords[a]
                if b == -1:
                    # detector "a" was matched to the boundary; "a" will always be a real detector
                    coord_b: Node = self.get_matched_virtual_det_coord(s, coord_a)
                    matched_det_coords_including_virtual.append((coord_a, coord_b))
                    added_virtual_detectors.append(coord_b)
                else:
                    matched_det_coords_including_virtual.append((coord_a, self.detector_coords[b]))

        virtual_matches = self.match_virtual_detectors(added_virtual_detectors)
        return virtual_matches + matched_det_coords_including_virtual
