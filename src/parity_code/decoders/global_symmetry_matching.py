# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
from collections import defaultdict

import numpy as np
import pymatching
import stim

from parity_code.decoders.base import SymmetryMatchingDecoderBase
from parity_code.decoders.common import Edge, filter_dem


def _offset_detector_ids(dem: stim.DetectorErrorModel, offset: int) -> stim.DetectorErrorModel:
    """
    Offsets the detector IDs in a DetectorErrorModel by a specified amount.
    """
    new_dem = stim.DetectorErrorModel()
    for instruction in dem:
        if instruction.type == "error":
            new_targets = []
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    new_targets.append(stim.DemTarget.relative_detector_id(t.val + offset))
                else:
                    new_targets.append(t)  # keep logical observables/logical frame changes as they are
            new_dem.append("error", instruction.args_copy(), new_targets)
        elif instruction.type == "detector":
            new_dem.append(
                "detector",
                parens_arguments=instruction.args_copy(),
                targets=[
                    stim.DemTarget.relative_detector_id(instruction.targets_copy()[0].val + offset)
                ],
            )
    return new_dem


class GlobalSymmetryMatchingDecoder(SymmetryMatchingDecoderBase):
    """
    Decodes the parity code under circuit-level noise using Global Symmetry Matching (GSM), see
    https://arxiv.org/pdf/2505.05210: the per-symmetry detector error models are combined (with their
    detector IDs offset to stay disjoint) into a single global matching graph, together with the
    virtual symmetry, and decoded with one minimum-weight perfect matching.
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

        self.dems_per_symmetry = self._get_dems_per_real_symmetry()
        self.global_dem, self.symmetry_of_det = self._get_global_dem()
        self.global_detector_coords = {
            d: (int(coord[0]), int(coord[1]), int(coord[2]))
            for d, coord in self.global_dem.get_detector_coordinates().items()
        }

        self.global_matching, virtual_det_id_to_coord = self._get_global_matching()
        self.global_detector_coords.update(virtual_det_id_to_coord)

        # Precompute a mapping from a detector of the original DEM to its two counterparts in the
        # global DEM (one per real symmetry the detector belongs to).
        det_coord_to_global_det_ids = defaultdict(list)
        for d, coord in self.global_detector_coords.items():
            det_coord_to_global_det_ids[coord].append(d)
        self.det_id_to_global_det_ids = np.zeros((len(self.detector_coords), 2), dtype=int)
        for d, coord in self.detector_coords.items():
            self.det_id_to_global_det_ids[d, :] = det_coord_to_global_det_ids[coord]

    def _get_dems_per_real_symmetry(self) -> dict[int, stim.DetectorErrorModel]:
        return {
            s: filter_dem(self.dem, detectors, reindex=True)
            for s, detectors in self.detectors_of_real_symmetries.items()
        }

    def _get_global_dem(self) -> tuple[stim.DetectorErrorModel, dict[int, int]]:
        """Concatenate the per-symmetry DEMs into one DEM, offsetting detector IDs so that each
        symmetry occupies a disjoint range, and record which symmetry each global detector ID
        belongs to."""
        new_dem = stim.DetectorErrorModel()
        symmetry_of_det = {}
        offset = 0
        for s, dem_s in self.dems_per_symmetry.items():
            max_detector_id = max(dem_s.get_detector_coordinates().keys())
            offset_dem = _offset_detector_ids(dem_s, offset=offset)

            for det_id in range(offset, offset + max_detector_id + 1):
                symmetry_of_det[det_id] = s

            offset += max_detector_id + 1
            new_dem += offset_dem
        return new_dem, symmetry_of_det

    def _get_global_matching(self) -> tuple[pymatching.Matching, dict[int, tuple[int, int, int]]]:
        max_t = max(coord[2] for coord in self.detector_coords.values())
        virtual_measure_qbt_coords = self.measurement_qubits_of_symmetry[0]

        matching_without_virtuals = pymatching.Matching.from_detector_error_model(self.global_dem)
        global_matching = pymatching.Matching()

        virtual_det_coord_to_id = {}
        virtual_det_id_to_coord: dict[int, tuple[int, int, int]] = {}
        virtual_det_id = max(self.global_dem.get_detector_coordinates().keys()) + 1
        for t in range(max_t + 1):
            for x, y in virtual_measure_qbt_coords:
                virtual_det_coord_to_id[(x, y, t)] = virtual_det_id
                virtual_det_id_to_coord[virtual_det_id] = (x, y, t)
                virtual_det_id += 1

        for a, b, info in matching_without_virtuals.edges():
            if b is None:  # boundary edges: b is None, a is always a real detector
                s = self.symmetry_of_det[a]
                coord_a = self.global_detector_coords[a]
                coord_b = self.get_matched_virtual_det_coord(s, coord_a)
                global_matching.add_edge(a, virtual_det_coord_to_id[coord_b], weight=info["weight"])
            else:
                global_matching.add_edge(a, b, weight=info["weight"])

        # spatial edges between virtual detectors
        for t in range(max_t + 1):
            for i in range(len(virtual_measure_qbt_coords) - 1):
                a_x, a_y = virtual_measure_qbt_coords[i]
                b_x, b_y = virtual_measure_qbt_coords[i + 1]
                global_matching.add_edge(
                    virtual_det_coord_to_id[(a_x, a_y, t)],
                    virtual_det_coord_to_id[(b_x, b_y, t)],
                    weight=1,
                )

        # temporal edges between virtual detectors
        for x, y in virtual_measure_qbt_coords:
            for t in range(max_t):
                global_matching.add_edge(
                    virtual_det_coord_to_id[(x, y, t)],
                    virtual_det_coord_to_id[(x, y, t + 1)],
                    weight=1,
                )

        global_matching.ensure_num_fault_ids(len(self.one_lines.keys()) - 1)
        return global_matching, virtual_det_id_to_coord

    def _match(self, syndrome: list | np.ndarray) -> list[Edge]:
        # The input syndrome corresponds to the original DEM; adjust it to the global DEM, where each
        # original detector maps to two detectors (one per real symmetry it belongs to).
        syndrome = np.asarray(syndrome, dtype=bool)  # a mask, not detector indices
        global_syndrome = np.zeros(len(self.global_detector_coords), dtype=bool)
        active_global_dets = self.det_id_to_global_det_ids[syndrome, :].flatten()
        global_syndrome[active_global_dets] = True

        edges = self.global_matching.decode_to_edges_array(global_syndrome)

        matched_det_coords_including_virtual = []
        for a, b in edges:
            coord_a = self.global_detector_coords[a]
            if b == -1:
                # detector "a" was matched to the boundary; "a" is always a real detector
                s = self.symmetry_of_det[a]
                coord_b = self.get_matched_virtual_det_coord(s, coord_a)
            else:
                coord_b = self.global_detector_coords[b]
            matched_det_coords_including_virtual.append((coord_a, coord_b))
        return matched_det_coords_including_virtual
