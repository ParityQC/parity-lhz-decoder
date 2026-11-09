# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
import numpy as np
from compiled_post_processing import post_process_components, post_process_slices

from parity_code.decoders.base import SymmetryMatchingDecoderBase
from parity_code.decoders.common import Node


class ComponentPostProcessing:
    """
    Post-processing of the matched edges of a symmetry matching decoder by connected components in
    space-time, in compiled code (`compiled_post_processing.post_process_components`).

    For every shot, given the matched edges as pairs of space-time points (x, y, t), including the
    edges along the virtual symmetry:
      1. Group the edges into connected components in space-time (union-find over the points).
      2. The data qubits flipped by a component are the XOR of the data qubits flipped by its edges
         (`SymmetryMatchingDecoderBase.flipped_data_qubits_of_edge`); this is the ray casting of the
         original decoders, applied to the component, since every component of the matched edges
         is a union of closed loops.
      3. Assign every component to a single time slice by majority vote: every edge that represents
         a data-qubit error, i.e. connects two different stabilizer positions of which at least one
         is real, votes for the time steps of its two ends. The time slice with the most votes wins,
         the earliest one on ties. Time-like edges (measurement errors) and edges between two
         virtual stabilizers do not vote.
      4. In every time slice, XOR the flips of its components and post-process the result against
         the 1-lines exactly as `SymmetryMatchingDecoderBase.correction_postprocessing` does.
      5. The correction is the XOR of the post-processed time slices.

    The original decoders instead split the matched edges into closed walks
    (`divide_into_clusters`), assign each walk to the time step that occurs most often among its
    nodes, and find its interior by ray casting. Components can contain several closed walks, so the
    two methods can assign some loops to different time slices; everything else is identical.

    Args:
        decoder: A decoder of `parity_code.decoders`, whose geometry (data qubits, symmetries, ray
            casting) and 1-lines are used.
    """

    def __init__(self, decoder: SymmetryMatchingDecoderBase):
        data_qubit_coords = decoder.data_qubit_coords
        self.num_data_qubits = len(data_qubit_coords)
        self.num_words = (self.num_data_qubits + 63) // 64
        data_index = {coord: i for i, coord in enumerate(data_qubit_coords)}

        # Space-time points: every stabilizer position (real and virtual) at every time step
        chains = decoder.measurement_qubits_of_symmetry
        positions = sorted({p for chain in chains.values() for p in chain})
        self._position_index = {p: i for i, p in enumerate(positions)}
        self.num_positions = len(positions)
        self.num_times = 1 + max(t for _, _, t in decoder.detector_coords.values())
        self.num_points = self.num_positions * self.num_times
        virtual = set(chains[0])
        self._is_virtual = np.array([p in virtual for p in positions], dtype=np.uint8)

        # Only edges between consecutive positions of a symmetry flip data qubits (as in the ray
        # casting of the original decoders). For every position: its neighbors along the symmetries
        # (padded with -1) and the data qubits flipped by the edge to each of them.
        neighbors = {p: [] for p in positions}
        for chain in chains.values():
            for p, q in zip(chain[:-1], chain[1:]):
                for a, b in ((p, q), (q, p)):
                    if b not in neighbors[a]:
                        neighbors[a].append(b)
        max_degree = max(len(n) for n in neighbors.values())
        self._neighbors = np.full((self.num_positions, max_degree), -1, dtype=np.int32)
        self._neighbor_flips = np.zeros(
            (self.num_positions * max_degree, self.num_words), dtype=np.uint64
        )
        for i, p in enumerate(positions):
            for j, q in enumerate(neighbors[p]):
                self._neighbors[i, j] = self._position_index[q]
                self._neighbor_flips[i * max_degree + j] = self.pack(
                    decoder.flipped_data_qubits_of_edge(p, q)
                )

        # 1-lines in the order of decoder.one_lines (which decides ties)
        one_lines = list(decoder.one_lines.values())
        self._one_lines = np.stack(
            [self.pack(data_index[q] for q in one_line) for one_line in one_lines]
        )
        self._one_line_lengths = np.array([len(one_line) for one_line in one_lines], dtype=np.int32)

    def pack(self, data_qubit_indices) -> np.ndarray:
        """Bit mask (uint64 words) of a set of data qubits, given by their indices."""
        mask = np.zeros(self.num_words, dtype=np.uint64)
        for i in data_qubit_indices:
            mask[i // 64] |= np.uint64(1) << np.uint64(i % 64)
        return mask

    def pack_corrections(self, corrections: np.ndarray) -> np.ndarray:
        """Bit masks (..., num_words) of a boolean array of corrections (..., num_data_qubits); the
        inverse of `unpack`."""
        corrections = np.asarray(corrections, dtype=bool)
        padding = [(0, 0)] * (corrections.ndim - 1) + [
            (0, 64 * self.num_words - self.num_data_qubits)
        ]
        bits = np.pad(corrections, padding)
        packed = np.packbits(bits, axis=-1, bitorder="little")
        return np.ascontiguousarray(packed).view("<u8").astype(np.uint64)

    def unpack(self, masks: np.ndarray) -> np.ndarray:
        """Boolean array (..., num_data_qubits) of an array of bit masks (..., num_words)."""
        masks = np.ascontiguousarray(masks, dtype="<u8")
        bits = np.unpackbits(masks.view(np.uint8), axis=-1, bitorder="little")
        return bits[..., : self.num_data_qubits].astype(bool)

    def point(self, coord: Node) -> int:
        """Index of the space-time point (x, y, t) of a real or virtual stabilizer."""
        x, y, t = coord
        return self._position_index[(x, y)] * self.num_times + t

    def __call__(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
        shot_offsets: np.ndarray,
        post_process: bool = True,
    ) -> np.ndarray:
        """
        Post-processed corrections of a batch of shots from their matched edges.

        Args:
            point_a, point_b: Space-time points (see `point`) of the two ends of every matched edge;
                the edges of shot i are shot_offsets[i] : shot_offsets[i + 1].
            shot_offsets: Integer array of length num_shots + 1. Edges that come in several
                segments, each grouped by shot (e.g. one per matching), need no reordering: then
                shot_offsets has shape (num_segments, num_shots + 1), and the edges of shot i are
                shot_offsets[k, i] : shot_offsets[k, i + 1] of every segment k. The result does not
                depend on the order of the edges.
            post_process: If False, return the corrections without post-processing, i.e. the XOR of
                the flips of all matched edges (the correction of the original decoders without
                post-processing).

        Returns:
            uint64 array (num_shots, num_words) with the bit mask of the corrected data qubits of
            every shot (see `unpack`).
        """
        return post_process_components(
            np.ascontiguousarray(point_a, dtype=np.int32),
            np.ascontiguousarray(point_b, dtype=np.int32),
            np.ascontiguousarray(np.atleast_2d(shot_offsets), dtype=np.int64),
            self.num_times,
            self._neighbors,
            self._neighbor_flips,
            self._is_virtual,
            self._one_lines,
            self._one_line_lengths,
            post_process,
        )

    def post_process_slices(self, corrections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        1-line post-processing of single time slices, as applied to every time slice.

        Args:
            corrections: Boolean array (num_slices, num_data_qubits) of the corrections to
                post-process.

        Returns:
            The post-processed corrections (same shape) and the number of post-processing cycles
            (1-line multiplications) of every correction.
        """
        masks = self.pack_corrections(corrections)
        cycles = post_process_slices(masks, self._one_lines, self._one_line_lengths)
        return self.unpack(masks), cycles


def concatenate_edges(edges_per_shot: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the matched edges of several shots, as returned by
    `pymatching.Matching.decode_to_edges_array`, into one (num_edges, 2) array and the offsets of
    the shots in it."""
    shot_offsets = np.zeros(len(edges_per_shot) + 1, dtype=np.int64)
    np.cumsum([len(edges) for edges in edges_per_shot], out=shot_offsets[1:])
    if shot_offsets[-1] == 0:
        return np.zeros((0, 2), dtype=np.int64), shot_offsets
    return np.concatenate(edges_per_shot).reshape(-1, 2), shot_offsets
