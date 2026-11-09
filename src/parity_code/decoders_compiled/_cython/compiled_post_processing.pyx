# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""
Compiled kernel of the post-processing by space-time connected components, see
`parity_code.decoders_compiled.post_processing` for the algorithm and the meaning of the inputs.

Sets of data qubits are bit masks of `num_words` 64-bit words (data qubit i is bit i % 64 of word
i // 64). A space-time point, i.e. a stabilizer position at a time step, is the integer
`position * num_times + time`.
"""
import numpy as np

from libc.stdint cimport int32_t, int64_t, uint8_t, uint64_t


cdef inline int popcount64(uint64_t x) noexcept nogil:
    # portable population count (number of set bits)
    x = x - ((x >> 1) & 0x5555555555555555ULL)
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL
    return <int>((x * 0x0101010101010101ULL) >> 56)


cdef inline int32_t find_root(int32_t* parent, int32_t x) noexcept nogil:
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path halving
        x = parent[x]
    return x


cdef int post_process_slice(
    uint64_t* flips,
    const uint64_t* one_lines,
    const int32_t* one_line_lengths,
    Py_ssize_t num_one_lines,
    Py_ssize_t num_words,
) noexcept nogil:
    # Post-processing within one time slice, in place (as SymmetryMatchingDecoderBase.
    # correction_postprocessing): while the 1-line with the largest overlap with the correction (the
    # first one on ties) is covered at least half, multiply the correction by it. Returns the number
    # of cycles, i.e. of multiplications.
    cdef Py_ssize_t k, w, best
    cdef int overlap, best_overlap
    cdef int cycles = 0
    while True:
        best = 0
        best_overlap = -1
        for k in range(num_one_lines):
            overlap = 0
            for w in range(num_words):
                overlap += popcount64(flips[w] & one_lines[k * num_words + w])
            if overlap > best_overlap:
                best_overlap = overlap
                best = k
        if 2 * best_overlap < one_line_lengths[best] + 1:
            return cycles
        # every multiplication shortens the correction, so the loop terminates
        for w in range(num_words):
            flips[w] ^= one_lines[best * num_words + w]
        cycles += 1


def post_process_slices(
    uint64_t[:, ::1] corrections,
    const uint64_t[:, ::1] one_lines,
    const int32_t[::1] one_line_lengths,
):
    """Apply the 1-line post-processing to every row of `corrections` (bit masks), in place. Returns
    the number of cycles (1-line multiplications) of every row."""
    cycles_arr = np.zeros(corrections.shape[0], dtype=np.int32)
    cdef int32_t[::1] cycles = cycles_arr
    cdef Py_ssize_t i
    with nogil:
        for i in range(corrections.shape[0]):
            cycles[i] = post_process_slice(
                &corrections[i, 0],
                &one_lines[0, 0],
                &one_line_lengths[0],
                one_lines.shape[0],
                one_lines.shape[1],
            )
    return cycles_arr


def post_process_components(
    const int32_t[::1] point_a,
    const int32_t[::1] point_b,
    const int64_t[:, ::1] shot_offsets,
    int num_times,
    const int32_t[:, ::1] neighbors,
    const uint64_t[:, ::1] neighbor_flips,
    const uint8_t[::1] is_virtual,
    const uint64_t[:, ::1] one_lines,
    const int32_t[::1] one_line_lengths,
    bint post_process=True,
):
    """
    Post-processed corrections of a batch of shots from their matched edges.

    Args:
        point_a, point_b: Space-time points of the two ends of every matched edge. The edges come in
            segments (e.g. one per matching), each grouped by shot: the edges of shot i in segment k
            are shot_offsets[k, i] : shot_offsets[k, i + 1]. The result does not depend on the order
            of the edges.
        shot_offsets: Array of shape (num_segments, num_shots + 1).
        num_times: Number of time steps.
        neighbors: (num_positions, max_degree) the positions next to every stabilizer position along
            the symmetries, padded with -1. Only edges between such positions flip data qubits.
        neighbor_flips: (num_positions * max_degree, num_words) data qubits flipped by an edge between
            position p and neighbors[p, j] (row p * max_degree + j).
        is_virtual: (num_positions,) whether a position is a virtual stabilizer. An edge between two
            different positions votes for the time steps of its ends unless both are virtual.
        one_lines: (num_one_lines, num_words) the 1-lines, in the order of their priority on ties.
        one_line_lengths: (num_one_lines,) the number of qubits of every 1-line.
        post_process: If False, return the correction without post-processing, the XOR of the flips
            of all components (i.e. of all matched edges).

    Returns:
        uint64 array (num_shots, num_words) with the (post-processed) correction of every shot.
    """
    cdef Py_ssize_t num_segments = shot_offsets.shape[0]
    cdef Py_ssize_t num_shots = shot_offsets.shape[1] - 1
    cdef Py_ssize_t num_words = neighbor_flips.shape[1]
    cdef Py_ssize_t max_degree = neighbors.shape[1]
    cdef Py_ssize_t num_points = neighbors.shape[0] * <Py_ssize_t>num_times
    cdef Py_ssize_t num_one_lines = one_lines.shape[0]

    corrections_arr = np.zeros((num_shots, num_words), dtype=np.uint64)
    # workspace, reset after every shot: union-find forest over the space-time points (-1: not part of
    # any matched edge), and the flips and per-time-step votes of every component, kept at its root
    parent_arr = np.full(num_points, -1, dtype=np.int32)
    component_flips_arr = np.zeros((num_points, num_words), dtype=np.uint64)
    component_votes_arr = np.zeros((num_points, num_times), dtype=np.int32)
    touched_arr = np.zeros(num_points, dtype=np.int32)
    slice_flips_arr = np.zeros((num_times, num_words), dtype=np.uint64)
    slice_used_arr = np.zeros(num_times, dtype=np.uint8)

    cdef uint64_t[:, ::1] corrections = corrections_arr
    cdef int32_t[::1] parent = parent_arr
    cdef uint64_t[:, ::1] component_flips = component_flips_arr
    cdef int32_t[:, ::1] component_votes = component_votes_arr
    cdef int32_t[::1] touched = touched_arr
    cdef uint64_t[:, ::1] slice_flips = slice_flips_arr
    cdef uint8_t[::1] slice_used = slice_used_arr

    cdef Py_ssize_t shot, k, e, i, w, t, j, row
    cdef int32_t a, b, position_a, position_b, root_a, root_b, root, num_touched, best_t, best_votes
    cdef bint has_flips

    with nogil:
        for shot in range(num_shots):
            # 1. connected components of the matched edges in space-time
            num_touched = 0
            for k in range(num_segments):
                for e in range(shot_offsets[k, shot], shot_offsets[k, shot + 1]):
                    a = point_a[e]
                    b = point_b[e]
                    if parent[a] < 0:
                        parent[a] = a
                        touched[num_touched] = a
                        num_touched += 1
                    if parent[b] < 0:
                        parent[b] = b
                        touched[num_touched] = b
                        num_touched += 1
                    root_a = find_root(&parent[0], a)
                    root_b = find_root(&parent[0], b)
                    if root_a != root_b:
                        parent[root_a] = root_b

            # 2. flipped data qubits (XOR of the per-edge flips) and time votes of every component.
            # Time-like edges (measurement errors, usually most of the edges) neither flip nor vote.
            for k in range(num_segments):
                for e in range(shot_offsets[k, shot], shot_offsets[k, shot + 1]):
                    a = point_a[e]
                    b = point_b[e]
                    position_a = a // num_times
                    position_b = b // num_times
                    if position_a == position_b:
                        continue
                    root = find_root(&parent[0], a)
                    for j in range(max_degree):
                        if neighbors[position_a, j] == position_b:
                            row = position_a * max_degree + j
                            for w in range(num_words):
                                component_flips[root, w] ^= neighbor_flips[row, w]
                            break
                    if not (is_virtual[position_a] and is_virtual[position_b]):
                        component_votes[root, a % num_times] += 1
                        component_votes[root, b % num_times] += 1

            # 3. every component goes to the time slice with the most votes (the earliest on ties)
            for i in range(num_touched):
                root = touched[i]
                if parent[root] != root:
                    continue
                has_flips = False
                for w in range(num_words):
                    if component_flips[root, w]:
                        has_flips = True
                if not has_flips:
                    continue
                if not post_process:
                    for w in range(num_words):
                        corrections[shot, w] ^= component_flips[root, w]
                    continue
                # fallback for a component without votes; cannot occur, since only edges that vote
                # (with a real end) flip data qubits
                best_t = root % num_times
                best_votes = 0
                for t in range(num_times):
                    if component_votes[root, t] > best_votes:
                        best_votes = component_votes[root, t]
                        best_t = t
                for w in range(num_words):
                    slice_flips[best_t, w] ^= component_flips[root, w]
                slice_used[best_t] = 1

            # 4. post-process every time slice and XOR the slices
            for t in range(num_times):
                if slice_used[t]:
                    post_process_slice(
                        &slice_flips[t, 0],
                        &one_lines[0, 0],
                        &one_line_lengths[0],
                        num_one_lines,
                        num_words,
                    )
                    for w in range(num_words):
                        corrections[shot, w] ^= slice_flips[t, w]
                        slice_flips[t, w] = 0
                    slice_used[t] = 0

            # reset the workspace; only the roots of the components carry flips and votes
            for i in range(num_touched):
                a = touched[i]
                if parent[a] == a:
                    for w in range(num_words):
                        component_flips[a, w] = 0
                    for t in range(num_times):
                        component_votes[a, t] = 0
                parent[a] = -1

    return corrections_arr
