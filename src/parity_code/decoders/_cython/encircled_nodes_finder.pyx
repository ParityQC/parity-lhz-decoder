# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.

# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np

ctypedef np.int32_t INT32_t
ctypedef np.uint8_t UINT8_t
ctypedef np.intp_t INTP_t  # for indexing arrays

cdef class EncircledNodesFinder:
    # Ray casting: the data qubits encircled by closed loops of edges are the XOR of the data
    # qubits flipped by each edge (see SymmetryMatchingDecoderBase._get_flipped_data_qubits_of_edge)
    cdef:
        np.ndarray data_qubit_coords      # shape (N, 3)
        dict flipped_qubit_indices_of_edge  # edge → np.ndarray[INTP_t] (sparse)
        Py_ssize_t num_qubits
        np.ndarray flip_mask              # shape (N,)

    def __init__(
        self,
        np.ndarray data_qubit_coords_,
        dict flipped_qubit_indices_of_edge_,
    ):
        self.data_qubit_coords = data_qubit_coords_
        self.flipped_qubit_indices_of_edge = flipped_qubit_indices_of_edge_
        self.num_qubits = data_qubit_coords_.shape[0]
        self.flip_mask = np.zeros(self.num_qubits, dtype=np.uint8)

    cpdef list find_encircled_nodes_from_matches(self, list matches):
        cdef:
            np.ndarray[INT32_t, ndim=2] coords = self.data_qubit_coords
            np.ndarray[UINT8_t, ndim=1] flip_mask = self.flip_mask
            Py_ssize_t i, idx
            tuple node1, node2, a, b, edge
            int x1, y1, t1, x2, y2, t2
            np.ndarray[INTP_t, ndim=1] flipped_indices
            list result = []

        # Reset flip_mask
        flip_mask[:] = 0

        for node1, node2 in matches:
            x1, y1, t1 = node1
            x2, y2, t2 = node2

            a = (x1, y1)
            b = (x2, y2)
            edge = (b, a) if a > b else (a, b)

            flipped_indices = self.flipped_qubit_indices_of_edge.get(edge, None)
            if flipped_indices is not None:
                for i in range(flipped_indices.shape[0]):
                    idx = flipped_indices[i]
                    flip_mask[idx] ^= 1

        for i in range(self.num_qubits):
            if flip_mask[i]:
                result.append((int(coords[i, 0]), int(coords[i, 1])))

        return result

    cpdef list find_encircled_nodes_from_closed_path(self, list closed_path):
        cdef:
            np.ndarray[INT32_t, ndim=2] coords = self.data_qubit_coords
            np.ndarray[UINT8_t, ndim=1] flip_mask = self.flip_mask
            Py_ssize_t i, idx, ii, jj
            tuple node1, node2, a, b, edge
            int x1, y1, x2, y2
            np.ndarray[INTP_t, ndim=1] flipped_indices
            list result = []

        # Reset flip_mask
        flip_mask[:] = 0

        for ii in range(len(closed_path)-1):
            jj = ii+1
            x1, y1 = closed_path[ii]
            x2, y2 = closed_path[jj]

            a = (x1, y1)
            b = (x2, y2)
            edge = (b, a) if a > b else (a, b)

            flipped_indices = self.flipped_qubit_indices_of_edge.get(edge, None)
            if flipped_indices is not None:
                for i in range(flipped_indices.shape[0]):
                    idx = flipped_indices[i]
                    flip_mask[idx] ^= 1

        for i in range(self.num_qubits):
            if flip_mask[i]:
                result.append((int(coords[i, 0]), int(coords[i, 1])))

        return result
