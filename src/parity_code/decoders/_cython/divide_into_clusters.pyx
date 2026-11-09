# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.

# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

ctypedef long long EdgeKey  # Encoded edge as (a << 32 | b)

cdef inline int get_other_node(EdgeKey key, int current):
    cdef int a = key >> 32
    cdef int b = key & 0xffffffff
    return b if a == current else a

def cy_divide_into_clusters(edges):
    """
    edges: list[int] where each int encodes a pair (a << 32 | b)
    returns: list[list[int]]
    """
    cdef vector[EdgeKey] c_edges
    cdef EdgeKey edge
    for edge in edges:
        c_edges.push_back(edge)

    cdef unordered_map[int, vector[EdgeKey]] edge_map
    cdef unordered_map[EdgeKey, int] edge_count
    cdef vector[vector[int]] clusters
    cdef vector[int] loop
    cdef int a, b, start, current, next_node
    cdef EdgeKey next_edge
    cdef vector[EdgeKey]* neighbors
    cdef vector[EdgeKey] neighbor_list
    cdef size_t i
    cdef bint found

    # Build edge map and edge count
    for edge in c_edges:
        a = edge >> 32
        b = edge & 0xffffffff
        edge_map[a].push_back(edge)
        edge_map[b].push_back(edge)

        if edge_count.find(edge) == edge_count.end():
            edge_count[edge] = 0
        edge_count[edge] += 1

    # Traverse and extract clusters
    for edge in c_edges:
        if edge_count[edge] == 0:
            continue

        a = edge >> 32
        b = edge & 0xffffffff
        start = a
        current = b
        loop.clear()
        loop.push_back(a)
        loop.push_back(b)
        edge_count[edge] -= 1

        while current != start:
            neighbors = &edge_map[current]
            neighbor_list = neighbors[0]
            found = False
            for i in range(neighbor_list.size()):
                next_edge = neighbor_list[i]
                if edge_count[next_edge] > 0:
                    edge_count[next_edge] -= 1
                    next_node = get_other_node(next_edge, current)
                    loop.push_back(next_node)
                    current = next_node
                    found = True
                    break
            if not found:
                raise ValueError(f"Stuck at node {current} with no unused edges.")
        clusters.push_back(loop[:])

    # Convert result to Python-native types
    cdef list py_clusters = []
    cdef list py_loop
    for i in range(clusters.size()):
        py_loop = [clusters[i][j] for j in range(clusters[i].size())]
        py_clusters.append(py_loop)

    return py_clusters