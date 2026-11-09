# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import stim
from divide_into_clusters_cython import cy_divide_into_clusters

Node = tuple[int, int, int]
Edge = tuple[Node, Node]


def filter_dem(
    dem: stim.DetectorErrorModel, kept_detector_ids: list[int], reindex: bool = False
) -> stim.DetectorErrorModel:
    """
    Filter a DetectorErrorModel to only include specified detector IDs and reindex them if needed.
    Args:
        dem (stim.DetectorErrorModel): The original DetectorErrorModel.
        kept_detector_ids (list[int]): List of detector IDs to keep.
        reindex (bool): If True, reindex the kept detector IDs to be consecutive starting from 0.
    Returns:
        stim.DetectorErrorModel: The filtered and possibly reindexed DetectorErrorModel.
    """
    old_to_new = {old_id: i if reindex else old_id for i, old_id in enumerate(kept_detector_ids)}
    dem = dem.flattened()
    new_dem = stim.DetectorErrorModel()

    for instruction in dem:
        if instruction.type == "error":
            new_targets = []
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    old_d = t.val
                    if old_d in old_to_new:
                        new_targets.append(stim.DemTarget.relative_detector_id(old_to_new[old_d]))

            # Only keep if there's at least one detector in the error instruction
            if any(t.is_relative_detector_id() for t in new_targets):
                new_instruction = stim.DemInstruction("error", instruction.args_copy(), new_targets)
                new_dem.append(new_instruction)

        elif instruction.type == "detector":
            old_d = instruction.targets_copy()[0].val
            if reindex and old_d in old_to_new:
                new_dem.append(
                    "detector",
                    parens_arguments=instruction.args_copy(),
                    targets=[stim.DemTarget.relative_detector_id(old_to_new[old_d])],
                )

            elif not reindex:
                new_dem.append(
                    "detector",
                    parens_arguments=instruction.args_copy(),
                    targets=[stim.DemTarget.relative_detector_id(old_d)],
                )
    return new_dem


def divide_into_clusters(matches: list[Edge]) -> list[list[Node]]:
    """
    Divides a list of matches (tuples of nodes) into a list of disjoint clusters, where each cluster
    is a list of nodes and cluster[0] == cluster[-1].

    The input "matches" is a list of tuples, where each tuple corresponds to a pair of connected
    nodes (an edge in the matching graph). The way the matching is performed, the resulting matches
    always form closed loops; this splits them back up into individual loops.
    """
    node_to_index: dict[Node, int] = {}
    index_to_node: list[Node] = []
    encoded_edges = []
    for a, b in matches:
        for node in (a, b):
            if node not in node_to_index:
                node_to_index[node] = len(index_to_node)
                index_to_node.append(node)
        encoded_edges.append((node_to_index[a] << 32) | node_to_index[b])

    clusters = cy_divide_into_clusters(encoded_edges)
    return [[index_to_node[i] for i in cluster] for cluster in clusters]


def remove_pure_measurement_errors(clusters: Iterable[Sequence[Node]]) -> list[Sequence[Node]]:
    """Remove clusters that consist purely of a single edge back and forth between two nodes at the
    same spatial position (a measurement error rather than a data-qubit error)."""
    return [cluster for cluster in clusters if len({(x, y) for x, y, _ in cluster}) > 1]


def remove_even_occurrences(correction_operator: Iterable):
    """
    Count occurrences of each element in the list.
    Keep one occurrence of elements that appear an odd number of times.
    """
    count = Counter(correction_operator)
    return [elem for elem, count in count.items() if count % 2 != 0]


def project_cluster_to_plane(cluster: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    """Project a cluster of (x, y, t) nodes to (x, y) by dropping the time coordinate."""
    return [(x, y) for x, y, _ in cluster]


def project_cluster_to_majority_voted_plane(
    cluster: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Project all nodes of a cluster onto the single time plane that occurs most often in it."""
    time_counts: dict[int, int] = defaultdict(int)
    for _, _, t in cluster:
        time_counts[t] += 1
    majority_t = max(time_counts, key=time_counts.get)
    return [(x, y, majority_t) for x, y, _ in cluster]
