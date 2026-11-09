# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import stim


class LhzLegEncoding:
    """
    Class defining the LHZ error correcting code, where data qubits (qubits with a single index) are
    placed on one leg of the isoscele triangle.

    `create_stim_circuit` builds a noiseless syndrome-extraction circuit; use a class from
    `parity_code.noise_models` to add noise to it.
    """

    def __init__(self, num_logical_qubits):
        self.num_logical_qubits = num_logical_qubits
        self.qubits = {}  # {qubit_coord: {'label': tuple of logical qubit indices, 'type': 'data'/'parity'}, ...}
        self.base_qubit_coords = []
        self.parity_qubit_coords = []
        self.measure_qubit_coords = []
        self.logical_x_observables = []
        self.logical_z_observables = []
        self.constraints = {}  # {measure_qubit_coord: [tuples of qubit coordinates of the constraint], ...}
        self.labels_of_constraint = {}
        self._create_lhz_layout()

    def _create_lhz_layout(self):
        # define the data/base qubits, parity qubits and measurement qubits by their coordinates
        # for data and parity qubits both the x and y coordinate is even
        # measurement qubits have both x and y coordinate odd
        # qubits corresponding to products of stabilizers would have one even and one odd coordinate, but are not considered at the moment

        # the base qubits are placed along the y axis
        for i, y in enumerate(range(0, 2 * self.num_logical_qubits - 1, 2)):
            self.base_qubit_coords.append((0, y))
            self.qubits[(0, y)] = {"label": (i,), "type": "data"}

        # there are k*(k-1)/2 parity qubits, where k is the number of data qubits
        for i in range(0, self.num_logical_qubits):
            for j in range(i + 1, self.num_logical_qubits):
                self.parity_qubit_coords.append((2 * i + 2, 2 * j))
                self.qubits[(2 * i + 2, 2 * j)] = {"label": (i, j), "type": "parity"}

        # define the logical observables
        self.logical_x_observables = np.zeros(
            (self.num_logical_qubits, len(self.base_qubit_coords + self.parity_qubit_coords))
        )
        for i in range(self.num_logical_qubits):
            for j, qubit in enumerate(self.base_qubit_coords + self.parity_qubit_coords):
                if i in self.qubits[qubit]["label"]:
                    self.logical_x_observables[i][j] = 1

        self.logical_z_observables = np.zeros(
            (self.num_logical_qubits, len(self.base_qubit_coords + self.parity_qubit_coords))
        )
        for i in range(self.num_logical_qubits):
            self.logical_z_observables[i][i] = 1

        # there are k*(k-1)/2 measurement qubits, each checking the parity of 3 or 4 physical qubits
        self.constraints = {}
        for x in range(1, 2 * self.num_logical_qubits - 1, 2):
            for y in range(x, 2 * self.num_logical_qubits - 1, 2):
                self.measure_qubit_coords.append((x, y))
                const = []
                const.append((x - 1, y - 1))
                if (x + 1, y - 1) in self.parity_qubit_coords:
                    const.append((x + 1, y - 1))
                const.append((x - 1, y + 1))
                const.append((x + 1, y + 1))
                self.constraints[(x, y)] = const
                # define the labels of the qubits appearing in the constraint
                self.labels_of_constraint[(x, y)] = [self.qubits[q]["label"] for q in const]

    def create_stim_circuit(self, rounds: int) -> stim.Circuit:
        """Build a noiseless syndrome-extraction circuit with the given number of rounds. Use a class
        from `parity_code.noise_models` to add noise to the result."""
        # the qubits are index by their position in the list of data_qubits + parity_qubits + measurement_qubits
        # IMPORTANT: This specified order is very important for code to work
        pos_to_q_index = {
            pos: i
            for i, pos in enumerate(
                self.base_qubit_coords + self.parity_qubit_coords + self.measure_qubit_coords
            )
        }

        # specifying the order in which the CNOT gates of the syndrome extraction are applied
        # we start with the lower left qubit of each constraint. The order is (lower left, upper left, upper right, lower right) as seen from the measurement qubit
        # the threebody constraints consist of only 3 qubits, in the direction (lower left, upper left, upper right) as seen from the measurement qubit
        # sorting the constraints and reverting the order of the last two entries for the 4-body constraints ensures the above specified order
        # and helps to treat both 3- and 4-body constraints similarly for the steps of applying the CNOTs
        modified_constraints = {}
        for measure_q_coord, constr in self.constraints.items():
            sorted_constr = tuple(sorted(constr))
            modified_constraints[measure_q_coord] = (
                sorted_constr[:2] + sorted_constr[-1:1:-1] if len(constr) > 3 else sorted_constr
            )

        # Create the CNOT pairs for each step of the syndrome extraction
        CNOT_pairs = []
        for step in range(4):
            CNOT_step_pairs = []
            for measure_q_coord, constr in modified_constraints.items():
                if step < len(constr):
                    data_q_index = pos_to_q_index[constr[step]]
                    measure_q_index = pos_to_q_index[measure_q_coord]
                    if (
                        measure_q_index not in CNOT_step_pairs
                        and data_q_index not in CNOT_step_pairs
                    ):
                        CNOT_step_pairs.extend([data_q_index, measure_q_index])
            if CNOT_step_pairs:
                CNOT_pairs.append(CNOT_step_pairs)

        # Create the STIM circuit
        #####--CYCLE--###################################################
        cycle_action = stim.Circuit()

        for targets in CNOT_pairs:
            cycle_action.append("TICK")
            cycle_action.append("CX", targets)

        # Measure the measurement qubits to extract the syndrome
        measure_indices = [pos_to_q_index[coord] for coord in self.measure_qubit_coords]
        cycle_action.append("TICK")
        cycle_action.append("M", measure_indices)
        cycle_action.append("TICK")
        cycle_action.append("R", measure_indices)

        ####--HEAD--####################################################
        # Build the start of the circuit, getting a state that's ready to cycle
        # In particular, the first cycle has different detectors and so has to be handled special.
        head = stim.Circuit()
        for index, pos in enumerate(
            self.base_qubit_coords + self.parity_qubit_coords + self.measure_qubit_coords
        ):
            head.append("QUBIT_COORDS", [index], [pos[0], pos[1]])

        # Reset all data, parity, and measurement qubits
        head.append("TICK")
        head.append(
            "R",
            [
                pos_to_q_index[q]
                for q in self.base_qubit_coords
                + self.parity_qubit_coords
                + self.measure_qubit_coords
            ],
        )

        head += cycle_action
        for i, measure in enumerate(self.measure_qubit_coords):
            head.append(
                "DETECTOR",
                [stim.target_rec(-len(self.measure_qubit_coords) + i)],
                [measure[0], measure[1], 0.0],
            )

        ####--BODY--####################################################
        # Build the repeated body of the circuit, including the detectors that compare measurement outcomes to previous cycles.
        body = cycle_action.copy()
        m = len(self.measure_qubit_coords)
        body.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
        for i, m_coord in enumerate(self.measure_qubit_coords):
            k = len(self.measure_qubit_coords) - i - 1
            body.append(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                [m_coord[0], m_coord[1], 0.0],
            )

        ####--TAIL--####################################################
        # Build the end of the circuit, getting out of the cycle state and terminating.
        # In particular, the data measurements create detectors that have to be handled special.
        # Also, the tail is responsible for identifying the logical observable.
        tail = stim.Circuit()
        tail.append("TICK")
        tail.append(
            "M", [pos_to_q_index[q] for q in self.base_qubit_coords + self.parity_qubit_coords]
        )

        # Add detectors, that compare the measurement outcomes of the measurment qubits between each round and with the data and parity qubits in the last round
        for i, measure in enumerate(self.measure_qubit_coords):
            detectors = []
            for qubit in self.constraints[measure]:
                detectors.append(
                    -len(self.base_qubit_coords + self.parity_qubit_coords) + pos_to_q_index[qubit]
                )
            detectors.append(
                -len(self.base_qubit_coords + self.parity_qubit_coords)
                - len(self.measure_qubit_coords)
                + i
            )
            detectors.sort(reverse=True)
            tail.append(
                "DETECTOR", [stim.target_rec(x) for x in detectors], [measure[0], measure[1], 1.0]
            )

        # Include the logical Z observables
        # For each observable only the base qubit is included, since measuring these qubits respectively corresponds to measuring the logical Z observables
        for o, observable in enumerate(self.logical_z_observables):
            obs_qubits = [
                q
                for i, q in enumerate(self.base_qubit_coords + self.parity_qubit_coords)
                if observable[i] == 1
            ]
            obs_inc = []
            for q in obs_qubits:
                obs_inc.append(
                    -len(self.base_qubit_coords + self.parity_qubit_coords) + pos_to_q_index[q]
                )
            obs_inc.sort(reverse=True)
            tail.append("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in obs_inc], o)

        return head + body * (rounds - 1) + tail

    @property
    def measurement_qubits_of_symmetry(self) -> dict:
        """
        Return a dict {symmetry: [measurement qubit coordinates of symmetry]} corresponding to a specific symmetry line.
        Both real and virtual measurement qubits are specified.

        The virtual symmetry line has index 0
        The real symmetry lines are indexed from 1 to K, where K is the number of logical qubits.
        """
        measure_qubits_of_symmetry = {}
        max_coord = 2 * (self.num_logical_qubits - 1)

        # define virtual measurment qubits of the virtual symmetry line (0)
        measure_qubits_of_symmetry[0] = [(-1, y) for y in range(-1, max_coord + 1, 2)]
        measure_qubits_of_symmetry[0].extend((x, max_coord + 1) for x in range(1, max_coord + 3, 2))

        # define the real measurement qubits of the real symmetry lines (1 to K)
        for symmetry_index in range(1, self.num_logical_qubits + 1):
            measure_qubits_of_symmetry[symmetry_index] = []
            for x in range(1, 2 * symmetry_index - 2, 2):
                measure_qubits_of_symmetry[symmetry_index].append((x, 2 * symmetry_index - 3))
            for y in range(2 * symmetry_index - 1, max_coord + 1, 2):
                measure_qubits_of_symmetry[symmetry_index].append((2 * symmetry_index - 1, y))

        # add the virutal measurement qubits to the real symmetry lines at the beginning and end of the lists
        for symmetry_index in range(1, self.num_logical_qubits + 1):
            measure_qubits_of_symmetry[symmetry_index].insert(
                0, measure_qubits_of_symmetry[0][symmetry_index - 1]
            )
            measure_qubits_of_symmetry[symmetry_index].append(
                measure_qubits_of_symmetry[0][symmetry_index - 1 + self.num_logical_qubits]
            )

        return measure_qubits_of_symmetry

    @property
    def boundary_edges_of_symmetry(self) -> dict[int, dict[tuple, tuple]]:
        """For each real symmetry line, there are two real measurement qubits at the boundary of the symmetry line.
        The boundary edges are the edges connecting the virtual measurement qubits to the real measurement qubits.
        Define for each real measurement qubit at the boundary, which virtual measurement qubit it is connected to.

        Returns:
            dict: {symmetry_index: {real_measurement_qubit: virtual_measurement_qubit}}
        """
        bdry_edges_of_symmetry = {}
        virtual_measure_qubits = self.measurement_qubits_of_symmetry[0]
        for symmetry_index, measure_qubits in self.measurement_qubits_of_symmetry.items():
            if symmetry_index == 0:
                continue
            bdry_edges_of_symmetry[symmetry_index] = {}
            for a, b in itertools.pairwise(measure_qubits):
                if a in virtual_measure_qubits:
                    bdry_edges_of_symmetry[symmetry_index][b] = a
                elif b in virtual_measure_qubits:
                    bdry_edges_of_symmetry[symmetry_index][a] = b
        return bdry_edges_of_symmetry

    @property
    def one_lines(self) -> dict:
        """
        Defines the one lines of the LHZ code, see definition of k-lines in the paper: https://arxiv.org/pdf/2505.05210
        """
        one_lines = {}
        # define the logical observables
        for i in range(self.num_logical_qubits):
            one_lines[i] = []
            for qubit in self.base_qubit_coords + self.parity_qubit_coords:
                if i in self.qubits[qubit]["label"]:
                    one_lines[i].append(qubit)

        one_lines[self.num_logical_qubits] = self.base_qubit_coords

        return one_lines

    def plot_tanner_graph(self):
        """
        Create a plot of the Tanner graph of the LHZ code without any measurement errors.

        The Tanner graph is a bipartite graph with the data qubits and parity qubits (circles) on
        one side and the measurement qubits (squares) on the other side.
        The edges indicate which qubits are involved in a measurement.
        """

        pos_to_q_index = {
            pos: i
            for i, pos in enumerate(
                self.base_qubit_coords + self.parity_qubit_coords + self.measure_qubit_coords
            )
        }

        graph = nx.Graph()
        # add the variable nodes
        variable_node_indices = [
            pos_to_q_index[pos] for pos in self.base_qubit_coords + self.parity_qubit_coords
        ]
        labels = {pos_to_q_index[q]: q_prop["label"] for q, q_prop in self.qubits.items()}
        graph.add_nodes_from(variable_node_indices, bipartite=0)

        # add the check/measure nodes
        check_node_indices = [pos_to_q_index[pos] for pos in self.measure_qubit_coords]
        graph.add_nodes_from(check_node_indices, bipartite=1)

        # add the edges
        for measure_q_coord, qubit_coords in self.constraints.items():
            for q in qubit_coords:
                graph.add_edge(pos_to_q_index[measure_q_coord], pos_to_q_index[q])

        # plot the graph
        pos = nx.bipartite_layout(graph, check_node_indices, align="horizontal")
        nx.draw_networkx_nodes(
            graph, pos, nodelist=variable_node_indices, node_color="lightblue", node_shape="o"
        )
        nx.draw_networkx_nodes(
            graph, pos, nodelist=check_node_indices, node_color="lightgreen", node_shape="s"
        )
        nx.draw_networkx_labels(graph, pos, labels)
        nx.draw_networkx_edges(graph, pos)
        plt.show()

    def plot_code_layout(self):
        """
        Plot the layout of the code.
        Data and parity qubits are shown as circles, measurement qubits as squares.
        """
        graph = nx.Graph()

        pos_to_q_index = {
            pos: i
            for i, pos in enumerate(
                self.base_qubit_coords + self.parity_qubit_coords + self.measure_qubit_coords
            )
        }
        index_to_pos = {i: pos for pos, i in pos_to_q_index.items()}
        nodes = pos_to_q_index.values()
        qubit_nodes = [pos_to_q_index[q] for q in self.base_qubit_coords + self.parity_qubit_coords]
        measure_nodes = [pos_to_q_index[q] for q in self.measure_qubit_coords]
        labels = {i: q_prop["label"] for i, (pos, q_prop) in enumerate(self.qubits.items())}

        # add qubits and checks as nodes
        graph.add_nodes_from(nodes)

        # add edges between qubits and checks
        for measure_q_coord, qubit_coords in self.constraints.items():
            for q in qubit_coords:
                graph.add_edge(pos_to_q_index[measure_q_coord], pos_to_q_index[q])

        # plot the graph
        pos = index_to_pos
        nx.draw_networkx_nodes(graph, pos, qubit_nodes, node_shape="o", node_color="lightblue")
        nx.draw_networkx_nodes(graph, pos, measure_nodes, node_shape="s", node_color="lightgreen")
        nx.draw_networkx_labels(graph, pos, labels)
        nx.draw_networkx_edges(graph, pos)
        plt.show()
