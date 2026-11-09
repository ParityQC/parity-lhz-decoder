# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
import stim

from parity_code.circuits.bit_flip import LhzLegEncoding


class LhzLegEncodingPhaseFlip(LhzLegEncoding):
    """
    Class defining the LHZ error correcting code, where data qubits (qubits with a single index) are placed
    on one leg of the isoscele triangle.

    This version implements a phase-flip protecting Parity code with X-type stabilizers.

    `create_stim_circuit` builds a noiseless syndrome-extraction circuit; use a class from
    `parity_code.noise_models` to add noise to it.
    """

    # use the __init__ method of the parent class LhzLegEncoding
    def __init__(self, num_logical_qubits: int):
        """
        The layout is the same as for LhzLegEncoding, therefore we can use its init function.
        """
        super().__init__(num_logical_qubits)
        # swap the logical x and z observables, since we are now protecting against phase-flip errors
        self.logical_x_observables, self.logical_z_observables = (
            self.logical_z_observables,
            self.logical_x_observables,
        )

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
        # NOTE: target and control are swapped since we are doing a phase-flip code
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
                        CNOT_step_pairs.extend([measure_q_index, data_q_index])  # control, target
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

        # NOTE: we measure the ancillas in the X basis, since we are doing a phase-flip code
        cycle_action.append("MX", measure_indices)
        cycle_action.append("TICK")

        # NOTE: we reset the ancillas in the X basis, since we are doing a phase-flip code
        cycle_action.append("RX", measure_indices)

        ####--HEAD--####################################################
        # Build the start of the circuit, getting a state that's ready to cycle
        # In particular, the first cycle has different detectors and so has to be handled special.
        head = stim.Circuit()
        for index, pos in enumerate(
            self.base_qubit_coords + self.parity_qubit_coords + self.measure_qubit_coords
        ):
            head.append("QUBIT_COORDS", [index], [pos[0], pos[1]])

        # Reset all qubits in the X basis, since we are doing a phase-flip code
        head.append("TICK")
        head.append(
            "RX",
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
            "MX", [pos_to_q_index[q] for q in self.base_qubit_coords + self.parity_qubit_coords]
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

        # Include the logical observables. For each observable only the base qubit is included, since
        # measuring these qubits respectively corresponds to measuring the logical observable.
        for o, observable in enumerate(self.logical_x_observables):
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
