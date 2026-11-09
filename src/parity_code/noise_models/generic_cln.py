# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
import stim

from parity_code.noise_models.common import measurement_qubit_targets, qubit_targets


class GenericCircuitLevelNoiseModel:
    """
    Generic circuit-level bit-flip noise model.

    Parameters
    ----------
    before_round_bit_flip_prob:
        Probability of an independent X error on each data/parity qubit at the start of every round
        (i.e. right after the preceding reset).

    before_measure_flip_prob:
        Probability of an independent X error on a qubit immediately before it is measured — applied
        to both the per-round ancilla measurements and the final data/parity measurement.

    after_reset_flip_prob:
        Probability of an independent X error on a qubit immediately after it is reset — applied to
        both the initial reset and every per-round ancilla reset.

    p_2qubits_gate:
        Total probability of an X-type error on a CX gate, split evenly between "X on the first
        qubit", "X on the second qubit", and "X on both" (i.e. p/3 each).
    """

    def __init__(
        self,
        before_round_bit_flip_prob: float = 0,
        before_measure_flip_prob: float = 0,
        after_reset_flip_prob: float = 0,
        p_2qubits_gate: float = 0,
    ):
        self.before_round_bit_flip_prob = float(before_round_bit_flip_prob)
        self.before_measure_flip_prob = float(before_measure_flip_prob)
        self.after_reset_flip_prob = float(after_reset_flip_prob)
        self.p_2qubits_gate = float(p_2qubits_gate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append_x_noise(circuit: stim.Circuit, qubits: list[int], probability: float) -> None:
        """Append independent X noise if the target list is non-empty."""
        if qubits and probability > 0:
            circuit.append("X_ERROR", qubits, probability)

    def _two_qubit_x_channel(self) -> list[float]:
        """
        PAULI_CHANNEL_2 probabilities for an X-type-only two-qubit channel, split evenly between
        IX, XI and XX.

        PAULI_CHANNEL_2 ordering: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ.
        """
        p = self.p_2qubits_gate / 3
        return [p, 0.0, 0.0, p, p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # ------------------------------------------------------------------
    # Main routine
    # ------------------------------------------------------------------

    def add_noise_to_circuit(self, ideal: stim.Circuit, num_rounds: int) -> stim.Circuit:
        """
        Add the generic circuit-level bit-flip noise model to an ideal Stim circuit.

        Parameters
        ----------
        ideal:
            Ideal Stim circuit, built with the given number of rounds.

        num_rounds:
            Number of noisy syndrome-extraction rounds (equal to the `rounds` the ideal circuit was
            built with).

        Returns
        -------
        stim.Circuit
            Noisy circuit.
        """
        ideal_circuit = ideal.flattened()

        all_qubits = sorted(ideal_circuit.get_final_qubit_coordinates().keys())
        data_qubits = sorted(set(all_qubits) - set(measurement_qubit_targets(ideal_circuit)))

        noisy = stim.Circuit()
        measurements_seen = 0

        for instruction in ideal_circuit:
            name = instruction.name

            if name in {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}:
                noisy.append(instruction)
                continue

            used_qubits = qubit_targets(instruction)

            if name == "R":
                noisy.append(instruction)
                self._append_x_noise(noisy, used_qubits, self.after_reset_flip_prob)

                # A new round of syndrome extraction always immediately follows a reset — except
                # after the very last round's ancilla reset, since no further round follows it.
                if measurements_seen < num_rounds:
                    self._append_x_noise(noisy, data_qubits, self.before_round_bit_flip_prob)
                continue

            if name == "CX":
                noisy.append(instruction)
                if self.p_2qubits_gate > 0:
                    noisy.append("PAULI_CHANNEL_2", used_qubits, self._two_qubit_x_channel())
                continue

            if name == "M":
                self._append_x_noise(noisy, used_qubits, self.before_measure_flip_prob)
                noisy.append(instruction)
                measurements_seen += 1
                continue

            raise ValueError(f"Unsupported instruction in GenericCircuitLevelNoiseModel: {instruction}")

        # num_rounds per-round ancilla measurements, plus one final data/parity measurement.
        if measurements_seen != num_rounds + 1:
            raise ValueError(
                "Number of measurements encountered does not match num_rounds: saw "
                f"{measurements_seen}, expected {num_rounds + 1} (num_rounds + 1)."
            )

        return noisy
