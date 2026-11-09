# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
import stim

from parity_code.noise_models.common import measurement_qubit_targets


class CodeCapacityNoiseModel:
    """
    Code-capacity noise model: an independent bit-flip error on every data/parity qubit right after
    they are reset, followed by a single, otherwise perfect round of syndrome extraction.

    For `parity_code.circuits.LhzLegEncoding` (Z-type stabilizers): X errors on the data qubits are
    exactly what those stabilizers detect. This model isn't meaningful for the phase-flip circuit
    (`LhzLegEncodingPhaseFlip`), whose X-type stabilizers commute with X errors and would never
    detect them.

    Parameters
    ----------
    bit_flip_prob:
        Probability of an independent X error on each data/parity qubit.
    """

    def __init__(self, bit_flip_prob: float):
        self.bit_flip_prob = float(bit_flip_prob)

    def add_noise_to_circuit(self, ideal: stim.Circuit) -> stim.Circuit:
        """
        Add code-capacity noise to an ideal, single-round Stim circuit: every instruction is passed
        through unchanged, except that one X_ERROR layer at `bit_flip_prob` is inserted on the
        data/parity qubits immediately after the initial reset.
        """
        ideal_circuit = ideal.flattened()

        all_qubits = sorted(ideal_circuit.get_final_qubit_coordinates().keys())
        data_qubits = sorted(set(all_qubits) - set(measurement_qubit_targets(ideal_circuit)))

        noisy = stim.Circuit()
        noise_injected = False
        for instruction in ideal_circuit:
            noisy.append(instruction)
            if instruction.name == "R" and not noise_injected:
                # The initial reset of every qubit — inject the one-shot data-qubit bit flip right
                # after it, before the single round of syndrome extraction that follows.
                if self.bit_flip_prob > 0:
                    noisy.append("X_ERROR", data_qubits, self.bit_flip_prob)
                noise_injected = True

        if not noise_injected:
            raise ValueError("Circuit contains no reset instruction.")

        return noisy
