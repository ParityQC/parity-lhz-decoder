# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
import stim


def qubit_targets(instruction: stim.CircuitInstruction) -> list[int]:
    """Return the ordinary qubit targets of a Stim instruction."""
    return [target.qubit_value for target in instruction.targets_copy() if target.is_qubit_target]


def measurement_qubit_targets(circuit: stim.Circuit) -> list[int]:
    """
    Qubits targeted by the first measurement in the circuit — for the parity-code circuits in
    `parity_code.circuits`, that is always the ancilla qubits measured every round of syndrome
    extraction (the data/parity qubits are only ever measured once, at the very end).
    """
    for instruction in circuit:
        if instruction.name in ("M", "MX"):
            return qubit_targets(instruction)
    raise ValueError("Circuit contains no measurement instruction.")
