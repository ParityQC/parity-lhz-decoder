# ParityQC © 2020-2026. See the LICENSE file in the top level directory for details.
import math
import stim

from parity_code.noise_models.common import qubit_targets


class CatNoiseModel:
    """
    Phase-flip-only cat-qubit circuit-level noise model based on
    Table III(c) of arXiv:2401.09541.

    The model assumes

        T_prep = T_meas = T_CX = 1 / kappa_2

    and uses

        r = kappa_1 / kappa_2.

    Only Z-type errors are included.

    Noise probabilities
    -------------------
    Preparation in the X basis:
        p_RX = nbar * r

    X-basis measurement:
        p_MX = nbar * r

    Idle:
        p_idle = nbar * r

    CX:
        Z_control:
            p_Zc = nbar * r + pi^2 / (64 * nbar)

        Z_target:
            p_Zt = 0.5 * nbar * r

        Z_control Z_target:
            p_ZcZt = 0.5 * nbar * r


    Circuit convention
    ------------------
    The supplied parity-code circuit has:

        1. Initial RX on all qubits.
        2. d noisy syndrome-extraction rounds.
        3. A final MX measurement of the data/parity qubits.

    The final data measurement is treated as PERFECT, as in the
    circuit-level simulations of arXiv:2401.09541. It therefore receives
    neither measurement noise nor an additional idle-noise timestep.

    During every noisy operation timestep, all qubits not participating
    in the operation receive idle noise.

    Parameters
    ----------
    nbar:
        Average photon number.

    kappa1_kappa2_ratio:
        Ratio kappa_1 / kappa_2.
    """

    def __init__(self, nbar: float, kappa1_kappa2_ratio: float):
        self.nbar = float(nbar)
        self.r = float(kappa1_kappa2_ratio)

    @property
    def p_RX(self) -> float:
        """Phase-flip probability associated with X-basis preparation."""
        return self.nbar * self.r

    @property
    def p_MX(self) -> float:
        """Phase-flip probability associated with X-basis measurement."""
        return self.nbar * self.r

    @property
    def p_CX_control(self) -> float:
        """Probability of a Z error on the CX control."""
        return (
            self.nbar * self.r
            + (math.pi**2) / (64.0 * self.nbar)
        )

    @property
    def p_CX_target(self) -> float:
        """Probability of a Z error on the CX target."""
        return 0.5 * self.nbar * self.r

    @property
    def p_CX_correlated(self) -> float:
        """Probability of a correlated Z_control Z_target error."""
        return 0.5 * self.nbar * self.r

    @property
    def p_idle(self) -> float:
        """Phase-flip probability during one idle timestep."""
        return self.nbar * self.r

    @staticmethod
    def _append_z_noise(
        circuit: stim.Circuit,
        qubits: list[int],
        probability: float,
    ) -> None:
        """Append independent Z noise if the target list is non-empty."""
        if qubits and probability > 0:
            circuit.append(
                "PAULI_CHANNEL_1",
                qubits,
                [0.0, 0.0, probability],
            )

    def _append_idle_noise(
        self,
        circuit: stim.Circuit,
        idle_qubits: list[int],
    ) -> None:
        """Append one timestep of idle Z noise."""
        self._append_z_noise(
            circuit,
            idle_qubits,
            self.p_idle,
        )
        
    def add_noise_to_circuit(
        self,
        ideal: stim.Circuit,
        num_rounds: int,
    ) -> stim.Circuit:
        """
        Add the phase-flip cat-qubit noise model to an ideal Stim circuit.

        The routine is tailored to the circuit structure produced by
        LhzLegEncodingPhaseFlip.create_stim_circuit():

            initial RX
            [syndrome extraction] x num_rounds
            final perfect MX of data/parity qubits

        The final data measurement is intentionally kept perfect.

        Parameters
        ----------
        ideal:
            Ideal Stim circuit.

        num_rounds:
            Number of noisy QEC rounds. For the simulations considered
            here this is normally equal to the code distance d.

        Returns
        -------
        stim.Circuit
            Noisy circuit.
        """

        ideal_circuit = ideal.flattened()

        all_qubits = sorted(
            ideal_circuit.get_final_qubit_coordinates().keys()
        )

        noisy = stim.Circuit()

        syndrome_measurements_seen = 0

        for instruction in ideal_circuit:
            name = instruction.name
            
            if name in {
                "TICK",
                "DETECTOR",
                "OBSERVABLE_INCLUDE",
                "QUBIT_COORDS",
                "SHIFT_COORDS",
            }:
                noisy.append(instruction)
                continue

            used_qubits = qubit_targets(instruction)
            used_set = set(used_qubits)

            idle_qubits = [
                q for q in all_qubits
                if q not in used_set
            ]

            if name == "RX":
                noisy.append(instruction)

                # RX operations belonging to the noisy portion of the
                # experiment receive preparation noise.
                #
                # The final RX after the last syndrome measurement is
                # irrelevant because those ancillas are never used again.
                # We leave it perfect.
                if syndrome_measurements_seen < num_rounds:
                    self._append_z_noise(
                        noisy,
                        used_qubits,
                        self.p_RX,
                    )

                    # All other qubits idle while these qubits are
                    # prepared.
                    self._append_idle_noise(
                        noisy,
                        idle_qubits,
                    )

                continue

            if name == "CX":
                noisy.append(instruction)
                
                # Stim target ordering:
                #     IZ -> Z on target
                #     ZI -> Z on control
                #     ZZ -> correlated Zc Zt

                p_CX = [
                    0.0,  # IX
                    0.0,  # IY
                    self.p_CX_target,      # IZ
                    0.0,  # XI
                    0.0,  # XX
                    0.0,  # XY
                    0.0,  # XZ
                    0.0,  # YI
                    0.0,  # YX
                    0.0,  # YY
                    0.0,  # YZ
                    self.p_CX_control,     # ZI
                    0.0,  # ZX
                    0.0,  # ZY
                    self.p_CX_correlated, # ZZ
                ]

                if syndrome_measurements_seen < num_rounds:
                    noisy.append(
                        "PAULI_CHANNEL_2",
                        used_qubits,
                        p_CX,
                    )
                    # Every qubit not participating in this CNOT layer
                    # idles for one CX timestep.
                    self._append_idle_noise(
                        noisy,
                        idle_qubits,
                    )
                continue

            if name == "MX":
                if syndrome_measurements_seen < num_rounds:
                    # Noisy syndrome-ancilla measurement.
                    #
                    # Measurement noise is represented as a Z error
                    # immediately before the ideal X measurement.
                    self._append_z_noise(
                        noisy,
                        used_qubits,
                        self.p_MX,
                    )

                    # All other qubits idle while the ancillas are
                    # measured.
                    self._append_idle_noise(
                        noisy,
                        idle_qubits,
                    )

                    noisy.append(instruction)

                    syndrome_measurements_seen += 1

                else:
                    # Final data/parity MX.
                    #
                    # This implements the perfect final syndrome/readout
                    # used in the Ruiz et al. circuit-level simulations.
                    # No measurement noise and no additional idle noise
                    # are inserted here.
                    noisy.append(instruction)
                    
                continue

            raise ValueError(
                f"Unsupported instruction in CatNoiseModel: {instruction}"
            )

        if syndrome_measurements_seen != num_rounds:
            raise ValueError(
                "Number of syndrome measurements encountered does not "
                f"match num_rounds: saw {syndrome_measurements_seen}, "
                f"expected {num_rounds}."
            )

        return noisy
