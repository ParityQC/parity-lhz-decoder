import numpy as np
import pytest

from parity_code import (
    CatNoiseModel,
    CodeCapacityNoiseModel,
    GenericCircuitLevelNoiseModel,
    GlobalSymmetryMatchingDecoder,
    IndependentSymmetryMatchingDecoder,
    LhzLegEncoding,
    LhzLegEncodingPhaseFlip,
)

DECODER_CLASSES = [IndependentSymmetryMatchingDecoder, GlobalSymmetryMatchingDecoder]


def _make_decoder(decoder_cls, code, dem):
    return decoder_cls(
        dem,
        data_qubit_coords=code.base_qubit_coords + code.parity_qubit_coords,
        measurement_qubits_of_symmetry=code.measurement_qubits_of_symmetry,
        boundary_edges_of_real_symmetry=code.boundary_edges_of_symmetry,
        one_lines=code.one_lines,
    )


@pytest.mark.parametrize("decoder_cls", DECODER_CLASSES)
@pytest.mark.parametrize("post_processing", [False, True])
def test_noiseless_circuit_decodes_to_no_correction(decoder_cls, post_processing):
    """A noiseless circuit's syndrome is always all-zero, so both decoders should suggest no
    correction at all."""
    code = LhzLegEncoding(num_logical_qubits=5)
    circuit = code.create_stim_circuit(rounds=5)
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    decoder = _make_decoder(decoder_cls, code, dem)

    syndrome = np.zeros(dem.num_detectors, dtype=bool)
    assert decoder.decode(syndrome, post_processing=post_processing) == []


@pytest.mark.parametrize("decoder_cls", DECODER_CLASSES)
@pytest.mark.parametrize("post_processing", [False, True])
def test_noisy_circuit_decodes_without_error(decoder_cls, post_processing):
    """Both decoders should run end-to-end on sampled noisy syndromes and return a list of data-qubit
    coordinates."""
    code = LhzLegEncoding(num_logical_qubits=5)
    ideal_circuit = code.create_stim_circuit(rounds=5)
    noise_model = GenericCircuitLevelNoiseModel(
        before_round_bit_flip_prob=0.01,
        before_measure_flip_prob=0.01,
        after_reset_flip_prob=0.01,
        p_2qubits_gate=0.01,
    )
    circuit = noise_model.add_noise_to_circuit(ideal_circuit, num_rounds=5)
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    decoder = _make_decoder(decoder_cls, code, dem)

    all_qubit_coords = set(code.base_qubit_coords + code.parity_qubit_coords)
    sampler = circuit.compile_detector_sampler(seed=1234)
    syndromes, _ = sampler.sample(shots=50, separate_observables=True)

    for syndrome in syndromes:
        correction = decoder.decode(syndrome, post_processing=post_processing)
        assert isinstance(correction, list)
        assert set(correction) <= all_qubit_coords


@pytest.mark.parametrize("decoder_cls", DECODER_CLASSES)
@pytest.mark.parametrize("post_processing", [False, True])
def test_single_qubit_error_is_corrected_exactly(decoder_cls, post_processing):
    """Under code-capacity noise, a single X error on any data qubit must be corrected by flipping
    exactly that qubit. This covers the ray casting for every kind of qubit, including the qubits
    that a diagonal edge passes through: those on the diagonal of the triangle and the corner base
    qubit."""
    code = LhzLegEncoding(num_logical_qubits=5)
    circuit = CodeCapacityNoiseModel(bit_flip_prob=0.05).add_noise_to_circuit(
        code.create_stim_circuit(rounds=1)
    )
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    decoder = _make_decoder(decoder_cls, code, dem)

    # an error before the single round flips the first-round detectors of its stabilizers
    detector_of_stabilizer = {
        (int(x), int(y)): d for d, (x, y, t) in dem.get_detector_coordinates().items() if t == 0
    }
    for qubit in code.base_qubit_coords + code.parity_qubit_coords:
        syndrome = np.zeros(dem.num_detectors, dtype=bool)
        for stabilizer, qubits in code.constraints.items():
            if qubit in qubits:
                syndrome[detector_of_stabilizer[stabilizer]] = True
        assert decoder.decode(syndrome, post_processing=post_processing) == [qubit]


def _shared_qubit(code, i, j):
    """The qubit shared by 1-lines i and j (every two 1-lines share exactly one qubit)."""
    (qubit,) = set(code.one_lines[i]) & set(code.one_lines[j])
    return qubit


@pytest.mark.parametrize(
    "correction_lines, expected_lines",
    [
        # 1-line 1 has the largest overlap (4 of 5 qubits) and is flipped, although 1-line 0 comes
        # first and is covered at least half as well (3 of 5 qubits)
        (
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4), (2, 5)],
            [(0, 2), (0, 3), (1, 5), (2, 5)],
        ),
        # 1-lines 0 and 1 tie with the largest overlap (3 of 5 qubits); the first one is flipped
        (
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)],
            [(0, 4), (0, 5), (1, 2), (1, 3)],
        ),
    ],
)
def test_post_processing_flips_one_line_with_largest_overlap(correction_lines, expected_lines):
    """Post-processing (Algorithm 1 of the paper) flips the 1-line with the largest overlap with the
    correction, the first one on ties, as long as that 1-line is covered at least half. Every qubit
    lies on exactly two 1-lines, so qubits are given by the pair of 1-lines they lie on. The
    post-processing is shared by both decoders."""
    code = LhzLegEncoding(num_logical_qubits=5)
    dem = code.create_stim_circuit(rounds=1).detector_error_model(approximate_disjoint_errors=True)
    decoder = _make_decoder(GlobalSymmetryMatchingDecoder, code, dem)

    correction = [_shared_qubit(code, i, j) for i, j in correction_lines]
    expected = {_shared_qubit(code, i, j) for i, j in expected_lines}
    assert set(decoder.correction_postprocessing(correction)) == expected


@pytest.mark.parametrize("decoder_cls", DECODER_CLASSES)
def test_phase_flip_cat_noise_decodes_without_error(decoder_cls):
    """The phase-flip circuit under the cat-qubit noise model should also decode end-to-end without
    error. (X-type noise models like CodeCapacityNoiseModel don't apply here: X errors on data qubits
    commute with the X-type stabilizers of the phase-flip circuit and would never be detected.)"""
    code = LhzLegEncodingPhaseFlip(num_logical_qubits=5)
    ideal_circuit = code.create_stim_circuit(rounds=5)
    noise_model = CatNoiseModel(nbar=11, kappa1_kappa2_ratio=0.001)
    circuit = noise_model.add_noise_to_circuit(ideal_circuit, num_rounds=5)
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    decoder = _make_decoder(decoder_cls, code, dem)

    all_qubit_coords = set(code.base_qubit_coords + code.parity_qubit_coords)
    sampler = circuit.compile_detector_sampler(seed=1234)
    syndromes, _ = sampler.sample(shots=50, separate_observables=True)

    for syndrome in syndromes:
        correction = decoder.decode(syndrome, post_processing=True)
        assert isinstance(correction, list)
        assert set(correction) <= all_qubit_coords
