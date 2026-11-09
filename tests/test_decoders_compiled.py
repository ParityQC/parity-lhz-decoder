from collections import Counter, defaultdict

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
from parity_code.decoders.common import remove_even_occurrences
from parity_code.decoders_compiled import (
    CompiledGlobalSymmetryMatchingDecoder,
    CompiledIndependentSymmetryMatchingDecoder,
)

DECODER_PAIRS = [
    (GlobalSymmetryMatchingDecoder, CompiledGlobalSymmetryMatchingDecoder),
    (IndependentSymmetryMatchingDecoder, CompiledIndependentSymmetryMatchingDecoder),
]


def _bit_flip_cln(d, p):
    code = LhzLegEncoding(num_logical_qubits=d)
    circuit = GenericCircuitLevelNoiseModel(p, p, p, p).add_noise_to_circuit(
        code.create_stim_circuit(rounds=d), num_rounds=d
    )
    return code, circuit


def _code_capacity(d, p):
    code = LhzLegEncoding(num_logical_qubits=d)
    circuit = CodeCapacityNoiseModel(bit_flip_prob=p).add_noise_to_circuit(
        code.create_stim_circuit(rounds=1)
    )
    return code, circuit


def _cat(d, kappa1_kappa2_ratio):
    code = LhzLegEncodingPhaseFlip(num_logical_qubits=d)
    circuit = CatNoiseModel(nbar=11, kappa1_kappa2_ratio=kappa1_kappa2_ratio).add_noise_to_circuit(
        code.create_stim_circuit(rounds=d), num_rounds=d
    )
    return code, circuit


SETTINGS = [
    pytest.param(_bit_flip_cln, 5, 0.01, id="bit-flip-cln-d5"),
    pytest.param(_bit_flip_cln, 7, 0.008, id="bit-flip-cln-d7"),
    # d >= 11: more than 64 data qubits, i.e. multi-word bit masks in the compiled post-processing
    pytest.param(_bit_flip_cln, 11, 0.008, id="bit-flip-cln-d11"),
    pytest.param(_code_capacity, 7, 0.1, id="code-capacity-d7"),
    pytest.param(_cat, 5, 0.001, id="cat-d5"),
    pytest.param(_cat, 7, 0.0005, id="cat-d7"),
]


def _decoder_args(code, circuit):
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    return dict(
        dem=dem,
        data_qubit_coords=code.base_qubit_coords + code.parity_qubit_coords,
        measurement_qubits_of_symmetry=code.measurement_qubits_of_symmetry,
        boundary_edges_of_real_symmetry=code.boundary_edges_of_symmetry,
        one_lines=code.one_lines,
    )


def _original_logical_flips(decoder, code, syndromes):
    """Base-qubit part of the corrections of an original decoder: its predicted logical flips."""
    flips = np.zeros((len(syndromes), code.num_logical_qubits), dtype=bool)
    for i, syndrome in enumerate(syndromes):
        for coord in decoder.decode(syndrome, post_processing=False):
            if coord in code.base_qubit_coords:
                flips[i, code.base_qubit_coords.index(coord)] = True
    return flips


@pytest.mark.parametrize("original_cls, compiled_cls", DECODER_PAIRS)
@pytest.mark.parametrize("make_circuit, d, p", SETTINGS)
def test_compiled_decoder_reproduces_original(original_cls, compiled_cls, make_circuit, d, p):
    """The compiled decoders read the logical flips off PyMatching's fault IDs; they must predict
    exactly the same logical flips as the original decoders without post-processing, shot by
    shot."""
    code, circuit = make_circuit(d, p)
    args = _decoder_args(code, circuit)
    original = original_cls(**args)
    compiled = compiled_cls(**args, base_qubit_coords=code.base_qubit_coords)

    syndromes, _ = circuit.compile_detector_sampler(seed=2024).sample(
        shots=1000, separate_observables=True
    )
    expected = _original_logical_flips(original, code, syndromes)
    assert np.array_equal(compiled.decode_batch(syndromes), expected)
    assert np.array_equal(compiled.decode(syndromes[0]), expected[0])


@pytest.mark.parametrize("post_processing", [False, True])
@pytest.mark.parametrize("_, compiled_cls", DECODER_PAIRS)
def test_compiled_decoder_noiseless_syndrome(_, compiled_cls, post_processing):
    """An all-zero syndrome must not flip any logical observable."""
    code = LhzLegEncoding(num_logical_qubits=5)
    circuit = code.create_stim_circuit(rounds=5)
    compiled = compiled_cls(
        **_decoder_args(code, circuit), base_qubit_coords=code.base_qubit_coords
    )
    syndromes = np.zeros((3, circuit.num_detectors), dtype=bool)
    assert not compiled.decode_batch(syndromes, post_processing=post_processing).any()


@pytest.mark.parametrize("original_cls, compiled_cls", DECODER_PAIRS)
@pytest.mark.parametrize("make_circuit, d, p", SETTINGS)
def test_compiled_corrections_without_post_processing_equal_original(
    original_cls, compiled_cls, make_circuit, d, p
):
    """Without post-processing, the full corrections of the compiled decoders (XOR of the flips of
    all matched edges) must be those of the original decoders, shot by shot, and their base-qubit
    part must be the logical flips read off the fault IDs."""
    code, circuit = make_circuit(d, p)
    args = _decoder_args(code, circuit)
    original = original_cls(**args)
    compiled = compiled_cls(**args, base_qubit_coords=code.base_qubit_coords)

    syndromes, _ = circuit.compile_detector_sampler(seed=2026).sample(
        shots=300, separate_observables=True
    )
    corrections = compiled.decode_batch_to_corrections(syndromes, post_processing=False)
    data_qubit_coords = args["data_qubit_coords"]
    for syndrome, correction in zip(syndromes, corrections):
        expected = set(original.decode(syndrome, post_processing=False))
        assert {data_qubit_coords[i] for i in np.nonzero(correction)[0]} == expected

    base_indices = [data_qubit_coords.index(coord) for coord in code.base_qubit_coords]
    assert np.array_equal(corrections[:, base_indices], compiled.decode_batch(syndromes))


def _reference_component_post_processing(decoder, syndrome):
    """Plain-Python version of the post-processing by space-time connected components (see
    `ComponentPostProcessing`), applied to the matched edges of an original decoder. It uses the
    original ray casting on whole components and the original post-processing of time slices."""
    matches = decoder._match(syndrome)
    virtual = set(decoder.measurement_qubits_of_symmetry[0])

    parent = {}

    def find(node):
        while parent[node] != node:
            node = parent[node]
        return node

    for a, b in matches:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        parent[find(a)] = find(b)
    components = defaultdict(list)
    for a, b in matches:
        components[find(a)].append((a, b))

    corrections = defaultdict(list)
    for edges in components.values():
        interior = decoder._encircled_nodes_finder.find_encircled_nodes_from_matches(edges)
        if not interior:
            continue
        votes = Counter()
        for a, b in edges:
            if a[:2] != b[:2] and not (a[:2] in virtual and b[:2] in virtual):
                votes[a[2]] += 1
                votes[b[2]] += 1
        most_votes = max(votes.values())
        corrections[min(t for t, n in votes.items() if n == most_votes)].extend(interior)
    return set(remove_even_occurrences(decoder.correction_postprocessing_ftp(dict(corrections))))


@pytest.mark.parametrize("original_cls, compiled_cls", DECODER_PAIRS)
@pytest.mark.parametrize("make_circuit, d, p", SETTINGS)
def test_compiled_post_processing_reproduces_reference(
    original_cls, compiled_cls, make_circuit, d, p
):
    """The compiled post-processing must give exactly the corrections of the plain-Python reference
    (on the matched edges of the original decoder), shot by shot, and its logical flips must be the
    base-qubit part of these corrections."""
    code, circuit = make_circuit(d, p)
    args = _decoder_args(code, circuit)
    original = original_cls(**args)
    compiled = compiled_cls(**args, base_qubit_coords=code.base_qubit_coords)

    syndromes, _ = circuit.compile_detector_sampler(seed=2025).sample(
        shots=300, separate_observables=True
    )
    corrections = compiled.decode_batch_to_corrections(syndromes)
    data_qubit_coords = args["data_qubit_coords"]
    for syndrome, correction in zip(syndromes, corrections):
        expected = _reference_component_post_processing(original, syndrome)
        assert {data_qubit_coords[i] for i in np.nonzero(correction)[0]} == expected

    base_indices = [data_qubit_coords.index(coord) for coord in code.base_qubit_coords]
    flips = compiled.decode_batch(syndromes, post_processing=True)
    assert np.array_equal(flips, corrections[:, base_indices])
    assert np.array_equal(compiled.decode(syndromes[0], post_processing=True), flips[0])


@pytest.mark.parametrize("original_cls, compiled_cls", DECODER_PAIRS)
def test_compiled_post_processing_equals_original_in_one_time_slice(original_cls, compiled_cls):
    """Under code-capacity noise all matched edges lie in one time slice, where the grouping of the
    edges does not matter: the compiled post-processing must reproduce the original one exactly."""
    code, circuit = _code_capacity(13, 0.12)
    args = _decoder_args(code, circuit)
    original = original_cls(**args)
    compiled = compiled_cls(**args, base_qubit_coords=code.base_qubit_coords)

    syndromes, _ = circuit.compile_detector_sampler(seed=7).sample(
        shots=300, separate_observables=True
    )
    corrections = compiled.decode_batch_to_corrections(syndromes)
    data_qubit_coords = args["data_qubit_coords"]
    for syndrome, correction in zip(syndromes, corrections):
        expected = set(original.decode(syndrome, post_processing=True))
        assert {data_qubit_coords[i] for i in np.nonzero(correction)[0]} == expected


def _reference_cycles(correction, one_lines):
    """Number of cycles of the post-processing of one time slice (Algorithm 1 of the paper), counted
    with numpy on a boolean correction and a boolean matrix of 1-lines (one row per 1-line)."""
    correction = correction.copy()
    cycles = 0
    while True:
        overlaps = (one_lines & correction).sum(axis=1)
        m = np.argmax(overlaps)
        if 2 * overlaps[m] < one_lines[m].sum() + 1:
            return cycles
        correction ^= one_lines[m]
        cycles += 1


@pytest.mark.parametrize("d", [7, 13])
def test_compiled_slice_post_processing_equals_original(d):
    """The compiled post-processing of a single time slice must agree with
    `correction_postprocessing` of the original decoders on arbitrary corrections, including
    corrections that need several 1-line multiplications and ties between 1-lines, and must count
    the cycles (1-line multiplications) correctly."""
    code, circuit = _code_capacity(d, 0.1)
    args = _decoder_args(code, circuit)
    original = GlobalSymmetryMatchingDecoder(**args)
    compiled = CompiledGlobalSymmetryMatchingDecoder(
        **args, base_qubit_coords=code.base_qubit_coords
    )
    data_qubit_coords = args["data_qubit_coords"]
    one_lines = np.array(
        [[q in line for q in data_qubit_coords] for line in code.one_lines.values()], dtype=bool
    )

    rng = np.random.default_rng(11)
    corrections = rng.random((2000, len(data_qubit_coords))) < rng.random((2000, 1))
    processed, cycles = compiled.component_post_processing.post_process_slices(corrections)
    assert cycles.max() >= 2
    for correction, result, num_cycles in zip(corrections, processed, cycles):
        expected = original.correction_postprocessing(
            [data_qubit_coords[i] for i in np.nonzero(correction)[0]]
        )
        assert {data_qubit_coords[i] for i in np.nonzero(result)[0]} == set(expected)
        assert num_cycles == _reference_cycles(correction, one_lines)
