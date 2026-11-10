# parity-lhz-decoder

Code and data for the numerical results of

> K. Tiurev, P. Schnabl, C. Goeller, L. Stenzel, A. Messinger, M. Fellner and W. Lechner,
> *Optimal Decoder for the Error Correcting LHZ Parity Code*,
> [arXiv:2505.05210](https://arxiv.org/abs/2505.05210).

The Python package `parity_code` builds [Stim](https://github.com/quantumlib/Stim) circuits of the
parity (LHZ) code, adds code-capacity, bit-flip circuit-level or cat-qubit circuit-level noise, and
decodes them with the two decoders of the paper: Independent Symmetry Matching (ISM) and Global
Symmetry Matching (GSM, called MWPM in the paper), each with or without the 1-line post-processing
(PP). The matching itself is done by [PyMatching](https://github.com/oscarhiggott/PyMatching). The
`simulations/` folder holds the scripts, the data and the notebooks of all figures.

## Contents

```text
src/parity_code/           the Python package
├── circuits/              noiseless syndrome-extraction circuits
│   ├── bit_flip.py        LhzLegEncoding: Z-type stabilizers, detects bit flips
│   └── phase_flip.py      LhzLegEncodingPhaseFlip: X-type stabilizers, detects phase flips
├── noise_models/
│   ├── code_capacity.py   CodeCapacityNoiseModel
│   ├── generic_cln.py     GenericCircuitLevelNoiseModel: bit-flip circuit-level noise
│   └── cat_qubit_cln.py   CatNoiseModel: phase-flip circuit-level noise of cat qubits
├── decoders/              reference implementation of ISM and GSM, one shot at a time
└── decoders_compiled/     compiled ISM and GSM for batches of shots, used for the paper's results
simulations/
├── code_capacity/         scripts, data and notebooks of the code-capacity figures
├── bit_flip_cln/          scripts, data and notebooks of the circuit-level figures
├── cat_qubit_cln/         script, data and notebook of the cat-qubit figure
└── run_all.sh             regenerates all code-capacity and bit-flip data
tests/                     unit tests
```

## Installation

You need Python 3.13 or newer and a C/C++ compiler, because pip compiles three Cython extensions
during the installation (macOS: Xcode Command Line Tools, `xcode-select --install`; Linux: `gcc` and
`g++`; Windows: Microsoft C++ Build Tools).

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[simulations]"
```

`pip install -e .` installs the package alone, with its dependencies NumPy, Stim, PyMatching,
NetworkX and Matplotlib. The `simulations` extra adds what the scripts and notebooks need (pandas,
multiprocess, SciPy and Jupyter).

Tested on macOS (Apple Silicon) with Python 3.13, Stim 1.15.0, PyMatching 2.2.2, NumPy 2.5,
Cython 3.3, pandas 3.0, SciPy 1.18 and Matplotlib 3.11.2.

## Quick start

Decode a distance-7 code under bit-flip circuit-level noise with GSM (MWPM) and post-processing:

```python
from parity_code import GenericCircuitLevelNoiseModel, LhzLegEncoding
from parity_code.decoders_compiled import CompiledGlobalSymmetryMatchingDecoder

d, p = 7, 0.01

# distance-d parity code (d logical qubits), d rounds of syndrome extraction, bit-flip circuit-level noise
code = LhzLegEncoding(num_logical_qubits=d)
noise_model = GenericCircuitLevelNoiseModel(
    before_round_bit_flip_prob=p,
    before_measure_flip_prob=p,
    after_reset_flip_prob=p,
    p_2qubits_gate=p,
)
circuit = noise_model.add_noise_to_circuit(code.create_stim_circuit(rounds=d), num_rounds=d)
dem = circuit.detector_error_model(approximate_disjoint_errors=True)

decoder = CompiledGlobalSymmetryMatchingDecoder(
    dem,
    data_qubit_coords=code.base_qubit_coords + code.parity_qubit_coords,
    measurement_qubits_of_symmetry=code.measurement_qubits_of_symmetry,
    boundary_edges_of_real_symmetry=code.boundary_edges_of_symmetry,
    one_lines=code.one_lines,
    base_qubit_coords=code.base_qubit_coords,
)

syndromes, observables = circuit.compile_detector_sampler().sample(10_000, separate_observables=True)
predictions = decoder.decode_batch(syndromes, post_processing=True)
print("code failure rate:", (predictions != observables).any(axis=1).mean())
```

`decode_batch` returns the predicted flips of the d logical observables, which are measured on the
base qubits. A shot fails if any of them is wrong; the fraction of failed shots is the code failure
rate of the paper. `decode_batch_to_corrections` returns the full corrections instead.
`CompiledIndependentSymmetryMatchingDecoder` takes the same arguments.

The other noise models work the same way. For code-capacity noise, use a single round:

```python
from parity_code import CodeCapacityNoiseModel

circuit = CodeCapacityNoiseModel(bit_flip_prob=0.1).add_noise_to_circuit(code.create_stim_circuit(rounds=1))
```

For cat qubits, use the phase-flip code and build the decoder from `cat_code`:

```python
from parity_code import CatNoiseModel, LhzLegEncodingPhaseFlip

cat_code = LhzLegEncodingPhaseFlip(num_logical_qubits=d)
circuit = CatNoiseModel(nbar=11, kappa1_kappa2_ratio=1e-4).add_noise_to_circuit(
    cat_code.create_stim_circuit(rounds=d), num_rounds=d
)
```

[`circuit_description.ipynb`](src/parity_code/circuits/circuit_description.ipynb) walks through the
layout and the circuits.

## Noise models

- **Code capacity** (`CodeCapacityNoiseModel`): an X error with probability p on every data and
  parity qubit, followed by one perfect round of syndrome extraction.
- **Bit-flip circuit-level noise** (`GenericCircuitLevelNoiseModel`):
  d rounds of syndrome extraction with an X error with probability p after every
  reset, on every data and parity qubit at the start of every round and before every measurement
  (including the final readout of the data and parity qubits), and IX, XI or XX with probability p/3
  each after every CNOT.
- **Cat-qubit circuit-level noise** (`CatNoiseModel`): the phase-flip errors of Table III(c) of
  [arXiv:2401.09541](https://arxiv.org/abs/2401.09541), with n̄ = 11 and the same duration 1/κ₂ for
  all operations, including idle errors. There are d noisy rounds, followed by a perfect readout of
  the data and parity qubits.

## Reproducing the results

Every figure folder contains a notebook that creates the figure from the CSV files in `data/`, or
from those of the folder it refers to. The notebook also lists the commands that
generated the data.

Run `bash simulations/run_all.sh` from the project root to rerun all code-capacity and bit-flip simulations with the commands of
the notebooks. This overwrites the CSV files in the `data/` folders.

## How to cite

If you use this code, please cite:

```bibtex
@misc{tiurev2025optimaldecodererrorcorrecting,
      title={Optimal Decoder for the Error Correcting LHZ Parity Code}, 
      author={Konstantin Tiurev and Paul Schnabl and Christophe Goeller and Leo Stenzel and Anette Messinger and Michael Fellner and Wolfgang Lechner},
      year={2025},
      eprint={2505.05210},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2505.05210}, 
}
```

## License

See [LICENSE](LICENSE.txt).

## Contact

Paul Schnabl, Institute for Theoretical Physics, University of Innsbruck,
paul.schnabl@uibk.ac.at

Konstantin Tiurev, Parity Quantum Computing Germany GmbH, 
k.tiurev@parityqc.com