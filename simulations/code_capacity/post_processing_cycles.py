"""
Average number of post-processing cycles needed for convergence, with the compiled decoders.

A cycle is one execution of the while-loop of the post-processing: the
1-line with the largest overlap with the correction is flipped, as long as this overlap is at least
half of the qubits of the 1-line. This is the post-processing of
`SymmetryMatchingDecoderBase.correction_postprocessing` (ties are broken the same way); the compiled
post-processing (`ComponentPostProcessing.post_process_slices`) counts the cycles.

Example usage:
python simulations/code_capacity/post_processing_cycles.py --decoder gsm --distances 7 11 15 19 23 27 31 35 39 --error-rates 0.1 0.21 0.05 --shots 50000
python simulations/code_capacity/post_processing_cycles.py --decoder ism --distances 7 11 15 19 23 27 31 35 39 --error-rates 0.1 0.21 0.05 --shots 50000
"""

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocess.pool import Pool

from parity_code import CodeCapacityNoiseModel, LhzLegEncoding
from parity_code.decoders_compiled import (
    CompiledGlobalSymmetryMatchingDecoder,
    CompiledIndependentSymmetryMatchingDecoder,
)

RESULTS_DIR = Path(__file__).parent / "results"
DECODERS = {
    "gsm": CompiledGlobalSymmetryMatchingDecoder,
    "ism": CompiledIndependentSymmetryMatchingDecoder,
}

# number of shots sampled and decoded at once (bounds memory)
BATCH_SHOTS = 10_000


def run_single_simulation(args: tuple[str, int, float, int]) -> tuple[str, int, float, float, float, int]:
    decoder_name, distance, p, num_shots = args
    code = LhzLegEncoding(num_logical_qubits=distance)
    ideal_circuit = code.create_stim_circuit(rounds=1)
    circuit = CodeCapacityNoiseModel(bit_flip_prob=p).add_noise_to_circuit(ideal_circuit)
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)

    decoder = DECODERS[decoder_name](
        dem,
        data_qubit_coords=code.base_qubit_coords + code.parity_qubit_coords,
        measurement_qubits_of_symmetry=code.measurement_qubits_of_symmetry,
        boundary_edges_of_real_symmetry=code.boundary_edges_of_symmetry,
        one_lines=code.one_lines,
        base_qubit_coords=code.base_qubit_coords,
    )

    sampler = circuit.compile_detector_sampler()
    cycles = []
    for start in range(0, num_shots, BATCH_SHOTS):
        syndromes, _ = sampler.sample(
            shots=min(BATCH_SHOTS, num_shots - start), separate_observables=True
        )
        # Under code-capacity noise all defects lie in a single time slice, so the correction without
        # post-processing is exactly the correction that the post-processing starts from.
        corrections = decoder.decode_batch_to_corrections(syndromes, post_processing=False)
        _, batch_cycles = decoder.component_post_processing.post_process_slices(corrections)
        cycles.append(batch_cycles)
    cycles = np.concatenate(cycles)

    print(f"decoder={decoder_name} distance={distance} p={p:.4f} num_cycles={cycles.mean():.4f}")
    return (decoder_name, distance, p, cycles.mean(), cycles.std(), num_shots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", choices=DECODERS, required=True)
    parser.add_argument("--distances", type=int, nargs="+", default=[7, 11, 15, 19, 23, 27, 31, 35, 39])
    parser.add_argument(
        "--error-rates",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        default=[0.1, 0.21, 0.05],
        help="Data-qubit bit-flip probabilities to sweep, as a numpy.arange(start, stop, step) range.",
    )
    parser.add_argument("--shots", type=int, default=50000)
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or RESULTS_DIR / f"post_processing_cycles_{args.decoder}.csv"

    error_rates = np.arange(*args.error_rates)

    task_args = list(itertools.product([args.decoder], args.distances, error_rates, [args.shots]))
    with Pool(processes=args.processes) as pool:
        results = pool.map(run_single_simulation, task_args)

    df = pd.DataFrame(
        results, columns=["decoder", "distance", "p_phys", "num_cycles", "num_cycles_std", "num_shots"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {output}")

    fig, ax = plt.subplots(figsize=(9, 6))
    for p in error_rates:
        subset = df[np.isclose(df["p_phys"], p)]
        ax.plot(subset["distance"], subset["num_cycles"], marker="o", label=f"p={p:.3f}")
    ax.set_title(f"{args.decoder.upper()} decoder (compiled), code-capacity noise")
    ax.set_xlabel("Code distance d")
    ax.set_ylabel("Number of post-processing cycles")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plot_path = output.with_suffix(".pdf")
    fig.savefig(plot_path, dpi=1000)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
