"""
Logical error rate vs. physical error rate for the Independent Symmetry Matching (ISM) decoder
under a bit flip circuit-level noise, decoded with the compiled decoder
(`parity_code.decoders_compiled.CompiledIndependentSymmetryMatchingDecoder`):

Usage:
python simulations/bit_flip_cln/logical_error_rate_ism.py --distances 3 7 11 15 --error-rates 0.005 0.02 0.001 --shots 50000 --no-post-processing
python simulations/bit_flip_cln/logical_error_rate_ism.py --distances 3 7 11 15 --error-rates 0.005 0.02 0.001 --shots 50000
"""

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocess.pool import Pool

from parity_code import GenericCircuitLevelNoiseModel, LhzLegEncoding
from parity_code.decoders_compiled import CompiledIndependentSymmetryMatchingDecoder

RESULTS_DIR = Path(__file__).parent / "results"

# number of shots sampled and decoded at once (bounds memory)
BATCH_SHOTS = 100_000


def run_single_simulation(args: tuple[int, float, int, bool]) -> tuple[int, float, float, int, bool]:
    distance, p, num_shots, post_processing = args
    code = LhzLegEncoding(num_logical_qubits=distance)
    ideal_circuit = code.create_stim_circuit(rounds=distance)
    noise_model = GenericCircuitLevelNoiseModel(
        before_round_bit_flip_prob=p,
        before_measure_flip_prob=p,
        after_reset_flip_prob=p,
        p_2qubits_gate=p,
    )
    circuit = noise_model.add_noise_to_circuit(ideal_circuit, num_rounds=distance)
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)

    decoder = CompiledIndependentSymmetryMatchingDecoder(
        dem,
        data_qubit_coords=code.base_qubit_coords + code.parity_qubit_coords,
        measurement_qubits_of_symmetry=code.measurement_qubits_of_symmetry,
        boundary_edges_of_real_symmetry=code.boundary_edges_of_symmetry,
        one_lines=code.one_lines,
        base_qubit_coords=code.base_qubit_coords,
    )

    sampler = circuit.compile_detector_sampler()
    num_errors = 0
    for start in range(0, num_shots, BATCH_SHOTS):
        syndromes, actual_obs = sampler.sample(
            shots=min(BATCH_SHOTS, num_shots - start), separate_observables=True
        )
        predictions = decoder.decode_batch(syndromes, post_processing=post_processing)
        num_errors += np.sum(np.any(predictions != actual_obs, axis=1))

    logical_error_rate = num_errors / num_shots
    print(f"distance={distance} p={p:.4f} logical_error_rate={logical_error_rate:.4f}")
    return (distance, p, logical_error_rate, num_shots, post_processing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 7, 11, 15])
    parser.add_argument(
        "--error-rates",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        default=[0.005, 0.02, 0.001],
        help="Physical error rates to sweep, as a numpy.arange(start, stop, step) range.",
    )
    parser.add_argument("--shots", type=int, default=50000)
    parser.add_argument("--no-post-processing", action="store_true")
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "logical_error_rate_ism_default.csv")
    args = parser.parse_args()

    error_rates = np.arange(*args.error_rates)
    post_processing = not args.no_post_processing

    task_args = list(
        itertools.product(args.distances, error_rates, [args.shots], [post_processing])
    )
    with Pool(processes=args.processes) as pool:
        results = pool.map(run_single_simulation, task_args)

    df = pd.DataFrame(results, columns=["distance", "p_phys", "p_log", "num_shots", "post_processing"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

    fig, ax = plt.subplots(figsize=(9, 6))
    for distance in args.distances:
        subset = df[df["distance"] == distance]
        ax.plot(subset["p_phys"], subset["p_log"], marker="o", label=f"d={distance}")
    ax.set_title(f"ISM decoder (compiled), circuit-level noise, post-processing={post_processing}")
    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("Logical error rate")
    ax.legend()
    ax.loglog()
    ax.grid(True)
    fig.tight_layout()
    plot_path = args.output.with_suffix(".pdf")
    fig.savefig(plot_path, dpi=1000)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
