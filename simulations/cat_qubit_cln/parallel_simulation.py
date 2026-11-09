"""
Cat-qubit circuit-level noise: logical error rates of the Global Symmetry Matching (GSM) decoder,
decoded with the compiled decoder (`parity_code.decoders_compiled.CompiledGlobalSymmetryMatchingDecoder`).
Every parameter point is simulated in chunks until TARGET_BLOCK_ERRORS block errors (or
MAX_SHOTS_PER_POINT shots) are reached; the cumulative results are checkpointed to OUTPUT_FILE after
every chunk, so an interrupted run resumes where it stopped.

Usage:
python simulations/cat_qubit_cln/parallel_simulation.py
"""

import os
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from multiprocess.pool import Pool

from parity_code import CatNoiseModel, LhzLegEncodingPhaseFlip
from parity_code.decoders_compiled import CompiledGlobalSymmetryMatchingDecoder

# =====================================================================
# Configuration
# =====================================================================

DISTANCES = [3, 7, 11]

KAPPA1_KAPPA2_RATIOS = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010]
NBAR = 11
POST_PROCESSING = True
TARGET_BLOCK_ERRORS = 100
MAX_SHOTS_PER_POINT = int(1e11)

CHUNK_SHOTS = 250_000 # Number of shots processed by one worker task.
DECODE_BATCH_SHOTS = 50_000  # Number of shots sampled and decoded at once within a chunk (bounds memory).
NUM_PROCESSES = 6

# read by plot_Fig_cat_qubit_LER.ipynb
OUTPUT_FILE = Path(__file__).parent / "data" / "checkpoint.csv"

# Used only for reproducible Stim sampling.
BASE_SEED = 50_000_000

# =====================================================================
# Per-worker cache
# =====================================================================
#
# Constructing the code, DEM, and decoder for every 250k-shot chunk
# would be unnecessarily expensive.
# Each worker process therefore caches these objects locally.
# =====================================================================
_WORKER_CACHE = {}
def get_worker_objects(d, p, post_processing):
    """
    Construct and cache everything that is independent of the chunk.

    Returns
    -------
    noisy_circuit
    decoder
    code
    """

    key = (d, float(p), bool(post_processing), NBAR)

    if key in _WORKER_CACHE:
        return _WORKER_CACHE[key]

    print(f"[worker] Building d={d}, p={p:g}")

    code = LhzLegEncodingPhaseFlip(num_logical_qubits=d)
    ideal_circuit = code.create_stim_circuit(rounds=d)
    noise_model = CatNoiseModel(nbar=NBAR, kappa1_kappa2_ratio=p)
    noisy_circuit = noise_model.add_noise_to_circuit(ideal_circuit, num_rounds=d)
    dem = noisy_circuit.detector_error_model(approximate_disjoint_errors=True)

    decoder = CompiledGlobalSymmetryMatchingDecoder(
        dem,
        data_qubit_coords=(code.base_qubit_coords + code.parity_qubit_coords),
        measurement_qubits_of_symmetry=(code.measurement_qubits_of_symmetry),
        boundary_edges_of_real_symmetry=(code.boundary_edges_of_symmetry),
        one_lines=code.one_lines,
        base_qubit_coords=code.base_qubit_coords,
    )

    _WORKER_CACHE[key] = (
        noisy_circuit,
        decoder,
        code,
    )

    return _WORKER_CACHE[key]


# =====================================================================
# Reproducible chunk seeds
# =====================================================================
def make_seed(d, p, chunk_index):
    """
    Generate a deterministic seed for a specific simulation chunk.
    The seed depends only on
        d,
        p,
        chunk_index,
    and therefore does not depend on process scheduling.

    This is useful when resuming interrupted simulations.
    """

    text = f"{BASE_SEED}|" f"{d}|" f"{p:.16g}|" f"{chunk_index}"

    digest = hashlib.blake2b(
        text.encode(),
        digest_size=8,
    ).digest()

    seed = int.from_bytes(
        digest,
        byteorder="little",
        signed=False,
    )

    # Keep seed comfortably within signed 64-bit range.
    return seed % (2**63 - 1)


# =====================================================================
# One parallel simulation task
# =====================================================================
def run_chunk(args):
    """
    Simulate one independent chunk.

    Returns cumulative ingredients rather than already-normalized
    probabilities so that chunks can be combined exactly.
    """
    (d, p, chunk_index, num_shots, post_processing) = args

    noisy_circuit, decoder, code = get_worker_objects(d=d, p=p, post_processing=post_processing)

    seed = make_seed( d=d, p=p, chunk_index=chunk_index)

    # Compile with a different deterministic seed for each chunk.
    sampler = noisy_circuit.compile_detector_sampler(seed=seed)

    num_block_errors = 0
    num_logical_errors = 0

    for start in range(0, num_shots, DECODE_BATCH_SHOTS):
        syndromes, actual_obs = sampler.sample(
            shots=min(DECODE_BATCH_SHOTS, num_shots - start), separate_observables=True
        )
        predicted_obs = decoder.decode_batch(syndromes, post_processing=post_processing)

        error_vectors = predicted_obs != actual_obs
        num_block_errors += int(np.count_nonzero(error_vectors.any(axis=1)))
        num_logical_errors += int(np.count_nonzero(error_vectors))

    return {
        "d": d,
        "p_ph": p,
        "chunk_index": chunk_index,
        "num_shots": num_shots,
        "num_block_errors": num_block_errors,
        "num_logical_errors": num_logical_errors,
    }


# =====================================================================
# Derived logical error rates
# =====================================================================


def compute_rates(
    d,
    num_shots,
    num_block_errors,
    num_logical_errors,
):
    """
    Compute the different logical-error quantities we want to save.
    """

    if num_shots == 0:
        return {
            "p_block": np.nan,
            "p_logical_per_k": np.nan,
            "p_logical_per_k_per_d": np.nan,
        }
    p_block = num_block_errors / num_shots
    p_logical_per_k = 1.0 - (1.0 - p_block) ** (1.0 / d)
    p_logical_per_k_per_d = p_logical_per_k / d

    return {
        "p_block": p_block,
        "p_logical_per_k": p_logical_per_k,
        "p_logical_per_k_per_d": p_logical_per_k_per_d,
    }


# =====================================================================
# Checkpoint handling
# =====================================================================
def initial_state(d, p):
    return {
        "d": int(d),
        "p_ph": float(p),
        "nbar": NBAR,
        "post_processing": POST_PROCESSING,
        "num_shots": 0,
        "num_block_errors": 0,
        "num_logical_errors": 0,
        # Number of completely finished chunks.
        "num_chunks": 0,
        "status": "running",
    }

def state_key(d, p):
    """
    Stable dictionary key for a parameter point.
    """
    return (int(d), f"{float(p):.16g}")

def load_checkpoint():
    """
    Load existing cumulative results if present.

    Missing requested parameter points are initialized automatically.
    """

    states = {}

    if OUTPUT_FILE.exists():
        print(f"Loading checkpoint from " f"{OUTPUT_FILE}")

        df = pd.read_csv(OUTPUT_FILE)

        for _, row in df.iterrows():
            key = state_key(
                row["d"],
                row["p_ph"],
            )

            states[key] = {
                "d": int(row["d"]),
                "p_ph": float(row["p_ph"]),
                "nbar": float(row["nbar"]),
                "post_processing": bool(row["post_processing"]),
                "num_shots": int(row["num_shots"]),
                "num_block_errors": int(row["num_block_errors"]),
                "num_logical_errors": int(row["num_logical_errors"]),
                "num_chunks": int(row["num_chunks"]),
                "status": str(row["status"]),
            }

    # Add parameter points which are not yet present.
    for d, p in itertools.product(
        DISTANCES,
        KAPPA1_KAPPA2_RATIOS,
    ):
        key = state_key(d, p)

        if key not in states:
            states[key] = initial_state(
                d,
                p,
            )

    return states

def checkpoint_dataframe(states):
    """
    Convert cumulative states into the complete output table.
    """

    rows = []
    for state in states.values():
        rates = compute_rates(
            d=state["d"],
            num_shots=state["num_shots"],
            num_block_errors=(state["num_block_errors"]),
            num_logical_errors=(state["num_logical_errors"]),
        )

        row = dict(state)
        row.update(rates)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["d", "p_ph"]).reset_index(drop=True)

    return df

def save_checkpoint(states):
    """
    Atomically write the current state.

    Writing to a temporary file first prevents a crash during to_csv()
    from destroying the previous checkpoint.
    """
    df = checkpoint_dataframe(states)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    df.to_csv( temporary_file, index=False)

    os.replace(temporary_file, OUTPUT_FILE)


# =====================================================================
# Determine whether a point is complete
# =====================================================================
def update_status(state):
    if state["num_block_errors"] >= TARGET_BLOCK_ERRORS:
        state["status"] = "target_errors_reached"

    elif state["num_shots"] >= MAX_SHOTS_PER_POINT:
        state["status"] = "max_shots_reached"

    else:
        state["status"] = "running"


def is_active(state):
    update_status(state)
    return state["status"] == "running"


# =====================================================================
# Build the next parallel batch
# =====================================================================
def build_tasks(states):
    """
    Fill the worker pool with independent chunks.
    If there is only one active parameter point and NUM_PROCESSES=6,
    six chunks of that point are scheduled in parallel.
    If there are two active points, approximately three chunks of each
    are scheduled.
    Chunk indices are unique and deterministic, so checkpoint/resume
    remains reproducible.
    """
    requested_keys = {state_key(d, p) for d, p in itertools.product(DISTANCES, KAPPA1_KAPPA2_RATIOS)}

    active = []
    for key, state in states.items():
        if key not in requested_keys:
            continue

        if not is_active(state):
            continue

        active.append((key, state))

    if not active:
        return []

    tasks = []
    # Number of chunks already scheduled for each point in THIS batch.
    scheduled_per_point = {key: 0 for key, _ in active}

    # Round-robin through active parameter points until every worker
    # has a task.
    active_index = 0

    while len(tasks) < NUM_PROCESSES:
        key, state = active[active_index]
        offset = scheduled_per_point[key]
        # Account for chunks that have already been scheduled in this
        # batch but have not yet completed.
        already_scheduled_shots = offset * CHUNK_SHOTS

        remaining_shots = MAX_SHOTS_PER_POINT - state["num_shots"] - already_scheduled_shots

        if remaining_shots > 0:
            this_chunk_shots = min( CHUNK_SHOTS, remaining_shots)
            chunk_index = state["num_chunks"] + offset

            tasks.append((state["d"], state["p_ph"], chunk_index, this_chunk_shots, POST_PROCESSING))

            scheduled_per_point[key] += 1

        active_index = (active_index + 1) % len(active)

        # Safety: all active points may have exhausted MAX_SHOTS.
        if all(
            (MAX_SHOTS_PER_POINT - state["num_shots"] - scheduled_per_point[key] * CHUNK_SHOTS) <= 0
            for key, state in active
        ):
            break

    return tasks


# =====================================================================
# Main simulation
# =====================================================================
def main():

    states = load_checkpoint()

    # Update status in case a previous run already completed points.
    for state in states.values():
        update_status(state)

    save_checkpoint(states)

    print("\nCurrent simulation state:")
    print(
        checkpoint_dataframe(states)[
            [
                "d",
                "p_ph",
                "num_shots",
                "num_block_errors",
                "p_block",
                "status",
            ]
        ].to_string(index=False)
    )

    pool = Pool(processes=NUM_PROCESSES)

    try:

        while True:
            tasks = build_tasks(states)
            if not tasks:
                print("\nAll requested parameter " "points are finished.")
                break
            print(f"\nStarting {len(tasks)} chunks...")

            # Results are returned as soon as individual workers finish.
            for result in pool.imap( run_chunk, tasks, chunksize=1):

                key = state_key(result["d"], result["p_ph"])
                state = states[key]
                state["num_shots"] += result["num_shots"]
                state["num_block_errors"] += result["num_block_errors"]
                state["num_logical_errors"] += result["num_logical_errors"]
                state["num_chunks"] += 1

                update_status(state)

                rates = compute_rates(
                    d=state["d"],
                    num_shots=state["num_shots"],
                    num_block_errors=(state["num_block_errors"]),
                    num_logical_errors=(state["num_logical_errors"]),
                )

                print(
                    f"d={state['d']:2d}, "
                    f"p={state['p_ph']:.2e}, "
                    f"shots={state['num_shots']:,}, "
                    f"block_errors="
                    f"{state['num_block_errors']:,}, "
                    f"p_block="
                    f"{rates['p_block']:.3e}, "
                    f"{state['status']}"
                )

                # Checkpoint immediately after every completed chunk.
                save_checkpoint(states)

    except KeyboardInterrupt:
        print("\nInterrupted by user. " "Saving completed chunks...")
        save_checkpoint(states)
        pool.terminate()
        pool.join()
        print(f"Checkpoint saved to " f"{OUTPUT_FILE}")
        return

    except Exception:
        # Preserve all successfully completed chunks even if a worker
        # encounters an unexpected error.
        save_checkpoint(states)
        pool.terminate()
        pool.join()
        raise

    else:
        pool.close()
        pool.join()
        save_checkpoint(states)
        print(f"\nFinal results saved to " f"{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
