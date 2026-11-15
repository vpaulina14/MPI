"""MPI version 1: broadcast of the query words + static block distribution of files."""
import os
import time
from collections import Counter

from mpi4py import MPI

TOP_N = 10


def load_query(path):
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def count_files(files, targets):
    """Count occurrences of the query words in the given files (same tokenization as the baseline)."""
    freq = Counter()
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                for w in line.lower().split():
                    if w in targets:
                        freq[w] += 1
    return freq


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

    comm.Barrier()
    t_start = MPI.Wtime()

    # Rank 0 reads consulta.txt and lists the corpus files
    words, files = None, None
    if rank == 0:
        words = sorted(load_query(os.path.join(dataset_dir, "consulta.txt")))
        files = sorted(
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.startswith("file_") and f.endswith(".txt")
        )

    # Broadcast of the query words
    words = comm.bcast(words, root=0)
    targets = set(words)

    # Static distribution: contiguous blocks of files, same number of files per process
    chunks = None
    if rank == 0:
        base, extra = divmod(len(files), size)
        chunks, start = [], 0
        for r in range(size):
            n = base + (1 if r < extra else 0)
            chunks.append(files[start:start + n])
            start += n
    my_files = comm.scatter(chunks, root=0)

    # Local counting
    t0 = MPI.Wtime()
    local_freq = count_files(my_files, targets)
    local_time = MPI.Wtime() - t0

    # Gather partial results and local stats in rank 0
    partials = comm.gather(local_freq, root=0)
    stats = comm.gather((rank, len(my_files), local_time), root=0)

    if rank == 0:
        global_freq = Counter()
        for p in partials:
            global_freq.update(p)
        total_time = MPI.Wtime() - t_start

        print(f"MPI version 1 | processes: {size} | files: {len(files)}")
        print("\nPer-process statistics:")
        for r, n, t in stats:
            print(f"  Rank {r}: files={n}, local_time={t:.6f} s")
        times = [t for _, _, t in stats]
        mean_t = sum(times) / len(times)
        print(f"\nLI = max/mean = {max(times) / mean_t:.4f}")
        print(f"Relative spread = {(max(times) - min(times)) / mean_t:.4f}")
        print(f"\nTotal occurrences: {sum(global_freq.values())}")
        print(f"\nTop {TOP_N} query words in the corpus:")
        for w, c in global_freq.most_common(TOP_N):
            print(f"  {w}: {c}")
        print(f"\nEXECUTION_TIME={total_time:.6f}")


if __name__ == "__main__":
    main()
