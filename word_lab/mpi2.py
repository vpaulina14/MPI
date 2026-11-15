"""MPI version 2: broadcast of the query words + size-aware static distribution (LPT greedy).

Files are sorted by size (largest first) and each file is assigned to the process with the
smallest accumulated size so far, so every process receives a similar number of bytes.
"""
import heapq
import os
import time
from collections import Counter

from mpi4py import MPI

TOP_N = 10


def load_query(path):
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def count_files(files, targets):
    freq = Counter()
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                for w in line.lower().split():
                    if w in targets:
                        freq[w] += 1
    return freq


def balanced_chunks(files, size):
    """Greedy LPT: largest files first, each one to the least loaded process."""
    sized = sorted(((os.path.getsize(p), p) for p in files), reverse=True)
    heap = [(0, r) for r in range(size)]  # (accumulated bytes, rank)
    heapq.heapify(heap)
    chunks = [[] for _ in range(size)]
    for nbytes, path in sized:
        load, r = heapq.heappop(heap)
        chunks[r].append(path)
        heapq.heappush(heap, (load + nbytes, r))
    return chunks


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

    comm.Barrier()
    t_start = MPI.Wtime()

    words, files = None, None
    if rank == 0:
        words = sorted(load_query(os.path.join(dataset_dir, "consulta.txt")))
        files = sorted(
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.startswith("file_") and f.endswith(".txt")
        )

    words = comm.bcast(words, root=0)
    targets = set(words)

    chunks = balanced_chunks(files, size) if rank == 0 else None
    my_files = comm.scatter(chunks, root=0)

    t0 = MPI.Wtime()
    local_freq = count_files(my_files, targets)
    local_time = MPI.Wtime() - t0

    partials = comm.gather(local_freq, root=0)
    stats = comm.gather((rank, len(my_files), local_time), root=0)

    if rank == 0:
        global_freq = Counter()
        for p in partials:
            global_freq.update(p)
        total_time = MPI.Wtime() - t_start

        print(f"MPI version 2 | processes: {size} | files: {len(files)}")
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
