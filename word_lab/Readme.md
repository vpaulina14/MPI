# Lab 2: Parallel Word Counting in a Text Corpus with MPI

## Problem Description

Given a query file `dataset/consulta.txt` (one word per line, 200 words) and a corpus of 3000 files `dataset/file_XXXX.txt` (about 45 million tokens; file sizes between 2,000 and 100,000 words), count how many times each query word appears in the corpus and report the 10 most frequent ones. The goal is to compare a sequential baseline with two MPI implementations that differ in how the files are distributed among the processes.

## Environment and Execution Instructions

- Docker image: `augustosalazar/slim-mpi:2` (Python + mpi4py + Open MPI).
- Machine: Lenovo IdeaPad 5 14ALC05 laptop, AMD Ryzen 7 5700U processor (8 cores / 16 threads, 1.80 GHz base), 16 GB RAM (15.3 GB usable), Windows 11 Home 25H2 (build 26200.9711), Docker Desktop 4.73.1 (Engine 29.4.3). The container sees `nproc = 16`.
- Number of processes `p = 16`, the value recommended by `detectar_procesos_mpi.py` ("N RECOMENDADO: 16").
- Commands (PowerShell; in Linux/macOS use `"$(pwd)":/app`):

```powershell
# 1. Generate the dataset (seed 42, 3000 files)
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/generator.py

# 2. Sequential baseline
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py

# 3. Detect p
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/detectar_procesos_mpi.py

# 4. MPI versions (p = 16)
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 16 python /app/mpi1.py
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 16 python /app/mpi2.py
```

`run_all.sh` (provided, unchanged) automates all of the above (`NP_LIST` and `RUNS` can be overridden).

## Experimental Plan

### Sequential baseline
`baseline_secuencial.py` (provided, unchanged) loads the query words into a set, goes through every `file_*.txt` line by line, lowercases and splits each line into tokens, and increments a `Counter` for each token that belongs to the query set. It prints the execution time (`Tseq`) and the top 10 words, and it is the correctness and timing reference.

### MPI version 1 (`mpi1.py`)
1. Rank 0 reads `consulta.txt` and lists the `file_*.txt` files (sorted by name).
2. The query words are sent to all processes with `comm.bcast`.
3. **Static distribution**: the sorted file list is split into `p` contiguous blocks with the same number of files each (187 or 188 files per process with `p = 16`) and sent with `comm.scatter`. File sizes are not considered.
4. Each process counts the query words in its files with the same tokenization as the baseline and measures its local processing time with `MPI.Wtime`.
5. Partial `Counter`s and per-process statistics are gathered in rank 0 (`comm.gather`), which merges them and prints files and local time of each process, LI, the top 10, and `EXECUTION_TIME`.

### Test procedure
- Correctness: the top 10 (words and counts) and the total number of occurrences printed by each MPI version must be identical to the baseline output.
- Timing: 3 runs of the baseline and 3 runs of each MPI version with the same `p = 16`. The total time (`EXECUTION_TIME`) is measured in rank 0 from after a `Barrier` until the global result is built, so it includes listing the files, the distribution, the local counting and the gather. `T_p` is the average of the 3 runs.
- Metrics: `S_p = Tseq / T_p`, `E_p = S_p / p`, `LI = max(t_i) / mean(t_i)` and, optionally, the relative spread `(max(t_i) - min(t_i)) / mean(t_i)`, where `t_i` is the local processing time of process `i`.

## Experimental Plan Execution

### Sequential baseline timing

| Run | Tseq (s) |
|-----|----------|
| 1 | 69.655 |
| 2 | 74.724 |
| 3 | 71.623 |
| **Average** | **72.001** |

Top 10 (reference): a 785774, para 392156, sus 228913, otros 105530, ante 99832, unos 88794, otra 83901, vosotros 61617, mios 58420, tuya 56635 (3,631,778 occurrences in total).

### MPI version 1 timing results (p = 16)

| Run | Total time (s) | Output matches baseline |
|-----|----------------|-------------------------|
| 1 | 7.954 | Yes |
| 2 | 8.129 | Yes |
| 3 | 8.068 | Yes |
| **Average T_p** | **8.050** | |

`S_p = 72.001 / 8.050 = 8.94`, `E_p = 8.94 / 16 = 0.559`

### Load imbalance evidence (version 1)

Local processing time per process (s):

| Rank | Files | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|-------|
| 0 | 188 | 7.159 | 7.265 | 7.236 |
| 1 | 188 | 7.394 | 7.536 | 7.479 |
| 2 | 188 | 6.820 | 6.886 | 6.915 |
| 3 | 188 | 6.326 | 6.395 | 6.429 |
| 4 | 188 | **7.887** | **8.042** | **7.982** |
| 5 | 188 | 6.800 | 6.916 | 6.904 |
| 6 | 188 | 7.242 | 7.350 | 7.309 |
| 7 | 188 | 7.357 | 7.457 | 7.440 |
| 8 | 187 | 6.654 | 6.730 | 6.724 |
| 9 | 187 | 7.184 | 7.274 | 7.254 |
| 10 | 187 | 6.929 | 6.973 | 6.981 |
| 11 | 187 | 7.110 | 7.143 | 7.127 |
| 12 | 187 | 7.435 | 7.539 | 7.525 |
| 13 | 187 | 6.348 | 6.410 | 6.392 |
| 14 | 187 | 6.762 | 6.843 | 6.848 |
| 15 | 187 | 7.447 | 7.592 | 7.541 |
| **LI** | | **1.1182** | **1.1252** | **1.1195** |
| Relative spread | | 0.2213 | 0.2304 | 0.2230 |

Average LI = **1.121**. All processes receive 187-188 files, but the slowest process (always rank 4) takes about 25% longer than the fastest (always rank 3).

### MPI version 2 (`mpi2.py`): how it corrects the imbalance, and its results

Version 1 gives every process the same number of files, but file sizes are very uneven (60% small, 30% medium, 10% large), so equal file counts do not mean equal work. Version 2 keeps the same communication scheme (`bcast`, `scatter`, `gather`), but rank 0 assigns the files with a greedy largest-first heuristic: files are sorted by size (`os.path.getsize`) in decreasing order and each file goes to the process with the smallest accumulated number of bytes. Every process ends with a similar amount of data even if the number of files differs (186 to 189 files per process).

| Run | Total time (s) | Max local time (s) | Output matches baseline | LI | Relative spread |
|-----|----------------|--------------------|-------------------------|----|-----------------|
| 1 | 17.229 | 7.510 | Yes | 1.0096 | 0.0221 |
| 2 | 16.655 | 7.349 | Yes | 1.0089 | 0.0220 |
| 3 | 16.956 | 7.432 | Yes | 1.0111 | 0.0239 |
| **Average** | **16.947** | 7.430 | | **1.0099** | 0.0227 |

`S_p = 72.001 / 16.947 = 4.25`, `E_p = 4.25 / 16 = 0.265`

### Comparison

| Version | T_p (s) | S_p | E_p | LI | Avg. max local time (s) | Total time minus max local time (s) |
|---------|---------|-----|-----|----|-------------------------|-------------------------------------|
| Sequential | 72.001 | 1.00 | - | - | - | - |
| MPI 1 (static blocks) | 8.050 | 8.94 | 0.559 | 1.121 | 7.970 | 0.08 |
| MPI 2 (size-aware) | 16.947 | 4.25 | 0.265 | 1.010 | 7.430 | 9.52 |

(The last column is the average of total time minus the largest local time of each run, that is, the time spent outside the local counting phase.)

## Analysis

**Parallel performance.** MPI version 1 improved the sequential baseline: the time went from 72.0 s to 8.05 s, a speedup of 8.94 against an ideal of 16, which is an efficiency of 0.56. The measurements point to three parts of the gap. (1) The total time is set by the slowest process: the largest local time (about 7.97 s) is 12% above the mean local time (about 7.11 s). (2) Even the mean local time is far above the ideal share `Tseq / p = 4.5 s`: each process handles 1/16 of the data but needs about 58% more time than a perfectly parallel execution would. This is consistent with the hardware: the machine has 8 physical cores and 16 hardware threads, so with `p = 16` pairs of processes share a physical core, and memory bandwidth and file access through the Docker volume on Windows are shared as well. Our data cannot separate these causes. (3) Communication and start-up costs are small: total time minus the largest local time is only about 0.08 s.

**Load balance.** Version 1 shows clear imbalance: every process has 187-188 files, yet local times range from 6.33 s to 8.04 s, with LI between 1.118 and 1.125 and a relative spread of about 0.22. The slowest process was rank 4 and the fastest rank 3 in all three runs, so the imbalance comes from the block distribution (which files each rank receives) and not from random noise. Version 2 reduced LI to about 1.010 (spread 0.023) and lowered the largest local time from 7.97 s to 7.43 s (-7%). However, this did **not** produce a real performance improvement: the total time rose from 8.05 s to 16.95 s (speedup 4.25, efficiency 0.27). The local counting phase is not responsible; the extra time appears outside it, where version 2 spends about 9.5 s against 0.08 s in version 1. The only step that version 2 adds before the timed counting is the size-aware assignment in rank 0 (reading the size of 3000 files and the greedy assignment), so it is the most likely source, probably because of slow file-metadata access through the Docker volume on Windows. We did not time that step separately, so this is a hypothesis. If the cost were similar to version 1's overhead, the total would be near 7.5 s (speedup of about 9.5); this is a projection, not a measurement.

**Limitations.** (1) A single dataset (seed 42), a single value of `p` and only 3 runs per version, so the averages have limited statistical support, although the run-to-run variation is small. (2) The experiment ran in Docker on a personal Windows computer with other processes running, so file access through the volume and background load add noise. (3) `p = 16` comes from the detector and corresponds to the 16 hardware threads of a laptop with only 8 physical cores, which limits the efficiency of both versions (an ideal `S_p = 16` is not realistic on this machine). (4) The local time covers only the counting phase; the assignment, scatter and gather are not timed separately, so the cause of version 2's extra time is not confirmed. (5) Synthetic data with a heavy-tailed size distribution, so conclusions about load balance may not transfer to other corpora. Limitations 3 and 4 weaken the explanation of the efficiency gap and of version 2's slowdown; the measured speedups, efficiencies and LI values themselves are consistent across runs.

## Conclusions

On this machine and dataset, I would recommend the static block distribution (MPI version 1): it reached a speedup of 8.94 and an efficiency of 0.56 with 16 processes (total time 8.05 s against 72.0 s sequentially), despite an LI of 1.12. The size-aware distribution (MPI version 2) balanced the work much better (LI 1.01 and a 7% lower largest local time), but its total time was 16.9 s (speedup 4.25, efficiency 0.27) because of about 9.5 s spent outside the counting phase, so better balance alone did not translate into better performance. The size-aware strategy would be preferable only if its assignment cost is reduced to a small fraction of the run time, since the gain it offers is limited to the roughly 0.5 s between the largest local times of the two versions.
