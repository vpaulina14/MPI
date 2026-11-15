# Lab 2: Parallel Word Counting in a Text Corpus with MPI

## Problem Description

Given a query file `dataset/consulta.txt` (one word per line, 200 words) and a corpus of 3000 files `dataset/file_XXXX.txt` (about 45 million tokens, file sizes between 2,000 and 100,000 words), count how many times each query word appears in the corpus and report the 10 most frequent ones. The goal is to compare a sequential baseline with two MPI implementations that differ in how files are distributed among processes.

## Environment and Execution Instructions

- Docker image: `augustosalazar/slim-mpi:2` (Python + mpi4py + Open MPI).
- Number of processes `p`: **TODO** (value reported by `detectar_procesos_mpi.py`).

```bash
# 1. Generate the dataset (Linux/macOS; on Windows cmd use "%cd%:/app")
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/generator.py

# 2. Sequential baseline
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py

# 3. Detect the number of processes p (detectar_procesos_mpi.py from the course repository)
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/detectar_procesos_mpi.py

# 4. MPI versions (replace P with the value found in step 3)
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np P python /app/mpi1.py
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np P python /app/mpi2.py

# Or everything at once (NP_LIST and RUNS can be overridden)
docker run --rm -e NP_LIST="P" -e RUNS=3 -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/run_all.sh
```

## Experimental Plan

### Sequential baseline
`baseline_secuencial.py` (provided, unchanged) loads the query words into a set, goes through every `file_*.txt` line by line, lowercases and splits each line into tokens, and increments a `Counter` for each token that belongs to the query set. It prints the execution time (`Tseq`) and the top 10 words, and is the correctness and timing reference.

### MPI version 1 (`mpi1.py`)
1. Rank 0 reads `consulta.txt` and lists the `file_*.txt` files (sorted by name).
2. The query words are sent to all processes with `comm.bcast`.
3. **Static distribution**: the sorted file list is split into `p` contiguous blocks with the same number of files each (`comm.scatter`). File sizes are not considered.
4. Each process counts the query words in its files with the same tokenization as the baseline and measures its local processing time (`MPI.Wtime`).
5. Partial `Counter`s and per-process statistics are gathered in rank 0 (`comm.gather`), which merges them and prints the files and local time of each process, LI, the top 10, and `EXECUTION_TIME`.

### Test procedure
- Correctness: the top 10 (words and counts) printed by `mpi1.py` and `mpi2.py` must be identical to the baseline output.
- Timing: for each version, at least 3 runs with the same `p`. The total time is measured in rank 0 from after a `Barrier` until the final result is built; the average of the runs is `T_p`.
- Metrics: `S_p = Tseq / T_p`, `E_p = S_p / p`, `LI = max(t_i) / mean(t_i)` (optional relative spread `(max - min) / mean`), where `t_i` is the local time of process `i`.

## Experimental Plan Execution

> Machine: **TODO** (CPU model, cores, RAM, Docker settings). `p = TODO`.

### Sequential baseline timing

| Run | Tseq (s) |
|-----|----------|
| 1 | TODO |
| 2 | TODO |
| 3 | TODO |

Top 10 (reference): a 785774, para 392156, sus 228913, otros 105530, ante 99832, unos 88794, otra 83901, vosotros 61617, mios 58420, tuya 56635.

### MPI version 1 timing results

| Run | Total time (s) | Output matches baseline |
|-----|----------------|-------------------------|
| 1 | TODO | TODO |
| 2 | TODO | TODO |
| 3 | TODO | TODO |
| **Average T_p** | TODO | |

`S_p = TODO`, `E_p = TODO`

### Load imbalance evidence (version 1)

| Rank | Files | Local time run 1 (s) | Run 2 (s) | Run 3 (s) |
|------|-------|----------------------|-----------|-----------|
| 0 | TODO | TODO | TODO | TODO |
| ... | | | | |
| **LI** | | TODO | TODO | TODO |

### MPI version 2 (`mpi2.py`): how it corrects the imbalance, and its results

Version 1 gives every process the same number of files, but file sizes are very uneven (60% small, 30% medium, 10% large), so equal file counts do not mean equal work. Version 2 keeps the same communication scheme (`bcast`, `scatter`, `gather`), but rank 0 assigns files with a greedy largest-first heuristic: files are sorted by size (`os.path.getsize`) in decreasing order and each file goes to the process with the smallest accumulated number of bytes. Every process ends with a similar amount of data, even if the number of files differs.

| Run | Total time (s) | Output matches baseline | LI |
|-----|----------------|-------------------------|----|
| 1 | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO |
| 3 | TODO | TODO | TODO |
| **Average** | TODO | | TODO |

### Comparison

| Version | T_p (s) | S_p | E_p | LI |
|---------|---------|-----|-----|----|
| Sequential | TODO | 1.00 | - | - |
| MPI 1 | TODO | TODO | TODO | TODO |
| MPI 2 | TODO | TODO | TODO | TODO |

## Analysis

**Parallel performance.** TODO: did MPI 1 improve on `Tseq`? Compare `S_p` with the ideal `S_p = p` and explain the gap (e.g., the slowest process dominates the total time, file I/O, scatter/gather and serialization overhead, rank 0 listing files, processes competing for the same cores).

**Load balance.** TODO: cite the per-process local times and LI of version 1, then the LI of version 2 and whether the reduction translated into a lower total time and a higher speedup.

**Limitations.** TODO: e.g., one dataset and one value of `p`, only 3 runs, Docker/host noise and file caching, hyper-threads counted as processes, synthetic data, local time does not include communication.

## Conclusions

TODO: one paragraph recommending a distribution strategy, citing the measured speedup, efficiency and LI of both versions.
