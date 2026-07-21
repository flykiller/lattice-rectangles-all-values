# Fast All-Values Lattice Rectangle Counting

Exact C++ implementations for computing the complete table `F(1), ..., F(N)`,
where `F(n)` is the number of axis-aligned and oblique rectangles with vertices
on an `n x n` grid of points ([OEIS A085582](https://oeis.org/A085582)).

Code accompanying the article
[Computing All Lattice-Rectangle Counts by Rational Staircase Sums](https://arxiv.org/abs/2607.17982).

## Files

- `all_values_fast.cpp` - proposed `O(N log^2 N)` implementation.
- `all_values_baseline.cpp` - previous `O(N^(3/2))` all-values algorithm.

Both programs are single-threaded and require a C++20 compiler with `__int128`
support (GCC or Clang). Benchmark mode uses Windows process priority and
affinity controls.

## Build

```sh
g++ -O3 -DNDEBUG -march=native -flto -std=c++20 all_values_fast.cpp -o all_values_fast
g++ -O3 -DNDEBUG -march=native -flto -std=c++20 all_values_baseline.cpp -o all_values_baseline
```

## Run

```sh
./all_values_fast FROM_EXPONENT TO_EXPONENT RUNS
```

For every exponent `k` in the given range, the program computes the complete
prefix through `N = 2^k` and prints tab-separated columns `power`, `n`, `trial`,
`seconds`, and `result = F(N)`.

Example:

```sh
./all_values_fast 10 20 10
```
