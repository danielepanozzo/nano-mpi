# nano-mpi

**MPI where every rank is a thread of one process.** No `mpirun`, no daemon, no
sockets — `MPI_Send` between ranks is a `memcpy`.

It exists so a library can use MPI-style domain decomposition *internally*
without forcing its caller to launch under `mpirun`. Write the ordinary
domain-decomposed algorithm; ship it as a plain shared library that anyone can
call from a plain single-threaded program.

```c
#include <nanompi.h>

static int rank_main(int argc, char **argv, void *user)
{
    int me, np, total, mine;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    mine = me + 1;
    MPI_Allreduce(&mine, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (me == 0) printf("%d ranks, sum %d\n", np, total);
    return 0;
}

int main(int argc, char **argv)
{
    return nanompi_run(8, rank_main, argc, argv, NULL);   /* 8 ranks, 1 process */
}
```

```
$ cc hello.c -lnanompi && ./a.out
8 ranks, sum 36
```

## The one thing you must know

Under `mpirun` each rank is a process, so each rank gets a **private copy** of
every global and every function-scope `static`. That is the operating system's
doing, not MPI's, and MPI programs lean on it constantly.

Here, ranks are threads. **They share one copy of everything.**

```c
static int counter = 0;          /* ONE counter for all ranks. A data race. */
static int cached  = -1;         /* the worst kind: a lazily-filled singleton */
```

Mutable state at file scope has to become `thread_local`, or move into a
context struct each rank owns. `const` data is fine. Libraries you call are
subject to the same rule.

This is the price of the architecture and no amount of work inside nano-mpi
lifts it. [SCOPE.md §5](SCOPE.md) covers it properly, along with what happens
when a rank calls `exit()` (it takes the process down) and what thread level is
provided.

## Build and install

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build            # 50 tests
cmake --install build --prefix /opt/nanompi
```

Requires C11 and pthreads. No other dependencies.

## Three ways to make ranks

There is no launcher, so something in your process has to create the threads.
Which one you want depends on who is calling whom.

### `nanompi_run` — you own `main`

Spawns the ranks, runs your function on all of them, joins, tears down.

```c
int main(int argc, char **argv)
{
    return nanompi_run(0, rank_main, argc, argv, &ctx);   /* 0 = take the default */
}
```

### `nanompi_team_create` / `_invoke` / `_destroy` — the ranks outlive one call

For a library whose object is built once and used many times: a solver you
factorize once and then solve with repeatedly. The calling thread is **not** a
rank; it blocks inside each `invoke`.

```c
nanompi_team *team;
nanompi_team_create(8, &team);
nanompi_team_invoke(team, factorize, ctx);      /* all 8 ranks run it */
nanompi_team_invoke(team, solve, ctx);          /* ... and again */
nanompi_team_destroy(team);
```

### `nanompi_team_start` / `_join` — you *are* rank 0

For code already written in the SPMD style. `team_start` returns to the calling
thread immediately, having made it rank 0 and spawned ranks 1..n-1 running your
worker. You then drive as rank 0, typically by broadcasting commands the
workers service in a loop.

```c
nanompi_team_start(0, worker, ctx, &team);
cmd = CMD_SOLVE;  MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
/* ... rank-0 half of the solve ... */
cmd = CMD_EXIT;   MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
nanompi_team_join(team);
```

**With no team at all**, the calling thread is a one-rank world:
`MPI_COMM_WORLD` has size 1 and rank 0, and every collective terminates. Library
code that happens to run single-threaded needs no special case.

## How many ranks

| you write | you get |
|---|---|
| `nanompi_run(8, ...)` | 8, always |
| `nanompi_run(0, ...)` | `NANOMPI_NUM_RANKS` if set, else the number of cores |

An explicit positive count wins over the environment, the way
`omp_set_num_threads()` wins over `OMP_NUM_THREADS`. So a program that wants the
variable to work should ask for the default rather than hard-coding a number.

```sh
NANOMPI_NUM_RANKS=32 ./my_program
nanompiexec -n 32 ./my_program      # the same thing, for scripts written for mpiexec
```

## Using it from your build

**Modern CMake** — take the package directly:

```cmake
find_package(nanompi REQUIRED)
target_link_libraries(myapp PRIVATE nanompi::nanompi)
```

**An existing MPI project** — nano-mpi installs a header named `mpi.h` and a
compiler wrapper, so `find_package(MPI)` finds it the same way it finds Open MPI:

```sh
cmake -S . -B build -DMPI_C_COMPILER=/opt/nanompi/bin/nanompicc
```

Nothing in the project changes. It calls `MPI_Init`, `MPI_Comm_rank` and
`MPI_Allreduce` exactly as before; the ranks are just threads now. What *does*
have to change is any per-rank global state — see above.

## Debugging

A hang reports itself instead of needing a debugger:

```sh
NANOMPI_WATCHDOG=10 ./my_program
```

If no rank makes progress for 10 seconds, every rank prints what it is blocked
on — barrier, receive, or probe, with the communicator, peer and tag — and the
process aborts.

The library is clean under ThreadSanitizer, AddressSanitizer and
UndefinedBehaviorSanitizer across the whole test suite; if you are chasing a
race in your own code, building against a sanitized nano-mpi is a good idea.

## What is and is not implemented

85 entry points: point-to-point (blocking, nonblocking, persistent, probes,
wildcards, `MPI_PROC_NULL`), the twelve common collectives with `MPI_IN_PLACE`,
communicators and groups, derived datatypes, the full predefined datatype set of
MPI-3 C, all the predefined reduction operators including `MPI_MAXLOC` /
`MPI_MINLOC`, and user-defined operators.

Deliberately absent, so that calling them is a **link error rather than a
runtime surprise**: one-sided/RMA, parallel I/O, dynamic process management
(impossible — there are no processes to spawn), Fortran bindings (impossible —
Fortran `SAVE` storage is per-process by language rule), `MPI_T`, sessions, and
error handlers.

[SCOPE.md](SCOPE.md) is the specification and lists all of it exactly. Read it
before assuming a call exists.

Reductions combine by **recursive doubling**, matching what Open MPI and MPICH
do rather than summing linearly, so floating-point results are bit-identical to
a real MPI run at the same rank count.

## Status

0.1. The interface is source-compatible with the standard C interface for the
subset above; it is **not** ABI compatible with any other MPI, and you must
recompile to switch. The MPI-5.0 standard ABI is the only ABI target worth
adopting and is a candidate for a later major version.

nano-mpi does not claim MPI conformance. The next milestone is a published pass
ledger against the MPICH test suite — which directories run, which tests pass,
and a one-line reason for every exclusion.

## Prior art

[GROMACS `thread_mpi`](https://www.gromacs.org) has shipped an MPI subset over
threads for over a decade, for the same reason. [Adaptive MPI
(AMPI)](https://charm.cs.illinois.edu/research/ampi) from the Charm++ group runs
MPI ranks as user-level threads and has spent twenty years on the shared-globals
problem specifically — global swapping, TLS privatisation, and per-rank address
spaces. If you want automatic privatisation rather than a source contract, read
their work first.

## Licence

Apache-2.0 OR MIT.
