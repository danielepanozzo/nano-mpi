# nano-mpi — scope contract

This document is the specification. If behaviour and this document disagree,
one of them is a bug; say which one you think it is when you report it.

Version: 0.1 (Phase 1)

---

## 1. What nano-mpi is

An implementation of a subset of MPI in which **each rank is a thread of a
single process**. There is no launcher, no daemon, no sockets, no shared-memory
segments, and no second process. `MPI_Send` between ranks is a `memcpy` between
two threads' buffers.

It exists so a library can use MPI-style domain decomposition internally
without forcing its caller to run under `mpirun`.

It is **not** a fast path for real distributed computing, not a debugging
shim, and not a replacement for Open MPI or MPICH on a cluster.

## 2. Interface level

nano-mpi installs a header named `mpi.h` and a library named `libnanompi`.
Source code that uses the supported subset compiles and links unchanged.

- **Source-level compatible.** Standard names, standard `int` signatures,
  `MPI_Status` with real `MPI_SOURCE` / `MPI_TAG` / `MPI_ERROR` members.
- **Not ABI compatible** with any other MPI. Handles are `int`s here and
  pointers or different `int`s elsewhere. You must recompile to switch.
- **The MPI-5.0 standard ABI is a stated non-goal for 0.x** and a candidate
  for a later major version. It is the only ABI target worth adopting, because
  it is a written specification rather than another project's internals.

## 3. Supported surface

Present and intended to work:

| Area | Coverage |
|---|---|
| Init / query | `MPI_Init`, `MPI_Init_thread`, `MPI_Initialized`, `MPI_Finalize`, `MPI_Finalized`, `MPI_Query_thread`, `MPI_Get_version`, `MPI_Get_library_version`, `MPI_Get_processor_name`, `MPI_Abort` |
| Point-to-point | `MPI_Send`, `MPI_Recv`, `MPI_Isend`, `MPI_Irecv`, `MPI_Irsend`, `MPI_Sendrecv`, `MPI_Probe`, `MPI_Iprobe`, `MPI_Get_count`, `MPI_PROC_NULL` |
| Completion | `MPI_Wait`, `MPI_Waitall`, `MPI_Waitany`, `MPI_Test`, `MPI_Testall`, `MPI_Request_free` |
| Persistent | `MPI_Send_init`, `MPI_Recv_init`, `MPI_Start`, `MPI_Startall` |
| Collectives | `MPI_Barrier`, `MPI_Bcast`, `MPI_Reduce`, `MPI_Allreduce`, `MPI_Scan`, `MPI_Exscan`, `MPI_Gather`, `MPI_Gatherv`, `MPI_Allgather`, `MPI_Allgatherv`, `MPI_Scatter`, `MPI_Scatterv`, `MPI_Alltoall` |
| Nonblocking collectives | The `MPI_I*` form of each of the above — but see below |
| Communicators | `MPI_Comm_rank`, `MPI_Comm_size`, `MPI_Comm_dup`, `MPI_Comm_free`, `MPI_Comm_split`, `MPI_Comm_split_type`, `MPI_Comm_create`, `MPI_Comm_group`, `MPI_Comm_compare` |
| Groups | `MPI_Group_incl`, `MPI_Group_rank`, `MPI_Group_size`, `MPI_Group_free` |
| Datatypes | `MPI_Type_contiguous`, `MPI_Type_vector`, `MPI_Type_hvector`, `MPI_Type_create_hvector`, `MPI_Type_struct`, `MPI_Type_create_struct`, `MPI_Type_commit`, `MPI_Type_free`, `MPI_Type_size`, `MPI_Type_extent`, `MPI_Get_address`, `MPI_Address` |
| Reductions | `MPI_MAX MIN SUM PROD LAND BAND LOR BOR LXOR BXOR MAXLOC MINLOC REPLACE`, and `MPI_Op_create` / `MPI_Op_free` for user operators |
| Info | `MPI_Info_create`, `MPI_Info_free` (accepted, carry no keys) |
| Shared-memory windows | `MPI_Win_allocate_shared`, `MPI_Win_shared_query`, `MPI_Win_free`, `MPI_Win_fence`, `MPI_Win_sync`, `MPI_Win_lock_all`, `MPI_Win_unlock_all` |
| Timing | `MPI_Wtime`, `MPI_Wtick` |

That is **106 entry points**.

### Nonblocking collectives

`MPI_Iallreduce` and the rest do the collective and hand back a request that is
**already complete**. That is legal — nothing in MPI requires a nonblocking call
to defer any work — but be clear about what you get: the call returns when the
collective is *done*, not before, so there is no overlap to exploit. Code
written for overlap runs correctly; it just does not go faster.

Doing better means running collectives on a helper thread, and a helper thread
has no rank (§7). That is a design question, not an oversight. The full predefined datatype set of MPI-3 C is
present, including the fixed-width `MPI_INT8_T` … `MPI_UINT64_T`, `MPI_C_BOOL`,
the complex types, and the `MPI_FLOAT_INT`-style pair types — plus `MPI_CXX_BOOL`
and the `MPI_CXX_*_COMPLEX` types, which are datatypes in the standard rather
than part of the C++ bindings MPI-3 removed.

### MPI_IN_PLACE

| accepted | refused |
|---|---|
| `MPI_Allreduce`, `MPI_Reduce`, `MPI_Scan`, `MPI_Allgather`, `MPI_Allgatherv`, `MPI_Gather`, `MPI_Gatherv` | `MPI_Scatter`, `MPI_Scatterv`, `MPI_Alltoall` |

The three on the right print a named message and abort rather than corrupting
memory quietly. They are implementable; they are simply not implemented yet.

## 4. Explicit non-goals

These are **absent by design**. Calling them is a link error, not a runtime
surprise — which is deliberate: you find out at build time.

| Absent | Why |
|---|---|
| `MPI_Issend` | A synchronous send must not complete until the matching receive is posted. Sends here are eagerly buffered, so implementing it as `MPI_Isend` would be silently wrong for the one thing `Issend` is used to detect. Absent until it can be real. |
| `MPI_Put`, `MPI_Get`, `MPI_Accumulate` and the rest of RMA | The *shared-memory* half of the one-sided chapter is supported and is listed above -- ranks are threads, so a shared window is a real allocation plus everyone's offset into it, touched with ordinary loads and stores. Data-moving RMA on top of that would be a copy from memory you can already address, and is not provided. |
| Parallel I/O (`MPI_File_*`) | Needs ROMIO or an equivalent. A separate project. |
| Dynamic processes (`MPI_Comm_spawn`, ports, names) | **Impossible here.** There are no processes to spawn. |
| Fortran bindings | **Impossible here.** Fortran `SAVE` and `COMMON` storage is per-process by language rule; see §5. |
| C++ bindings | Removed from the standard in MPI-3. |
| `MPI_T` tool interface | Out of scope. |
| Sessions (MPI-4) | Out of scope. |
| Fault tolerance / `MPI_Comm_revoke` | Out of scope. |
| Cartesian and graph topologies | Out of scope for 0.x; would be a pure convenience layer over what exists. |
| Attribute caching (`MPI_Comm_set_attr`) | Out of scope for 0.x. |
| Error handlers (`MPI_Errhandler_*`) | Out of scope for 0.x. Errors abort with a message; see §6. |
| Profiling interface (`PMPI_*`) | Out of scope for 0.x. |

## 5. The globals contract — read this one

Under `mpirun`, every rank is a process, so every rank gets a **private copy**
of every global and every function-scope `static`. That is a property of the
operating system, not of MPI, and MPI programs lean on it everywhere.

Under nano-mpi, ranks are threads. **They share one copy of everything.**

```c
static int counter = 0;         /* ONE counter for all ranks. Almost certainly a bug. */

void step(void) { counter++; }  /* data race */
```

The rules for code that must run under nano-mpi:

1. **Mutable state at file scope must be `thread_local`**, or moved into a
   context struct that each rank owns. There is no third option.
2. `const` data at file scope is fine and needs no change.
3. Lazily-initialised singletons (`static int done; if (!done) {...}`) are the
   most common failure and the hardest to see. Audit them first.
4. Third-party libraries you call are subject to the same rule. A library that
   is not thread-safe is not nano-mpi-safe.

This constraint cannot be lifted by any amount of work inside nano-mpi.
It is the price of the architecture.

*(Prior art: Adaptive MPI, from the Charm++ group, has attacked this same
problem for two decades with global-swapping, TLS privatisation, and per-rank
address spaces via PIE and `dlopen`. If you need automatic privatisation rather
than a source contract, read their work before inventing anything.)*

## 6. Process-lifetime contract

- **`exit()` in a rank terminates the whole program**, including every other
  rank, without unwinding them. Return from the rank function instead.
- **`abort()` likewise**, and `MPI_Abort` is implemented as `abort()`; the
  `errorcode` is reported on stderr but the process dies with `SIGABRT` rather
  than that code, because there is no launcher to translate it.
- A rank that segfaults takes the process with it, as any thread does.
- Signals are process-wide. There is no per-rank signal disposition.

### Debugging a hang

Set `NANOMPI_WATCHDOG=<seconds>`. If no rank makes progress for that long, every
rank prints what it is blocked on -- barrier, receive or probe, with the
communicator, peer and tag -- and the process aborts. Cheaper than attaching a
debugger to sixty-four threads.

## 7. Threading

`MPI_Init_thread` reports **`MPI_THREAD_FUNNELED`**, which here means: *only
the thread that is the rank may make MPI calls on that rank's behalf*.

A rank may spawn helper threads (OpenMP, a thread pool) and those threads may
do compute, but they must not call MPI — they have no rank identity, because
rank identity is thread-local. Requesting `MPI_THREAD_MULTIPLE` succeeds and
returns `MPI_THREAD_FUNNELED` as `provided`, per the standard's contract that
`provided` may be lower than `required`.

## 8. Known limits

| Limit | Value | Behaviour on exhaustion |
|---|---|---|
| Communicators | 4096 | message on stderr, `abort()` |
| Datatypes | 4096 | message on stderr, `abort()` |
| User reduction ops | 128 | message on stderr, `abort()` |
| In-flight requests per rank | 65536 | message on stderr, `abort()` |
| Elements per message | `INT_MAX` | undefined; `MPI_Count` is not yet supported |

These are static arrays. Raising them is a recompile, and making them grow
dynamically is a Phase 2 item.

## 9. Determinism

`MPI_Reduce` / `MPI_Allreduce` combine contributions by **recursive doubling**,
matching what Open MPI and MPICH do, rather than by a linear sum. Floating-point
addition is not associative, so this is the difference between reproducing a
reference result bit-for-bit and drifting away from it over a long iteration.
Verified bit-identical against Open MPI over random trials at every rank count
from 2 to 16, including non-powers of two.

## 10. Launching

There is no `mpirun`. Three entry points, in `nanompi.h`:

| Entry point | Caller's role | Use when |
|---|---|---|
| `nanompi_run(n, fn, ...)` | not a rank; blocks | you own `main` and just want N ranks |
| `nanompi_team_create` / `_invoke` / `_destroy` | not a rank; blocks per invoke | ranks must outlive one call — a solver object constructed once and solved many times |
| `nanompi_team_start` / `_join` | **is rank 0** | SPMD: you drive as rank 0 and broadcast commands to workers |

With no team started at all, the calling thread is a one-rank world:
`MPI_COMM_WORLD` has size 1 and rank 0. Library code that happens to run
single-threaded therefore needs no special case.

Rank count, when not given explicitly: `NANOMPI_NUM_RANKS` if set to a positive
integer, otherwise the number of cores online. An explicit positive argument
always wins, the way `omp_set_num_threads` wins over `OMP_NUM_THREADS`.

## 11. Compatibility statement

nano-mpi does **not** claim MPI conformance. What it will claim, once the
MPICH test suite is wired up, is a published pass ledger: which directories are
run, which tests pass, and a one-line reason for every exclusion. Until that
ledger exists, the claim is exactly this document plus the 50-test suite in
`test/`, which runs at world sizes 1, 2, 3, 4, 7 and 8 and is clean under
ThreadSanitizer, AddressSanitizer and UndefinedBehaviorSanitizer.
