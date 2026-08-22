/*****************************************************************************
 * nano-mpi launcher API.
 *
 * There is no mpirun. Ranks are threads, so somebody in the process has to
 * create them; these are the three ways to do that. Everything else you need
 * is the ordinary MPI interface in <mpi.h>.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *****************************************************************************/

#ifndef NANOMPI_H
#define NANOMPI_H

#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * How many ranks?
 *
 * Every entry point below takes an 'nranks'. Pass a positive number and it is
 * used as given. Pass <= 0 to take the default: the environment variable
 * NANOMPI_NUM_RANKS if it holds a positive integer, otherwise the number of
 * cores online.
 *
 * An explicit positive argument always wins over the environment, the way
 * omp_set_num_threads() wins over OMP_NUM_THREADS. A caller who wants the
 * variable to take effect should ask for the default rather than hard-coding
 * a count. A malformed setting is reported on stderr and ignored.
 *--------------------------------------------------------------------------*/
int nanompi_default_ranks(void);

/*--------------------------------------------------------------------------
 * Run once.
 *
 * Spawns 'nranks' threads, gives each one a rank of MPI_COMM_WORLD, runs 'fn'
 * on all of them, joins, and tears the world down. Returns the first nonzero
 * return value from any rank, else 0. 'user' is passed through untouched.
 *
 *   static int rank_main(int argc, char **argv, void *user) { ... }
 *   int main(int argc, char **argv)
 *   { return nanompi_run(0, rank_main, argc, argv, NULL); }
 *--------------------------------------------------------------------------*/
int nanompi_run(int nranks, int (*fn)(int argc, char **argv, void *user),
                int argc, char **argv, void *user);

/*--------------------------------------------------------------------------
 * Persistent team, caller NOT a rank.
 *
 * For a library whose ranks must outlive a single call -- a solver built once
 * and then factorized and solved many times. The calling thread blocks inside
 * each invoke while the team runs.
 *
 *   nanompi_team *team;
 *   nanompi_team_create(8, &team);
 *   nanompi_team_invoke(team, factorize, ctx);
 *   nanompi_team_invoke(team, solve,     ctx);
 *   nanompi_team_destroy(team);
 *
 * Only one team may exist per process: the rank universe is process-wide.
 *--------------------------------------------------------------------------*/
typedef struct nanompi_team_struct nanompi_team;

int nanompi_team_create(int nranks, nanompi_team **team);
int nanompi_team_invoke(nanompi_team *team, int (*fn)(void *user), void *user);
int nanompi_team_size(nanompi_team *team);
int nanompi_team_destroy(nanompi_team *team);

/*--------------------------------------------------------------------------
 * Persistent team, caller IS rank 0.
 *
 * For code written in the SPMD style. team_start() makes the calling thread
 * rank 0 and returns to it immediately, having spawned ranks 1..nranks-1 each
 * running 'worker'. You then drive as rank 0 -- typically broadcasting
 * commands that the workers service in a loop -- and call team_join() after
 * telling them to stop.
 *
 * Contrast nanompi_team_invoke, where the caller is not a rank at all.
 *--------------------------------------------------------------------------*/
int nanompi_team_start(int nranks, int (*worker)(void *user), void *user,
                       nanompi_team **team);
int nanompi_team_join(nanompi_team *team);

/*--------------------------------------------------------------------------
 * Where am I?
 *
 * These answer for MPI_COMM_WORLD without needing a communicator in hand, and
 * are safe to call before any team exists (you are then rank 0 of 1).
 *--------------------------------------------------------------------------*/
int nanompi_rank(void);
int nanompi_nranks(void);

#ifdef __cplusplus
}
#endif

#endif /* NANOMPI_H */
