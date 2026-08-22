/* The two persistent-team APIs, and the transitions between them and the
 * implicit one-rank world.
 *
 * This is the shape a library embedding nano-mpi actually uses: it does some
 * work single-threaded, brings a team up for a solve, tears it down, and must
 * find a working one-rank world on the other side. */
#include "harness.h"

static int g_nranks_wanted;

/* ---- what every rank of the invoke-style team runs ----------------------- */
typedef struct { int round; int sum; int size; } ctx;

static int team_step(void *user)
{
   ctx *c = (ctx *) user;
   int me, np, contrib, total = 0;

   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   CHECK_INT(np, g_nranks_wanted);
   CHECK(me >= 0 && me < np, "rank %d of %d", me, np);

   contrib = (me + 1) * (c->round + 1);
   MPI_Allreduce(&contrib, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   /* only rank 0 writes the shared context: the ranks are threads, so every
      other write would be a race -- which is the whole point of section 5 */
   if (me == 0) { c->sum = total; c->size = np; }

   MPI_Barrier(MPI_COMM_WORLD);
   return t_fails;
}

/* ---- the SPMD-style team: workers service broadcast commands ------------- */
enum { CMD_SUM = 1, CMD_MAX = 2, CMD_EXIT = 9 };

static int worker(void *user)
{
   int me, np, cmd = 0, contrib, out;
   (void) user;

   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   CHECK(me > 0, "a worker must not be rank 0, got %d", me);

   for (;;)
   {
      MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (cmd == CMD_EXIT) { break; }
      contrib = me + 1;
      out = 0;
      MPI_Allreduce(&contrib, &out, 1, MPI_INT,
                    (cmd == CMD_SUM) ? MPI_SUM : MPI_MAX, MPI_COMM_WORLD);
   }
   return t_fails;
}

static int check_solo(const char *when)
{
   int me = -1, np = -1, v = 5, got = -1, bad = 0;

   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   if (me != 0 || np != 1)
   {
      fprintf(stderr, "  FAIL %s: world is rank %d of %d, want 0 of 1\n",
              when, me, np);
      bad++;
   }
   MPI_Allreduce(&v, &got, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   if (got != 5)
   {
      fprintf(stderr, "  FAIL %s: solo Allreduce gave %d\n", when, got);
      bad++;
   }
   return bad;
}

int main(int argc, char **argv)
{
   int fails = 0, rc;
   nanompi_team *team = NULL;
   ctx c;

   g_nranks_wanted = (argc > 1) ? atoi(argv[1]) : 0;
   if (g_nranks_wanted <= 0) { g_nranks_wanted = nanompi_default_ranks(); }

   /* before any team exists */
   fails += check_solo("before any team");

   /* ---- caller is not a rank -------------------------------------------- */
   rc = nanompi_team_create(g_nranks_wanted, &team);
   if (rc) { fprintf(stderr, "team_create failed: %d\n", rc); return 1; }
   CHECK_INT(nanompi_team_size(team), g_nranks_wanted);

   for (c.round = 0; c.round < 3; c.round++)
   {
      int want;
      c.sum = -1; c.size = -1;
      rc = nanompi_team_invoke(team, team_step, &c);
      if (rc) { fprintf(stderr, "invoke round %d returned %d\n", c.round, rc); fails++; }
      want = (g_nranks_wanted * (g_nranks_wanted + 1) / 2) * (c.round + 1);
      if (c.sum != want)
      {
         fprintf(stderr, "  FAIL round %d: sum %d, want %d\n", c.round, c.sum, want);
         fails++;
      }
      if (c.size != g_nranks_wanted) { fprintf(stderr, "  FAIL size %d\n", c.size); fails++; }
   }
   nanompi_team_destroy(team);
   team = NULL;

   /* the world must come back, or anything else in the process that uses MPI
      after the solver is gone sees a zero-rank world */
   fails += check_solo("after team_destroy");

   /* ---- caller is rank 0 ------------------------------------------------- */
   rc = nanompi_team_start(g_nranks_wanted, worker, NULL, &team);
   if (rc) { fprintf(stderr, "team_start failed: %d\n", rc); return 1; }
   {
      int me = -1, np = -1;
      MPI_Comm_rank(MPI_COMM_WORLD, &me);
      MPI_Comm_size(MPI_COMM_WORLD, &np);
      if (me != 0) { fprintf(stderr, "  FAIL caller is rank %d, want 0\n", me); fails++; }
      if (np != g_nranks_wanted) { fprintf(stderr, "  FAIL world %d\n", np); fails++; }
   }
   {
      int cmd, contrib = 1, out = 0, want;
      cmd = CMD_SUM;
      MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Allreduce(&contrib, &out, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      want = g_nranks_wanted * (g_nranks_wanted + 1) / 2;   /* rank r sends r+1 */
      if (out != want) { fprintf(stderr, "  FAIL spmd sum %d want %d\n", out, want); fails++; }

      cmd = CMD_MAX;
      out = 0;
      MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Allreduce(&contrib, &out, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (out != g_nranks_wanted) { fprintf(stderr, "  FAIL spmd max %d\n", out); fails++; }

      cmd = CMD_EXIT;
      MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
   }
   nanompi_team_join(team);

   fails += check_solo("after team_join");

   if (fails || t_fails)
   {
      fprintf(stderr, "t_team: FAILED (%d)\n", fails + t_fails);
      return 1;
   }
   printf("t_team: ok (%d ranks)\n", g_nranks_wanted);
   return 0;
}
