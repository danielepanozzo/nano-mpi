/* No team, ever: the calling thread must be a one-rank world.
 *
 * This is the case a library hits when the process happens to be
 * single-threaded. Getting it wrong shows up as MPI_COMM_WORLD reporting size 0
 * and rank -1, which downstream code turns into an empty decomposition rather
 * than an error. There is no nanompi_run here on purpose. */
#include "harness.h"

int main(void)
{
   int me = -1, np = -1, flag = -1, v, got;
   MPI_Comm dup;

   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   CHECK_INT(me, 0);
   CHECK_INT(np, 1);

   MPI_Init(NULL, NULL);
   MPI_Initialized(&flag);
   CHECK_INT(flag, 1);

   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   CHECK_INT(me, 0);
   CHECK_INT(np, 1);
   CHECK_INT(nanompi_rank(), 0);
   CHECK_INT(nanompi_nranks(), 1);

   /* the collectives must all terminate with one rank */
   MPI_Barrier(MPI_COMM_WORLD);
   v = 17; got = -1;
   MPI_Allreduce(&v, &got, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK_INT(got, 17);
   MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   CHECK_INT(v, 17);
   got = -1;
   MPI_Allgather(&v, 1, MPI_INT, &got, 1, MPI_INT, MPI_COMM_WORLD);
   CHECK_INT(got, 17);
   MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
   CHECK_INT(v, 17);

   /* and sending to yourself must work */
   {
      MPI_Request r;
      int out = 99, in = -1;
      MPI_Irecv(&in, 1, MPI_INT, 0, 5, MPI_COMM_WORLD, &r);
      MPI_Send(&out, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
      MPI_Wait(&r, MPI_STATUS_IGNORE);
      CHECK_INT(in, 99);
   }

   MPI_Comm_dup(MPI_COMM_WORLD, &dup);
   MPI_Comm_size(dup, &np);
   CHECK_INT(np, 1);
   MPI_Comm_free(&dup);

   MPI_Finalize();

   if (t_fails) { fprintf(stderr, "t_solo: FAILED\n"); return 1; }
   printf("t_solo: ok (no team)\n");
   return 0;
}
