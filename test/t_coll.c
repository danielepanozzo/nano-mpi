/* Collectives: shapes and values. Reduction types and operators get their own
 * file; this one is about who ends up with what. */
#include "harness.h"

static int rank_main(int argc, char **argv, void *user)
{
   int me, np, i, j;
   (void) argc; (void) argv; (void) user;

   MPI_Init(NULL, NULL);
   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &np);

   MPI_Barrier(MPI_COMM_WORLD);

   /* ---- Bcast ----------------------------------------------------------- */
   {
      double v[4];
      for (i = 0; i < 4; i++) { v[i] = (me == 0) ? (i + 1) * 1.25 : -1.0; }
      MPI_Bcast(v, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      for (i = 0; i < 4; i++) { CHECK_DBL(v[i], (i + 1) * 1.25, 0.0); }
   }
   if (np > 2)   /* a root that is not rank 0 */
   {
      int v = (me == 2) ? 999 : -1;
      MPI_Bcast(&v, 1, MPI_INT, 2, MPI_COMM_WORLD);
      CHECK_INT(v, 999);
   }

   /* ---- Allreduce / Reduce ---------------------------------------------- */
   {
      int  s = me + 1, got = -1;
      int  want = np * (np + 1) / 2;
      MPI_Allreduce(&s, &got, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      CHECK_INT(got, want);

      got = -1;
      MPI_Reduce(&s, &got, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
      if (me == 0) { CHECK_INT(got, want); }

      got = -1;
      MPI_Allreduce(&s, &got, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      CHECK_INT(got, np);
      MPI_Allreduce(&s, &got, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      CHECK_INT(got, 1);
   }

   /* every rank must see bit-identical floating point, not just close */
   {
      double s = 1.0 / (double)(me + 3), got = 0.0, ref = 0.0;
      MPI_Allreduce(&s, &got, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      ref = got;
      MPI_Bcast(&ref, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      CHECK(got == ref, "Allreduce differs across ranks: %.17g vs %.17g", got, ref);
   }

   /* ---- Scan (inclusive prefix) ----------------------------------------- */
   {
      int s = me + 1, got = -1;
      MPI_Scan(&s, &got, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      CHECK_INT(got, (me + 1) * (me + 2) / 2);
   }

   /* ---- Allgather / Gather ---------------------------------------------- */
   {
      int  s = me * 5;
      int *r = (int *) malloc(sizeof(int) * (size_t) np);
      for (i = 0; i < np; i++) { r[i] = -1; }
      MPI_Allgather(&s, 1, MPI_INT, r, 1, MPI_INT, MPI_COMM_WORLD);
      for (i = 0; i < np; i++) { CHECK_INT(r[i], i * 5); }

      for (i = 0; i < np; i++) { r[i] = -1; }
      MPI_Gather(&s, 1, MPI_INT, r, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (me == 0) { for (i = 0; i < np; i++) { CHECK_INT(r[i], i * 5); } }
      free(r);
   }

   /* ---- Allgatherv / Gatherv: rank i contributes i+1 elements ------------ */
   {
      int  n = me + 1, total = np * (np + 1) / 2;
      int *send   = (int *) malloc(sizeof(int) * (size_t) n);
      int *recv   = (int *) malloc(sizeof(int) * (size_t) total);
      int *counts = (int *) malloc(sizeof(int) * (size_t) np);
      int *displs = (int *) malloc(sizeof(int) * (size_t) np);

      for (i = 0; i < n; i++) { send[i] = me * 100 + i; }
      for (i = 0; i < np; i++) { counts[i] = i + 1; }
      displs[0] = 0;
      for (i = 1; i < np; i++) { displs[i] = displs[i - 1] + counts[i - 1]; }
      for (i = 0; i < total; i++) { recv[i] = -1; }

      MPI_Allgatherv(send, n, MPI_INT, recv, counts, displs, MPI_INT, MPI_COMM_WORLD);
      for (i = 0; i < np; i++)
      {
         for (j = 0; j < counts[i]; j++)
         {
            CHECK_INT(recv[displs[i] + j], i * 100 + j);
         }
      }

      for (i = 0; i < total; i++) { recv[i] = -1; }
      MPI_Gatherv(send, n, MPI_INT, recv, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);
      if (me == 0)
      {
         for (i = 0; i < np; i++)
         {
            for (j = 0; j < counts[i]; j++)
            {
               CHECK_INT(recv[displs[i] + j], i * 100 + j);
            }
         }
      }
      free(send); free(recv); free(counts); free(displs);
   }

   /* ---- Scatter / Scatterv ---------------------------------------------- */
   {
      int *send = NULL, got[2] = { -1, -1 };
      if (me == 0)
      {
         send = (int *) malloc(sizeof(int) * (size_t) np * 2);
         for (i = 0; i < np * 2; i++) { send[i] = i; }
      }
      MPI_Scatter(send, 2, MPI_INT, got, 2, MPI_INT, 0, MPI_COMM_WORLD);
      CHECK_INT(got[0], me * 2);
      CHECK_INT(got[1], me * 2 + 1);
      free(send);
   }
   {
      int  n = me + 1;
      int *recv   = (int *) malloc(sizeof(int) * (size_t) n);
      int *send   = NULL;
      int *counts = (int *) malloc(sizeof(int) * (size_t) np);
      int *displs = (int *) malloc(sizeof(int) * (size_t) np);
      int  total  = np * (np + 1) / 2;

      for (i = 0; i < np; i++) { counts[i] = i + 1; }
      displs[0] = 0;
      for (i = 1; i < np; i++) { displs[i] = displs[i - 1] + counts[i - 1]; }
      if (me == 0)
      {
         send = (int *) malloc(sizeof(int) * (size_t) total);
         for (i = 0; i < total; i++) { send[i] = i; }
      }
      for (i = 0; i < n; i++) { recv[i] = -1; }
      MPI_Scatterv(send, counts, displs, MPI_INT, recv, n, MPI_INT, 0, MPI_COMM_WORLD);
      for (i = 0; i < n; i++) { CHECK_INT(recv[i], displs[me] + i); }
      free(send); free(recv); free(counts); free(displs);
   }

   /* ---- Alltoall: rank i sends (i*np + j) to rank j ---------------------- */
   {
      int *send = (int *) malloc(sizeof(int) * (size_t) np);
      int *recv = (int *) malloc(sizeof(int) * (size_t) np);
      for (i = 0; i < np; i++) { send[i] = me * np + i; recv[i] = -1; }
      MPI_Alltoall(send, 1, MPI_INT, recv, 1, MPI_INT, MPI_COMM_WORLD);
      for (i = 0; i < np; i++) { CHECK_INT(recv[i], i * np + me); }
      free(send); free(recv);
   }

   /* ---- a user-defined operator ----------------------------------------- */
   {
      MPI_Op  op;
      int     s = 1 << (me % 30), got = 0, want = 0;
      extern void t_or_fn(void *, void *, int *, MPI_Datatype *);
      MPI_Op_create((MPI_User_function *) t_or_fn, 1, &op);
      MPI_Allreduce(&s, &got, 1, MPI_INT, op, MPI_COMM_WORLD);
      for (i = 0; i < np; i++) { want |= 1 << (i % 30); }
      CHECK_INT(got, want);
      MPI_Op_free(&op);
   }

   MPI_Finalize();
   return t_fails;
}

void t_or_fn(void *invec, void *inoutvec, int *len, MPI_Datatype *dt)
{
   int *a = (int *) invec, *b = (int *) inoutvec, i;
   (void) dt;
   for (i = 0; i < *len; i++) { b[i] |= a[i]; }
}

TEST_MAIN(rank_main)
