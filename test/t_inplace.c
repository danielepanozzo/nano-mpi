/* MPI_IN_PLACE across the collectives that accept it. */
#include "harness.h"

static int rank_main(int argc, char **argv, void *user)
{
   int me, np, i, j;
   (void) argc; (void) argv; (void) user;

   MPI_Init(NULL, NULL);
   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &np);

   /* ---- Allreduce ------------------------------------------------------- */
   {
      int v[3];
      for (i = 0; i < 3; i++) { v[i] = (me + 1) * (i + 1); }
      MPI_Allreduce(MPI_IN_PLACE, v, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      for (i = 0; i < 3; i++)
      {
         CHECK_INT(v[i], (np * (np + 1) / 2) * (i + 1));
      }
   }
   {
      double v = 1.0 / (me + 2.0), ref;
      MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      ref = v;
      MPI_Bcast(&ref, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      CHECK(v == ref, "in-place Allreduce differs across ranks");
   }

   /* ---- Reduce: the standard says IN_PLACE at the root only -------------- */
   {
      int v = me + 1, want = np * (np + 1) / 2;
      if (me == 0)
      {
         MPI_Reduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
         CHECK_INT(v, want);
      }
      else
      {
         MPI_Reduce(&v, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
         CHECK_INT(v, me + 1);   /* non-root sendbuf untouched */
      }
   }

   /* ---- Scan ------------------------------------------------------------ */
   {
      int v = me + 1;
      MPI_Scan(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      CHECK_INT(v, (me + 1) * (me + 2) / 2);
   }

   /* ---- Allgather: our block is already in place ------------------------- */
   {
      int *buf = (int *) malloc(sizeof(int) * (size_t) np * 2);
      for (i = 0; i < np * 2; i++) { buf[i] = -1; }
      buf[me * 2]     = me * 10;
      buf[me * 2 + 1] = me * 10 + 1;
      MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                    buf, 2, MPI_INT, MPI_COMM_WORLD);
      for (i = 0; i < np; i++)
      {
         CHECK_INT(buf[i * 2],     i * 10);
         CHECK_INT(buf[i * 2 + 1], i * 10 + 1);
      }
      free(buf);
   }

   /* ---- Gather at the root ---------------------------------------------- */
   {
      int *buf = (int *) malloc(sizeof(int) * (size_t) np);
      for (i = 0; i < np; i++) { buf[i] = -1; }
      if (me == 0)
      {
         buf[0] = 0 * 7;
         MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                    buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
         for (i = 0; i < np; i++) { CHECK_INT(buf[i], i * 7); }
      }
      else
      {
         int v = me * 7;
         MPI_Gather(&v, 1, MPI_INT, NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);
      }
      free(buf);
   }

   /* ---- Allgatherv, uneven blocks --------------------------------------- */
   {
      int  n = me + 1, total = np * (np + 1) / 2;
      int *buf    = (int *) malloc(sizeof(int) * (size_t) total);
      int *counts = (int *) malloc(sizeof(int) * (size_t) np);
      int *displs = (int *) malloc(sizeof(int) * (size_t) np);

      for (i = 0; i < np; i++) { counts[i] = i + 1; }
      displs[0] = 0;
      for (i = 1; i < np; i++) { displs[i] = displs[i - 1] + counts[i - 1]; }
      for (i = 0; i < total; i++) { buf[i] = -1; }
      for (j = 0; j < n; j++) { buf[displs[me] + j] = me * 100 + j; }

      MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                     buf, counts, displs, MPI_INT, MPI_COMM_WORLD);
      for (i = 0; i < np; i++)
      {
         for (j = 0; j < counts[i]; j++)
         {
            CHECK_INT(buf[displs[i] + j], i * 100 + j);
         }
      }
      free(buf); free(counts); free(displs);
   }

   MPI_Finalize();
   return t_fails;
}

TEST_MAIN(rank_main)
