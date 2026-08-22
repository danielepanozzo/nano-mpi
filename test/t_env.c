/* Startup, shutdown and the environment queries. */
#include "harness.h"

static int rank_main(int argc, char **argv, void *user)
{
   int flag = -1, ver = 0, sub = -1, provided = -1, len = -1, me = -1, np = -1;
   char name[MPI_MAX_PROCESSOR_NAME], lib[MPI_MAX_LIBRARY_VERSION_STRING];
   char err[MPI_MAX_ERROR_STRING];
   (void) user;

   MPI_Initialized(&flag);
   CHECK_INT(flag, 0);

   MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
   /* asking for MULTIPLE is legal; getting less back is the honest answer */
   CHECK_INT(provided, MPI_THREAD_FUNNELED);

   MPI_Initialized(&flag);
   CHECK_INT(flag, 1);
   MPI_Finalized(&flag);
   CHECK_INT(flag, 0);

   MPI_Query_thread(&provided);
   CHECK_INT(provided, MPI_THREAD_FUNNELED);

   MPI_Get_version(&ver, &sub);
   CHECK_INT(ver, 3);
   CHECK(sub >= 0, "subversion %d", sub);

   MPI_Get_library_version(lib, &len);
   CHECK(len > 0 && strstr(lib, "nano-mpi") != NULL, "library version: %s", lib);

   MPI_Get_processor_name(name, &len);
   CHECK(len > 0 && strchr(name, ':') != NULL, "processor name: %s", name);

   MPI_Error_string(MPI_SUCCESS, err, &len);
   CHECK(len > 0, "error string for MPI_SUCCESS was empty");
   MPI_Error_string(MPI_ERR_SPAWN, err, &len);
   CHECK(strstr(err, "spawn") != NULL, "MPI_ERR_SPAWN text: %s", err);

   MPI_Error_class(MPI_ERR_TRUNCATE, &flag);
   CHECK_INT(flag, MPI_ERR_TRUNCATE);

   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   CHECK(me >= 0 && me < np, "rank %d of %d", me, np);
   CHECK_INT(me, nanompi_rank());
   CHECK_INT(np, nanompi_nranks());

   MPI_Comm_rank(MPI_COMM_SELF, &me);
   MPI_Comm_size(MPI_COMM_SELF, &np);
   CHECK_INT(me, 0);
   CHECK_INT(np, 1);

   /* Wtime must advance and Wtick must be positive */
   {
      double t0 = MPI_Wtime(), t1;
      volatile double acc = 0.0;
      int i;
      for (i = 0; i < 200000; i++) { acc += (double) i; }
      t1 = MPI_Wtime();
      CHECK(t1 > t0, "MPI_Wtime did not advance (%.9f -> %.9f)", t0, t1);
      CHECK(MPI_Wtick() > 0.0, "MPI_Wtick returned %g", MPI_Wtick());
   }

   MPI_Finalize();
   MPI_Finalized(&flag);
   CHECK_INT(flag, 1);

   return t_fails;
}

TEST_MAIN(rank_main)
