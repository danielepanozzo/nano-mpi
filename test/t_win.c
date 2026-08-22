/* Shared-memory windows. Ranks are threads, so the memory really is shared --
 * what has to be right is the layout MPI promises and the ordering. */
#include "harness.h"

static int rank_main(int argc, char **argv, void *user)
{
   int me, np, i, r, disp = -1;
   MPI_Aint qsize = -1;
   MPI_Win win;
   int *seg = NULL;
   (void) argc; (void) argv; (void) user;

   MPI_Init(NULL, NULL);
   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &np);

   /* ---- every rank contributes (me+1) ints ------------------------------ */
   {
      MPI_Aint bytes = (MPI_Aint)((me + 1) * (int) sizeof(int));
      MPI_Win_allocate_shared(bytes, (int) sizeof(int), MPI_INFO_NULL,
                              MPI_COMM_WORLD, &seg, &win);
      CHECK(seg != NULL, "our own segment came back NULL");

      /* a fresh window is zeroed here, which is stricter than MPI requires */
      for (i = 0; i < me + 1; i++) { CHECK_INT(seg[i], 0); }

      for (i = 0; i < me + 1; i++) { seg[i] = me * 100 + i; }
      MPI_Win_fence(0, win);

      /* everyone's segment must be visible, at the size and offset MPI says */
      for (r = 0; r < np; r++)
      {
         int *other = NULL;
         MPI_Win_shared_query(win, r, &qsize, &disp, &other);
         CHECK_INT((int) qsize, (r + 1) * (int) sizeof(int));
         CHECK_INT(disp, (int) sizeof(int));
         CHECK(other != NULL, "segment of rank %d came back NULL", r);
         for (i = 0; i < r + 1; i++) { CHECK_INT(other[i], r * 100 + i); }
      }

      /* the block is contiguous in rank order: rank r owns r+1 ints, so the
         last rank starts (np-1)np/2 ints past the first */
      {
         int *first = NULL, *last = NULL;
         MPI_Win_shared_query(win, 0, NULL, NULL, &first);
         MPI_Win_shared_query(win, np - 1, NULL, NULL, &last);
         CHECK(last - first == (long)((np - 1) * np / 2),
               "segments not contiguous: rank %d starts %ld ints in, want %d",
               np - 1, (long)(last - first), (np - 1) * np / 2);
      }

      /* The next section writes into a neighbour's segment, so everyone must
         be done reading first -- otherwise a fast rank corrupts a slow one's
         verification loop, which is a race in the test, not in the window. */
      MPI_Win_fence(0, win);

      /* writing through another rank's pointer is visible to its owner */
      if (np > 1)
      {
         int *right = NULL;
         MPI_Win_shared_query(win, (me + 1) % np, NULL, NULL, &right);
         right[0] = -7;
         MPI_Win_fence(0, win);
         CHECK_INT(seg[0], -7);
      }

      MPI_Win_fence(0, win);
      MPI_Win_free(&win);
      CHECK_INT(win, MPI_WIN_NULL);
   }

   /* ---- only rank 0 contributes: the pattern a "broadcast buffer" uses --- */
   {
      int *shared = NULL, *q = NULL;
      MPI_Aint bytes = (me == 0) ? (MPI_Aint)(16 * (int) sizeof(int)) : 0;

      MPI_Win_allocate_shared(bytes, (int) sizeof(int), MPI_INFO_NULL,
                              MPI_COMM_WORLD, &shared, &win);
      if (me == 0)
      {
         CHECK(shared != NULL, "rank 0 got no buffer");
         for (i = 0; i < 16; i++) { shared[i] = i * i; }
      }
      else
      {
         CHECK(shared == NULL, "a zero-size segment should come back NULL");
      }
      MPI_Win_fence(0, win);

      /* MPI_PROC_NULL means "the first non-empty segment" */
      MPI_Win_shared_query(win, MPI_PROC_NULL, &qsize, &disp, &q);
      CHECK_INT((int) qsize, 16 * (int) sizeof(int));
      CHECK(q != NULL, "MPI_PROC_NULL query returned NULL");
      for (i = 0; i < 16; i++) { CHECK_INT(q[i], i * i); }

      MPI_Win_lock_all(0, win);
      MPI_Win_sync(win);
      MPI_Win_unlock_all(win);

      MPI_Win_free(&win);
   }

   /* ---- a window on a sub-communicator ---------------------------------- */
   {
      MPI_Comm half;
      int hme, hnp, *hseg = NULL;
      MPI_Comm_split(MPI_COMM_WORLD, me % 2, me, &half);
      MPI_Comm_rank(half, &hme);
      MPI_Comm_size(half, &hnp);
      MPI_Win_allocate_shared((MPI_Aint) sizeof(int), (int) sizeof(int),
                              MPI_INFO_NULL, half, &hseg, &win);
      *hseg = hme + 1000;
      MPI_Win_fence(0, win);
      for (r = 0; r < hnp; r++)
      {
         int *o = NULL;
         MPI_Win_shared_query(win, r, NULL, NULL, &o);
         CHECK_INT(*o, r + 1000);
      }
      MPI_Win_free(&win);
      MPI_Comm_free(&half);
   }

   MPI_Finalize();
   return t_fails;
}

TEST_MAIN(rank_main)
