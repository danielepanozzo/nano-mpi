/* Communicators and groups. */
#include "harness.h"

static int rank_main(int argc, char **argv, void *user)
{
   int me, np, i, r = -1, s = -1, cmp = -1;
   (void) argc; (void) argv; (void) user;

   MPI_Init(NULL, NULL);
   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &np);

   /* ---- dup preserves rank and size, and is CONGRUENT not IDENT --------- */
   {
      MPI_Comm dup;
      MPI_Comm_dup(MPI_COMM_WORLD, &dup);
      MPI_Comm_rank(dup, &r); MPI_Comm_size(dup, &s);
      CHECK_INT(r, me); CHECK_INT(s, np);
      MPI_Comm_compare(MPI_COMM_WORLD, dup, &cmp);
      CHECK_INT(cmp, MPI_CONGRUENT);
      MPI_Comm_compare(dup, dup, &cmp);
      CHECK_INT(cmp, MPI_IDENT);

      /* a message on the duplicate must not be visible on the original */
      if (np > 1)
      {
         int v = -1, right = (me + 1) % np, left = (me + np - 1) % np;
         MPI_Request rq;
         int mine = me + 500;
         MPI_Irecv(&v, 1, MPI_INT, left, 7, dup, &rq);
         MPI_Send(&mine, 1, MPI_INT, right, 7, dup);
         MPI_Wait(&rq, MPI_STATUS_IGNORE);
         CHECK_INT(v, left + 500);
      }
      MPI_Barrier(dup);
      MPI_Comm_free(&dup);
      CHECK_INT(dup, MPI_COMM_NULL);
   }

   /* ---- split by parity ------------------------------------------------- */
   {
      MPI_Comm half;
      int expect_size = 0;
      MPI_Comm_split(MPI_COMM_WORLD, me % 2, me, &half);
      MPI_Comm_rank(half, &r);
      MPI_Comm_size(half, &s);
      for (i = 0; i < np; i++) { if (i % 2 == me % 2) { expect_size++; } }
      CHECK_INT(s, expect_size);
      CHECK_INT(r, me / 2);

      /* an Allreduce on the sub-communicator must only see its own members */
      {
         int one = 1, tot = 0;
         MPI_Allreduce(&one, &tot, 1, MPI_INT, MPI_SUM, half);
         CHECK_INT(tot, expect_size);
      }
      MPI_Comm_free(&half);
   }

   /* ---- split with MPI_UNDEFINED drops a rank out ------------------------ */
   if (np > 1)
   {
      MPI_Comm sub;
      MPI_Comm_split(MPI_COMM_WORLD, (me == 0) ? MPI_UNDEFINED : 0, me, &sub);
      if (me == 0)
      {
         CHECK_INT(sub, MPI_COMM_NULL);
      }
      else
      {
         MPI_Comm_size(sub, &s);
         CHECK_INT(s, np - 1);
         MPI_Comm_rank(sub, &r);
         CHECK_INT(r, me - 1);
         MPI_Comm_free(&sub);
      }
   }

   /* ---- split_type: every rank is in the same process, so one group ------ */
   {
      MPI_Comm shared;
      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                          MPI_INFO_NULL, &shared);
      MPI_Comm_size(shared, &s);
      CHECK_INT(s, np);
      MPI_Comm_free(&shared);
   }

   /* ---- groups ---------------------------------------------------------- */
   {
      MPI_Group world, sub;
      MPI_Comm  subcomm;
      int  n = (np + 1) / 2;
      int *ranks = (int *) malloc(sizeof(int) * (size_t) n);

      for (i = 0; i < n; i++) { ranks[i] = i * 2; }   /* even world ranks */
      MPI_Comm_group(MPI_COMM_WORLD, &world);
      MPI_Group_size(world, &s);
      CHECK_INT(s, np);
      MPI_Group_rank(world, &r);
      CHECK_INT(r, me);

      MPI_Group_incl(world, n, ranks, &sub);
      MPI_Group_size(sub, &s);
      CHECK_INT(s, n);
      MPI_Group_rank(sub, &r);
      CHECK_INT(r, (me % 2 == 0) ? me / 2 : MPI_UNDEFINED);

      MPI_Comm_create(MPI_COMM_WORLD, sub, &subcomm);
      if (me % 2 == 0)
      {
         int one = 1, tot = 0;
         MPI_Comm_size(subcomm, &s);
         CHECK_INT(s, n);
         MPI_Comm_rank(subcomm, &r);
         CHECK_INT(r, me / 2);
         MPI_Allreduce(&one, &tot, 1, MPI_INT, MPI_SUM, subcomm);
         CHECK_INT(tot, n);
         MPI_Comm_free(&subcomm);
      }
      else
      {
         CHECK_INT(subcomm, MPI_COMM_NULL);
      }

      MPI_Group_free(&sub);
      MPI_Group_free(&world);
      free(ranks);
   }

   /* ---- COMM_SELF is genuinely private ---------------------------------- */
   {
      int v = me, got = -1;
      MPI_Allreduce(&v, &got, 1, MPI_INT, MPI_SUM, MPI_COMM_SELF);
      CHECK_INT(got, me);
      MPI_Barrier(MPI_COMM_SELF);
   }

   MPI_Finalize();
   return t_fails;
}

TEST_MAIN(rank_main)
