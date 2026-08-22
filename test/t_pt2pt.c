/* Point-to-point: blocking, nonblocking, persistent, probes, wildcards,
 * MPI_PROC_NULL, and the posted-order matching rule. */
#include "harness.h"

/* One tag per section. Ranks that sit a section out run ahead into the next
 * one, so sharing a tag across sections lets a later message satisfy an earlier
 * wildcard receive. The barriers below are the other half of that fix. */
enum {
   TAG_RING = 11, TAG_SENDRECV, TAG_PROCNULL, TAG_FANIN, TAG_PROBE,
   TAG_ORDER, TAG_WILD, TAG_PERSIST, TAG_SHORT
};

static int rank_main(int argc, char **argv, void *user)
{
   int me, np, i;
   MPI_Status st;
   (void) argc; (void) argv; (void) user;

   MPI_Init(NULL, NULL);
   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &np);

   /* ---- ring: send right, receive left --------------------------------- */
   {
      int out = me * 100 + 7, in = -1;
      int right = (me + 1) % np, left = (me + np - 1) % np;

      if (np == 1)
      {
         in = out;   /* a one-rank ring would deadlock on a blocking pair */
      }
      else
      {
         MPI_Request req;
         MPI_Irecv(&in, 1, MPI_INT, left, TAG_RING, MPI_COMM_WORLD, &req);
         MPI_Send(&out, 1, MPI_INT, right, TAG_RING, MPI_COMM_WORLD);
         MPI_Wait(&req, &st);
         CHECK_INT(st.MPI_SOURCE, left);
         CHECK_INT(st.MPI_TAG, TAG_RING);
      }
      CHECK_INT(in, left * 100 + 7);
   }

   MPI_Barrier(MPI_COMM_WORLD);

   /* ---- MPI_Sendrecv, same ring ---------------------------------------- */
   {
      int out = me, in = -1;
      int right = (me + 1) % np, left = (me + np - 1) % np;
      MPI_Sendrecv(&out, 1, MPI_INT, right, TAG_SENDRECV,
                   &in,  1, MPI_INT, left,  TAG_SENDRECV, MPI_COMM_WORLD, &st);
      CHECK_INT(in, left);
   }

   MPI_Barrier(MPI_COMM_WORLD);

   /* ---- MPI_PROC_NULL: send vanishes, receive leaves the buffer alone --- */
   {
      int sentinel = 4242;
      MPI_Send(&sentinel, 1, MPI_INT, MPI_PROC_NULL, TAG_PROCNULL, MPI_COMM_WORLD);
      st.MPI_SOURCE = -99; st.MPI_TAG = -99;
      MPI_Recv(&sentinel, 1, MPI_INT, MPI_PROC_NULL, TAG_PROCNULL, MPI_COMM_WORLD, &st);
      CHECK_INT(sentinel, 4242);
      CHECK_INT(st.MPI_SOURCE, MPI_PROC_NULL);
      {
         int cnt = -1;
         MPI_Get_count(&st, MPI_INT, &cnt);
         CHECK_INT(cnt, 0);
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   /* ---- rank 0 gathers one message from everyone with wildcards --------- */
   if (np > 1)
   {
      if (me == 0)
      {
         int seen = 0;
         for (i = 1; i < np; i++)
         {
            int v = -1, cnt = -1;
            MPI_Recv(&v, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &st);
            MPI_Get_count(&st, MPI_INT, &cnt);
            CHECK_INT(cnt, 1);
            CHECK_INT(v, st.MPI_SOURCE * 3);
            CHECK_INT(st.MPI_TAG, TAG_WILD);
            seen++;
         }
         CHECK_INT(seen, np - 1);
      }
      else
      {
         int v = me * 3;
         MPI_Send(&v, 1, MPI_INT, 0, TAG_WILD, MPI_COMM_WORLD);
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   /* ---- probe, then receive the probed message -------------------------- */
   if (np > 1)
   {
      if (me == 0)
      {
         double buf[8];
         int cnt = -1;
         MPI_Probe(1, TAG_PROBE, MPI_COMM_WORLD, &st);
         MPI_Get_count(&st, MPI_DOUBLE, &cnt);
         CHECK_INT(cnt, 5);
         CHECK_INT(st.MPI_SOURCE, 1);
         MPI_Recv(buf, 8, MPI_DOUBLE, 1, TAG_PROBE, MPI_COMM_WORLD, &st);
         MPI_Get_count(&st, MPI_DOUBLE, &cnt);
         CHECK_INT(cnt, 5);
         for (i = 0; i < 5; i++) { CHECK_DBL(buf[i], i * 0.5, 0.0); }
      }
      else if (me == 1)
      {
         double buf[5];
         for (i = 0; i < 5; i++) { buf[i] = i * 0.5; }
         MPI_Send(buf, 5, MPI_DOUBLE, 0, TAG_PROBE, MPI_COMM_WORLD);
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   /* ---- matching is by POST order, not by wait order -------------------- */
   if (np > 1)
   {
      if (me == 0)
      {
         int a = -1, b = -1;
         MPI_Request r[2];
         /* both posted for the same (source, tag); first posted must take the
            first message sent, whichever we wait on first */
         MPI_Irecv(&a, 1, MPI_INT, 1, TAG_ORDER, MPI_COMM_WORLD, &r[0]);
         MPI_Irecv(&b, 1, MPI_INT, 1, TAG_ORDER, MPI_COMM_WORLD, &r[1]);
         MPI_Wait(&r[1], MPI_STATUS_IGNORE);   /* deliberately the later one */
         MPI_Wait(&r[0], MPI_STATUS_IGNORE);
         CHECK_INT(a, 1001);
         CHECK_INT(b, 1002);
      }
      else if (me == 1)
      {
         int a = 1001, b = 1002;
         MPI_Send(&a, 1, MPI_INT, 0, TAG_ORDER, MPI_COMM_WORLD);
         MPI_Send(&b, 1, MPI_INT, 0, TAG_ORDER, MPI_COMM_WORLD);
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   /* ---- Waitany / Testall over a fan-in --------------------------------- */
   if (np > 1)
   {
      if (me == 0)
      {
         int *buf = (int *) malloc(sizeof(int) * (size_t) np);
         MPI_Request *r = (MPI_Request *) malloc(sizeof(MPI_Request) * (size_t) np);
         int done = 0, flag = 0;

         for (i = 1; i < np; i++)
         {
            MPI_Irecv(&buf[i], 1, MPI_INT, i, TAG_FANIN, MPI_COMM_WORLD, &r[i]);
         }
         while (done < np - 1)
         {
            int idx = -1;
            MPI_Waitany(np - 1, r + 1, &idx, &st);
            CHECK(idx >= 0 && idx < np - 1, "Waitany returned index %d", idx);
            done++;
         }
         MPI_Testall(np - 1, r + 1, &flag, MPI_STATUSES_IGNORE);
         CHECK_INT(flag, 1);
         for (i = 1; i < np; i++) { CHECK_INT(buf[i], i * 11); }
         free(buf); free(r);
      }
      else
      {
         int v = me * 11;
         MPI_Send(&v, 1, MPI_INT, 0, TAG_FANIN, MPI_COMM_WORLD);
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   /* ---- persistent requests, replayed twice ----------------------------- */
   if (np > 1)
   {
      int right = (me + 1) % np, left = (me + np - 1) % np;
      int out = me, in = -1, round;
      MPI_Request sr, rr;

      MPI_Send_init(&out, 1, MPI_INT, right, TAG_PERSIST, MPI_COMM_WORLD, &sr);
      MPI_Recv_init(&in,  1, MPI_INT, left,  TAG_PERSIST, MPI_COMM_WORLD, &rr);

      for (round = 0; round < 2; round++)
      {
         out = me * 10 + round;
         in  = -1;
         MPI_Start(&rr);
         MPI_Start(&sr);
         MPI_Wait(&rr, MPI_STATUS_IGNORE);
         MPI_Wait(&sr, MPI_STATUS_IGNORE);
         CHECK_INT(in, left * 10 + round);
      }
      MPI_Request_free(&sr);
      MPI_Request_free(&rr);
   }

   MPI_Barrier(MPI_COMM_WORLD);

   /* ---- a receive posted with more capacity than the message is legal ---- */
   if (np > 1)
   {
      if (me == 0)
      {
         int buf[16], cnt = -1;
         for (i = 0; i < 16; i++) { buf[i] = -1; }
         MPI_Recv(buf, 16, MPI_INT, 1, TAG_SHORT, MPI_COMM_WORLD, &st);
         MPI_Get_count(&st, MPI_INT, &cnt);
         CHECK_INT(cnt, 3);
         CHECK_INT(buf[0], 7); CHECK_INT(buf[2], 9);
         CHECK_INT(buf[3], -1);   /* untouched past the message */
      }
      else if (me == 1)
      {
         int buf[3];
         buf[0] = 7; buf[1] = 8; buf[2] = 9;
         MPI_Send(buf, 3, MPI_INT, 0, TAG_SHORT, MPI_COMM_WORLD);
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
   return t_fails;
}

TEST_MAIN(rank_main)
