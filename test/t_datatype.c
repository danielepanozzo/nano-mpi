/* Derived datatypes, sizes and addresses. */
#include "harness.h"

typedef struct { int   id; double x[3]; char tag; } particle;

static int rank_main(int argc, char **argv, void *user)
{
   int me, np, i, sz = -1;
   MPI_Aint ext = 0;
   (void) argc; (void) argv; (void) user;

   MPI_Init(NULL, NULL);
   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &np);

   /* ---- the predefined set reports the local C sizes -------------------- */
   MPI_Type_size(MPI_INT, &sz);        CHECK_INT(sz, (int) sizeof(int));
   MPI_Type_size(MPI_DOUBLE, &sz);     CHECK_INT(sz, (int) sizeof(double));
   MPI_Type_size(MPI_CHAR, &sz);       CHECK_INT(sz, 1);
   MPI_Type_size(MPI_BYTE, &sz);       CHECK_INT(sz, 1);
   MPI_Type_size(MPI_INT64_T, &sz);    CHECK_INT(sz, 8);
   MPI_Type_size(MPI_UINT16_T, &sz);   CHECK_INT(sz, 2);
   MPI_Type_size(MPI_C_DOUBLE_COMPLEX, &sz);
   CHECK_INT(sz, 2 * (int) sizeof(double));
   MPI_Type_size(MPI_DOUBLE_INT, &sz);
   CHECK_INT(sz, (int) sizeof(MPI_Double_int));
   MPI_Type_extent(MPI_DOUBLE, &ext);  CHECK_INT((int) ext, (int) sizeof(double));

   /* ---- MPI_Get_address ------------------------------------------------- */
   {
      particle p = { 0, { 0, 0, 0 }, 0 };
      MPI_Aint base = 0, xaddr = 0;
      MPI_Get_address(&p, &base);
      MPI_Get_address(&p.x[0], &xaddr);
      CHECK_INT((int)(xaddr - base), (int) offsetof(particle, x));
      /* MPI_Address is the MPI-1 spelling of the same thing */
      MPI_Address(&p.x[0], &ext);
      CHECK(ext == xaddr, "MPI_Address and MPI_Get_address disagree");
   }

   /* ---- contiguous ------------------------------------------------------ */
   {
      MPI_Datatype v3;
      MPI_Type_contiguous(3, MPI_DOUBLE, &v3);
      MPI_Type_commit(&v3);
      MPI_Type_size(v3, &sz);
      CHECK_INT(sz, 3 * (int) sizeof(double));

      if (np > 1)
      {
         double out[3], in[3] = { -1, -1, -1 };
         int right = (me + 1) % np, left = (me + np - 1) % np;
         MPI_Request r;
         for (i = 0; i < 3; i++) { out[i] = me + i * 0.25; }
         MPI_Irecv(in, 1, v3, left, 1, MPI_COMM_WORLD, &r);
         MPI_Send(out, 1, v3, right, 1, MPI_COMM_WORLD);
         MPI_Wait(&r, MPI_STATUS_IGNORE);
         for (i = 0; i < 3; i++) { CHECK_DBL(in[i], left + i * 0.25, 0.0); }
      }
      MPI_Type_free(&v3);
   }

   /* ---- vector: every other element of a 10-element row ----------------- */
   {
      MPI_Datatype strided;
      double src[10], dst[10];
      MPI_Type_vector(5, 1, 2, MPI_DOUBLE, &strided);
      MPI_Type_commit(&strided);
      MPI_Type_size(strided, &sz);
      CHECK_INT(sz, 5 * (int) sizeof(double));

      for (i = 0; i < 10; i++) { src[i] = me * 10.0 + i; dst[i] = -1.0; }
      if (np > 1)
      {
         int right = (me + 1) % np, left = (me + np - 1) % np;
         MPI_Request r;
         MPI_Irecv(dst, 1, strided, left, 2, MPI_COMM_WORLD, &r);
         MPI_Send(src, 1, strided, right, 2, MPI_COMM_WORLD);
         MPI_Wait(&r, MPI_STATUS_IGNORE);
         for (i = 0; i < 10; i += 2) { CHECK_DBL(dst[i], left * 10.0 + i, 0.0); }
         for (i = 1; i < 10; i += 2) { CHECK_DBL(dst[i], -1.0, 0.0); }
      }
      MPI_Type_free(&strided);
   }

   /* ---- hvector: same, with a byte stride ------------------------------- */
   {
      MPI_Datatype hv;
      MPI_Type_create_hvector(4, 1, (MPI_Aint)(2 * sizeof(int)), MPI_INT, &hv);
      MPI_Type_commit(&hv);
      MPI_Type_size(hv, &sz);
      CHECK_INT(sz, 4 * (int) sizeof(int));

      if (np > 1)
      {
         int src[8], dst[8];
         int right = (me + 1) % np, left = (me + np - 1) % np;
         MPI_Request r;
         for (i = 0; i < 8; i++) { src[i] = me * 8 + i; dst[i] = -1; }
         MPI_Irecv(dst, 1, hv, left, 3, MPI_COMM_WORLD, &r);
         MPI_Send(src, 1, hv, right, 3, MPI_COMM_WORLD);
         MPI_Wait(&r, MPI_STATUS_IGNORE);
         for (i = 0; i < 8; i += 2) { CHECK_INT(dst[i], left * 8 + i); }
         for (i = 1; i < 8; i += 2) { CHECK_INT(dst[i], -1); }
      }
      MPI_Type_free(&hv);
   }

   /* ---- struct, built with absolute addresses and MPI_BOTTOM ------------ */
   if (np > 1)
   {
      /* Two structs, not one. Absolute displacements mean the datatype names a
         specific object, so sending and receiving "into p" would make the send
         buffer and the receive buffer the same memory -- which MPI forbids, and
         which here shows up as a peer's data arriving before we have packed our
         own. Send from p, receive into q. */
      particle p, q;
      MPI_Datatype ptype, qtype, types[3] = { MPI_INT, MPI_DOUBLE, MPI_CHAR };
      int blk[3] = { 1, 3, 1 };
      MPI_Aint pd[3], qd[3];
      int right = (me + 1) % np, left = (me + np - 1) % np;
      MPI_Request r;

      p.id = me * 1000; p.tag = (char)('A' + me % 26);
      for (i = 0; i < 3; i++) { p.x[i] = me + i / 8.0; }
      q.id = -1; q.tag = '?';
      for (i = 0; i < 3; i++) { q.x[i] = -1.0; }

      MPI_Get_address(&p.id,   &pd[0]);
      MPI_Get_address(&p.x[0], &pd[1]);
      MPI_Get_address(&p.tag,  &pd[2]);
      MPI_Get_address(&q.id,   &qd[0]);
      MPI_Get_address(&q.x[0], &qd[1]);
      MPI_Get_address(&q.tag,  &qd[2]);
      MPI_Type_create_struct(3, blk, pd, types, &ptype);
      MPI_Type_create_struct(3, blk, qd, types, &qtype);
      MPI_Type_commit(&ptype);
      MPI_Type_commit(&qtype);
      MPI_Type_size(ptype, &sz);
      CHECK_INT(sz, (int)(sizeof(int) + 3 * sizeof(double) + 1));

      /* absolute displacements mean the buffer argument is MPI_BOTTOM */
      MPI_Irecv(MPI_BOTTOM, 1, qtype, left,  4, MPI_COMM_WORLD, &r);
      MPI_Send (MPI_BOTTOM, 1, ptype, right, 4, MPI_COMM_WORLD);
      MPI_Wait(&r, MPI_STATUS_IGNORE);

      CHECK_INT(q.id, left * 1000);
      CHECK_INT(q.tag, 'A' + left % 26);
      for (i = 0; i < 3; i++) { CHECK_DBL(q.x[i], left + i / 8.0, 0.0); }
      CHECK_INT(p.id, me * 1000);   /* our own copy must be untouched */
      MPI_Type_free(&ptype);
      MPI_Type_free(&qtype);
   }

   MPI_Finalize();
   return t_fails;
}

TEST_MAIN(rank_main)
