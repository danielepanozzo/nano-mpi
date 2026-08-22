/* Every predefined datatype through every operator that is defined for it. */
#include "harness.h"
#include <stdint.h>
#include <stdbool.h>

/* Thread-local: ranks are threads, so a plain static here would be one
   variable shared by every rank -- see SCOPE.md section 5. */
static T_TLS int g_me, g_np;

/* Integer types: value is (me+1), so SUM is n(n+1)/2, PROD is n!, MAX is np. */
#define CHECK_INT_TYPE(CTYPE, MPITYPE)                                        \
   do {                                                                       \
      CTYPE s = (CTYPE)(g_me + 1), got;                                        \
      long long want, i;                                                       \
      got = 0; MPI_Allreduce(&s, &got, 1, MPITYPE, MPI_SUM, MPI_COMM_WORLD);   \
      for (want = 0, i = 1; i <= g_np; i++) { want += i; }                     \
      CHECK((long long) got == (long long)(CTYPE) want,                        \
            #MPITYPE " SUM: got %lld want %lld",                               \
            (long long) got, (long long)(CTYPE) want);                         \
      got = 0; MPI_Allreduce(&s, &got, 1, MPITYPE, MPI_MAX, MPI_COMM_WORLD);   \
      CHECK((long long) got == (long long)(CTYPE) g_np,                        \
            #MPITYPE " MAX: got %lld", (long long) got);                       \
      got = 0; MPI_Allreduce(&s, &got, 1, MPITYPE, MPI_MIN, MPI_COMM_WORLD);   \
      CHECK((long long) got == 1, #MPITYPE " MIN: got %lld", (long long) got); \
   } while (0)

/* Bitwise and logical, on a value with exactly one bit set. */
#define CHECK_BIT_TYPE(CTYPE, MPITYPE, NBITS)                                 \
   do {                                                                       \
      CTYPE s = (CTYPE)(1u << (unsigned)(g_me % (NBITS))), got;                \
      CTYPE want = 0; int i;                                                   \
      for (i = 0; i < g_np; i++) { want |= (CTYPE)(1u << (unsigned)(i % (NBITS))); } \
      got = 0; MPI_Allreduce(&s, &got, 1, MPITYPE, MPI_BOR, MPI_COMM_WORLD);   \
      CHECK((unsigned long long) got == (unsigned long long) want,             \
            #MPITYPE " BOR: got %llu want %llu",                               \
            (unsigned long long) got, (unsigned long long) want);              \
      got = 0; MPI_Allreduce(&s, &got, 1, MPITYPE, MPI_BAND, MPI_COMM_WORLD);  \
      CHECK((unsigned long long) got ==                                        \
            (unsigned long long)(g_np == 1 ? s : (CTYPE) 0),                   \
            #MPITYPE " BAND: got %llu", (unsigned long long) got);             \
      got = 0; MPI_Allreduce(&s, &got, 1, MPITYPE, MPI_LOR, MPI_COMM_WORLD);   \
      CHECK((long long) got == 1, #MPITYPE " LOR: got %lld", (long long) got); \
   } while (0)

#define CHECK_FLOAT_TYPE(CTYPE, MPITYPE)                                      \
   do {                                                                       \
      CTYPE s = (CTYPE)(g_me + 1), got = 0;                                    \
      double want = (double) g_np * (g_np + 1) / 2.0;                          \
      MPI_Allreduce(&s, &got, 1, MPITYPE, MPI_SUM, MPI_COMM_WORLD);            \
      CHECK_DBL((double) got, want, 1e-9 * (want + 1.0));                      \
      got = 0; MPI_Allreduce(&s, &got, 1, MPITYPE, MPI_MAX, MPI_COMM_WORLD);   \
      CHECK_DBL((double) got, (double) g_np, 0.0);                             \
      got = 0; MPI_Allreduce(&s, &got, 1, MPITYPE, MPI_PROD, MPI_COMM_WORLD);  \
      { double p = 1.0; int i; for (i = 1; i <= g_np; i++) { p *= i; }         \
        CHECK_DBL((double) got, p, 1e-9 * (p + 1.0)); }                        \
   } while (0)

static int rank_main(int argc, char **argv, void *user)
{
   (void) argc; (void) argv; (void) user;
   MPI_Init(NULL, NULL);
   MPI_Comm_rank(MPI_COMM_WORLD, &g_me);
   MPI_Comm_size(MPI_COMM_WORLD, &g_np);

   CHECK_INT_TYPE(int,                MPI_INT);
   CHECK_INT_TYPE(long,               MPI_LONG);
   CHECK_INT_TYPE(long long,          MPI_LONG_LONG_INT);
   CHECK_INT_TYPE(short,              MPI_SHORT);
   CHECK_INT_TYPE(unsigned,           MPI_UNSIGNED);
   CHECK_INT_TYPE(unsigned long,      MPI_UNSIGNED_LONG);
   CHECK_INT_TYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG);
   CHECK_INT_TYPE(unsigned short,     MPI_UNSIGNED_SHORT);
   CHECK_INT_TYPE(int32_t,            MPI_INT32_T);
   CHECK_INT_TYPE(int64_t,            MPI_INT64_T);
   CHECK_INT_TYPE(uint32_t,           MPI_UINT32_T);
   CHECK_INT_TYPE(uint64_t,           MPI_UINT64_T);
   CHECK_INT_TYPE(MPI_Aint,           MPI_AINT);
   CHECK_INT_TYPE(MPI_Count,          MPI_COUNT);

   CHECK_BIT_TYPE(int,           MPI_INT,           30);
   CHECK_BIT_TYPE(unsigned char, MPI_UNSIGNED_CHAR,  8);
   CHECK_BIT_TYPE(unsigned char, MPI_BYTE,           8);
   CHECK_BIT_TYPE(uint16_t,      MPI_UINT16_T,      16);
   CHECK_BIT_TYPE(int64_t,       MPI_INT64_T,       30);

   CHECK_FLOAT_TYPE(float,       MPI_FLOAT);
   CHECK_FLOAT_TYPE(double,      MPI_DOUBLE);
   CHECK_FLOAT_TYPE(long double, MPI_LONG_DOUBLE);

   /* MPI_C_BOOL under LOR / LAND */
   {
      bool s = (g_me == 0), got = false;
      MPI_Allreduce(&s, &got, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
      CHECK_INT((int) got, 1);
      MPI_Allreduce(&s, &got, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
      CHECK_INT((int) got, g_np == 1 ? 1 : 0);
   }

   /* complex, stored as adjacent (re, im) */
   {
      double s[2], got[2] = { -1.0, -1.0 };
      double wr = 0.0, wi = 0.0;
      int i;
      s[0] = g_me + 1.0; s[1] = -(g_me + 1.0);
      for (i = 0; i < g_np; i++) { wr += i + 1.0; wi -= i + 1.0; }
      MPI_Allreduce(s, got, 1, MPI_C_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
      CHECK_DBL(got[0], wr, 1e-9);
      CHECK_DBL(got[1], wi, 1e-9);
   }
   {
      /* PROD of (0 + 1i) per rank rotates by 90 degrees each time */
      double s[2], got[2];
      double wr = 1.0, wi = 0.0;
      int i;
      s[0] = 0.0; s[1] = 1.0;
      for (i = 0; i < g_np; i++) { double r = wr * 0.0 - wi * 1.0;
                                   double m = wr * 1.0 + wi * 0.0;
                                   wr = r; wi = m; }
      MPI_Allreduce(s, got, 1, MPI_C_DOUBLE_COMPLEX, MPI_PROD, MPI_COMM_WORLD);
      CHECK_DBL(got[0], wr, 1e-9);
      CHECK_DBL(got[1], wi, 1e-9);
   }

   /* the MPI_CXX_* complex types behave like their C counterparts */
   {
      float s[2], got[2] = { -1.0f, -1.0f };
      double wr = 0.0;
      int i;
      s[0] = (float)(g_me + 1); s[1] = 0.0f;
      for (i = 0; i < g_np; i++) { wr += i + 1.0; }
      MPI_Allreduce(s, got, 1, MPI_CXX_FLOAT_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
      CHECK_DBL((double) got[0], wr, 1e-4 * (wr + 1.0));
      CHECK_DBL((double) got[1], 0.0, 1e-6);
   }
   {
      bool s = (g_me == 0), got = false;
      MPI_Allreduce(&s, &got, 1, MPI_CXX_BOOL, MPI_LOR, MPI_COMM_WORLD);
      CHECK_INT((int) got, 1);
   }

   /* MAXLOC / MINLOC, including the smaller-index tie-break */
   {
      MPI_Double_int s, got;
      s.value = (double)((g_me * 7) % 13);
      s.index = g_me;
      MPI_Allreduce(&s, &got, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
      {
         double best = -1e300; int bi = -1, i;
         for (i = 0; i < g_np; i++)
         {
            double v = (double)((i * 7) % 13);
            if (v > best) { best = v; bi = i; }
         }
         CHECK_DBL(got.value, best, 0.0);
         CHECK_INT(got.index, bi);
      }
      /* everyone contributes the same value: the smallest index must win */
      s.value = 42.0; s.index = g_me;
      MPI_Allreduce(&s, &got, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
      CHECK_INT(got.index, 0);
      MPI_Allreduce(&s, &got, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
      CHECK_INT(got.index, 0);
   }
   {
      MPI_2int s, got;
      s.value = g_np - g_me; s.index = g_me;
      MPI_Allreduce(&s, &got, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
      CHECK_INT(got.value, 1);
      CHECK_INT(got.index, g_np - 1);
   }

   /* arrays, not just scalars */
   {
      int s[6], got[6], i;
      for (i = 0; i < 6; i++) { s[i] = (g_me + 1) * (i + 1); }
      MPI_Allreduce(s, got, 6, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      for (i = 0; i < 6; i++)
      {
         CHECK_INT(got[i], (g_np * (g_np + 1) / 2) * (i + 1));
      }
   }

   MPI_Finalize();
   return t_fails;
}

TEST_MAIN(rank_main)
