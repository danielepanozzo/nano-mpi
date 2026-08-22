/* Shared scaffolding for the nano-mpi tests.
 *
 * Note what t_fails has to be. Ranks are threads, so a plain file-scope counter
 * would be one counter shared by every rank -- the exact bug SCOPE.md section 5
 * warns about. The test suite is the first place that bites, so it is spelled
 * out here rather than hidden. */
#ifndef NANOMPI_TEST_HARNESS_H
#define NANOMPI_TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "nanompi.h"

#if defined(_MSC_VER)
#  define T_TLS __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define T_TLS _Thread_local
#else
#  define T_TLS __thread
#endif

static T_TLS int t_fails = 0;

#define CHECK(cond, ...)                                                      \
   do {                                                                       \
      if (!(cond)) {                                                          \
         int r_ = nanompi_rank();                                             \
         fprintf(stderr, "  FAIL rank %d  %s:%d  ", r_, __FILE__, __LINE__);  \
         fprintf(stderr, __VA_ARGS__);                                        \
         fputc('\n', stderr);                                                 \
         t_fails++;                                                           \
      }                                                                       \
   } while (0)

#define CHECK_INT(got, want)                                                  \
   CHECK((got) == (want), "%s: got %lld, want %lld", #got,                     \
         (long long)(got), (long long)(want))

#define CHECK_DBL(got, want, tol)                                             \
   CHECK(fabs((double)(got) - (double)(want)) <= (tol),                        \
         "%s: got %.17g, want %.17g", #got, (double)(got), (double)(want))

/* Every test is one program taking an optional rank count. Passing 0 (or
   nothing) takes the default, which is what makes NANOMPI_NUM_RANKS visible. */
#define TEST_MAIN(rank_fn)                                                    \
   int main(int argc, char **argv)                                            \
   {                                                                          \
      int n = (argc > 1) ? atoi(argv[1]) : 0;                                 \
      int rc = nanompi_run(n, rank_fn, argc, argv, NULL);                     \
      if (rc) { fprintf(stderr, "%s: FAILED\n", argv[0]); return 1; }         \
      printf("%s: ok (%d ranks)\n", argv[0], n > 0 ? n : nanompi_default_ranks());        \
      return 0;                                                               \
   }

#endif
