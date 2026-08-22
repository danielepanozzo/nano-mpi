/*****************************************************************************
 * The OS surface nano-mpi sits on: threads, mutexes, condition variables, four
 * atomics, and a handful of calls like "how many cores are there".
 *
 * pthreads is the model, because that is what the implementation was written
 * against; the Windows side is a thin mapping onto SRWLOCK, CONDITION_VARIABLE
 * and _beginthreadex. Nothing here is a portability layer in the grand sense --
 * it is exactly the set of primitives nanompi.c uses, and no more.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *****************************************************************************/

#ifndef NMPI_PORT_H
#define NMPI_PORT_H

#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>

/*===========================================================================*/
#if defined(_WIN32)
/*===========================================================================*/

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#include <intrin.h>

/* ---- mutex: SRWLOCK, always taken exclusively --------------------------- */
typedef SRWLOCK nmpi_mutex_t;
#define NMPI_MUTEX_INIT SRWLOCK_INIT

static __inline void nmpi_mutex_init(nmpi_mutex_t *m)    { InitializeSRWLock(m); }
static __inline void nmpi_mutex_lock(nmpi_mutex_t *m)    { AcquireSRWLockExclusive(m); }
static __inline void nmpi_mutex_unlock(nmpi_mutex_t *m)  { ReleaseSRWLockExclusive(m); }
static __inline void nmpi_mutex_destroy(nmpi_mutex_t *m) { (void) m; }  /* nothing to release */

/* ---- condition variable ------------------------------------------------- */
typedef CONDITION_VARIABLE nmpi_cond_t;
#define NMPI_COND_INIT CONDITION_VARIABLE_INIT

static __inline void nmpi_cond_init(nmpi_cond_t *c)      { InitializeConditionVariable(c); }
static __inline void nmpi_cond_broadcast(nmpi_cond_t *c) { WakeAllConditionVariable(c); }
static __inline void nmpi_cond_destroy(nmpi_cond_t *c)   { (void) c; }
static __inline void nmpi_cond_wait(nmpi_cond_t *c, nmpi_mutex_t *m)
{
   SleepConditionVariableSRW(c, m, INFINITE, 0);
}

/* ---- threads ------------------------------------------------------------ */
typedef HANDLE nmpi_thread_t;
typedef void *(*nmpi_thread_fn)(void *);

/* _beginthreadex wants unsigned __stdcall(void *); pthreads wants
   void *(void *). One trampoline reconciles them. */
typedef struct { nmpi_thread_fn fn; void *arg; } nmpi_thread_start;

static unsigned __stdcall nmpi_thread_trampoline(void *p)
{
   nmpi_thread_start s = *(nmpi_thread_start *) p;
   free(p);
   s.fn(s.arg);
   return 0;
}

static __inline int nmpi_thread_create(nmpi_thread_t *t, nmpi_thread_fn fn, void *arg)
{
   nmpi_thread_start *s = (nmpi_thread_start *) malloc(sizeof *s);
   uintptr_t h;
   if (!s) { return 1; }
   s->fn = fn; s->arg = arg;
   /* _beginthreadex, not CreateThread: the CRT needs its per-thread state */
   h = _beginthreadex(NULL, 0, nmpi_thread_trampoline, s, 0, NULL);
   if (!h) { free(s); return 1; }
   *t = (HANDLE) h;
   return 0;
}

static __inline int nmpi_thread_join(nmpi_thread_t t)
{
   WaitForSingleObject(t, INFINITE);
   CloseHandle(t);
   return 0;
}

static __inline int nmpi_thread_detach(nmpi_thread_t t) { CloseHandle(t); return 0; }

/* ---- atomics ------------------------------------------------------------
 * Interlocked* rather than <stdatomic.h>: it is available on every MSVC that
 * can build this, and these four operations are all the implementation needs.
 * Every Interlocked call is a full barrier, which is stronger than the acquire
 * or relaxed ordering asked for and therefore always correct. */
static __inline int nmpi_atomic_load_acq(volatile int *p)
{
   return (int) InterlockedCompareExchange((volatile LONG *) p, 0, 0);
}
static __inline void nmpi_atomic_store_rel(volatile int *p, int v)
{
   InterlockedExchange((volatile LONG *) p, (LONG) v);
}
static __inline int nmpi_atomic_inc_acqrel(volatile int *p)
{
   return (int) InterlockedIncrement((volatile LONG *) p);
}
static __inline void nmpi_atomic_fence(void) { MemoryBarrier(); }

/* ---- odds and ends ------------------------------------------------------ */
static __inline void nmpi_cpu_relax(void) { YieldProcessor(); }
static __inline void nmpi_yield(void)     { SwitchToThread(); }
static __inline void nmpi_sleep_ms(int ms) { Sleep((DWORD) ms); }

static __inline int nmpi_cpu_count(void)
{
   SYSTEM_INFO si;
   GetSystemInfo(&si);
   return (int) si.dwNumberOfProcessors;
}

static __inline void nmpi_hostname(char *buf, size_t n)
{
   DWORD len = (DWORD) n;
   /* GetComputerName, not gethostname: the latter needs Winsock started up,
      and a library has no business calling WSAStartup on its caller's behalf */
   if (!GetComputerNameA(buf, &len)) { len = 0; }
   if (len == 0 && n) { buf[0] = 0; }
}

static __inline double nmpi_wtime(void)
{
   LARGE_INTEGER f, t;
   QueryPerformanceFrequency(&f);
   QueryPerformanceCounter(&t);
   return (double) t.QuadPart / (double) f.QuadPart;
}

static __inline double nmpi_wtick(void)
{
   LARGE_INTEGER f;
   QueryPerformanceFrequency(&f);
   return 1.0 / (double) f.QuadPart;
}

/*===========================================================================*/
#else   /* POSIX */
/*===========================================================================*/

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

typedef pthread_mutex_t nmpi_mutex_t;
#define NMPI_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER

static inline void nmpi_mutex_init(nmpi_mutex_t *m)    { pthread_mutex_init(m, NULL); }
static inline void nmpi_mutex_lock(nmpi_mutex_t *m)    { pthread_mutex_lock(m); }
static inline void nmpi_mutex_unlock(nmpi_mutex_t *m)  { pthread_mutex_unlock(m); }
static inline void nmpi_mutex_destroy(nmpi_mutex_t *m) { pthread_mutex_destroy(m); }

typedef pthread_cond_t nmpi_cond_t;
#define NMPI_COND_INIT PTHREAD_COND_INITIALIZER

static inline void nmpi_cond_init(nmpi_cond_t *c)      { pthread_cond_init(c, NULL); }
static inline void nmpi_cond_broadcast(nmpi_cond_t *c) { pthread_cond_broadcast(c); }
static inline void nmpi_cond_destroy(nmpi_cond_t *c)   { pthread_cond_destroy(c); }
static inline void nmpi_cond_wait(nmpi_cond_t *c, nmpi_mutex_t *m)
{
   pthread_cond_wait(c, m);
}

typedef pthread_t nmpi_thread_t;
typedef void *(*nmpi_thread_fn)(void *);

static inline int nmpi_thread_create(nmpi_thread_t *t, nmpi_thread_fn fn, void *arg)
{
   return pthread_create(t, NULL, fn, arg);
}
static inline int nmpi_thread_join(nmpi_thread_t t)   { return pthread_join(t, NULL); }
static inline int nmpi_thread_detach(nmpi_thread_t t) { return pthread_detach(t); }

static inline int nmpi_atomic_load_acq(volatile int *p)
{
   return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
static inline void nmpi_atomic_store_rel(volatile int *p, int v)
{
   __atomic_store_n(p, v, __ATOMIC_RELEASE);
}
static inline int nmpi_atomic_inc_acqrel(volatile int *p)
{
   return __atomic_add_fetch(p, 1, __ATOMIC_ACQ_REL);
}
static inline void nmpi_atomic_fence(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }

static inline void nmpi_cpu_relax(void)
{
#if defined(__aarch64__) || defined(__arm__)
   __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
   __builtin_ia32_pause();
#else
   /* nothing to hint with; the yield below still bounds the spin */
#endif
}

static inline void nmpi_yield(void) { sched_yield(); }

static inline void nmpi_sleep_ms(int ms)
{
   struct timespec ts;
   ts.tv_sec  = ms / 1000;
   ts.tv_nsec = (long)(ms % 1000) * 1000000L;
   nanosleep(&ts, NULL);
}

static inline int nmpi_cpu_count(void)
{
   long n = sysconf(_SC_NPROCESSORS_ONLN);
   return (n > 0) ? (int) n : 1;
}

static inline void nmpi_hostname(char *buf, size_t n)
{
   if (gethostname(buf, n) != 0 && n) { buf[0] = 0; }
}

static inline double nmpi_wtime(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

static inline double nmpi_wtick(void) { return 1.0e-9; }

#endif /* _WIN32 */

#endif /* NMPI_PORT_H */
