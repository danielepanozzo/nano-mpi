/******************************************************************************
 * nano-mpi: MPI over POSIX threads. Each rank IS a thread of one process, so
 * MPI_Send between ranks is a memcpy and there is no launcher.
 *
 * The supported subset and the deliberate omissions are specified in SCOPE.md.
 * Section 5 of it -- ranks are threads, so your file-scope mutable state is
 * shared rather than per-rank -- is the one every caller has to read.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *****************************************************************************/

#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>

#include "mpi.h"
#include "nanompi.h"

#if defined(__cplusplus)
#  define NANOMPI_TLS thread_local
#elif defined(_MSC_VER)
#  define NANOMPI_TLS __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define NANOMPI_TLS _Thread_local
#elif defined(__GNUC__) || defined(__clang__)
#  define NANOMPI_TLS __thread
#else
#  error "nano-mpi needs thread-local storage; no supported spelling found"
#endif

#define NANOMPI_UNUSED(x) ((void) (x))

#define NMPI_MAX_COMMS   4096
#define NMPI_MAX_TYPES   4096
#define NMPI_REQS_PER_RANK 65536
#define NMPI_FIRST_USER_TYPE 64   /* predefined datatypes occupy 1..38 */

/*--------------------------------------------------------------------------
 * Datatypes
 *--------------------------------------------------------------------------*/
typedef struct
{
   int       used;
   int       nblk;
   size_t   *len;    /* bytes in each block */
   intptr_t *disp;   /* byte displacement; absolute when 'abs' is set */
   int       abs;    /* built from MPI_Address -> absolute addresses */
   size_t    size;   /* total bytes for one element */
} nmpi_dt;

static nmpi_dt         g_dt[NMPI_MAX_TYPES];
static pthread_mutex_t g_dt_mtx = PTHREAD_MUTEX_INITIALIZER;

/* The full predefined set of MPI-3 C. Sizes are of the local C types, which is
   the only sensible answer when every rank is a thread of the same process. */
static size_t predefined_size(int dt)
{
   switch (dt)
   {
      case MPI_CHAR:                   return sizeof(char);
      case MPI_SIGNED_CHAR:            return sizeof(signed char);
      case MPI_UNSIGNED_CHAR:          return sizeof(unsigned char);
      case MPI_BYTE:                   return 1;
      case MPI_WCHAR:                  return sizeof(wchar_t);
      case MPI_SHORT:                  return sizeof(short);
      case MPI_UNSIGNED_SHORT:         return sizeof(unsigned short);
      case MPI_INT:                    return sizeof(int);
      case MPI_UNSIGNED:               return sizeof(unsigned int);
      case MPI_LONG:                   return sizeof(long);
      case MPI_UNSIGNED_LONG:          return sizeof(unsigned long);
      case MPI_LONG_LONG_INT:          return sizeof(long long);
      case MPI_UNSIGNED_LONG_LONG:     return sizeof(unsigned long long);
      case MPI_FLOAT:                  return sizeof(float);
      case MPI_DOUBLE:                 return sizeof(double);
      case MPI_LONG_DOUBLE:            return sizeof(long double);
      case MPI_C_BOOL:                 return sizeof(bool);
      case MPI_INT8_T:                 return 1;
      case MPI_INT16_T:                return 2;
      case MPI_INT32_T:                return 4;
      case MPI_INT64_T:                return 8;
      case MPI_UINT8_T:                return 1;
      case MPI_UINT16_T:               return 2;
      case MPI_UINT32_T:               return 4;
      case MPI_UINT64_T:               return 8;
      case MPI_C_FLOAT_COMPLEX:        return 2 * sizeof(float);
      case MPI_C_DOUBLE_COMPLEX:       return 2 * sizeof(double);
      case MPI_C_LONG_DOUBLE_COMPLEX:  return 2 * sizeof(long double);
      case MPI_AINT:                   return sizeof(MPI_Aint);
      case MPI_OFFSET:                 return sizeof(MPI_Offset);
      case MPI_COUNT:                  return sizeof(MPI_Count);
      case MPI_PACKED:                 return 1;
      case MPI_FLOAT_INT:              return sizeof(MPI_Float_int);
      case MPI_DOUBLE_INT:             return sizeof(MPI_Double_int);
      case MPI_LONG_INT:               return sizeof(MPI_Long_int);
      case MPI_SHORT_INT:              return sizeof(MPI_Short_int);
      case MPI_2INT:                   return sizeof(MPI_2int);
      case MPI_LONG_DOUBLE_INT:        return sizeof(MPI_Long_double_int);
      default:                         return 0;
   }
}

static nmpi_dt *dt_get(int dt)
{
   if (dt >= NMPI_FIRST_USER_TYPE && dt < NMPI_MAX_TYPES && g_dt[dt].used)
   {
      return &g_dt[dt];
   }
   return NULL;
}

static size_t dt_size(int dt)
{
   nmpi_dt *d = dt_get(dt);
   return d ? d->size : predefined_size(dt);
}

static int dt_alloc(void)
{
   int i;
   pthread_mutex_lock(&g_dt_mtx);
   for (i = NMPI_FIRST_USER_TYPE; i < NMPI_MAX_TYPES; i++)
   {
      if (!g_dt[i].used) { g_dt[i].used = 1; pthread_mutex_unlock(&g_dt_mtx); return i; }
   }
   pthread_mutex_unlock(&g_dt_mtx);
   fprintf(stderr, "nano-mpi: out of datatype handles\n");
   abort();
}

/* number of bytes a (buf,count,dt) message occupies on the wire */
static size_t msg_bytes(int count, int dt)
{
   return (size_t)count * dt_size(dt);
}

/* pack (buf,count,dt) into contiguous 'out' (out must hold msg_bytes) */
static void pack(const void *buf, int count, int dt, char *out)
{
   nmpi_dt *d = dt_get(dt);
   int c, b;
   size_t off = 0;

   if (!d)
   {
      memcpy(out, buf, msg_bytes(count, dt));
      return;
   }
   for (c = 0; c < count; c++)
   {
      const char *base = (const char *) buf + (size_t) c * d->size;
      for (b = 0; b < d->nblk; b++)
      {
         /* An absolute type was built from MPI_Get_address and is used with
            MPI_BOTTOM, so each displacement IS the address; a relative one is
            an offset from the caller's buffer. Keep the two apart -- adding an
            offset to the null pointer MPI_BOTTOM is undefined behaviour, even
            though every compiler happens to do the obvious thing.
            (An absolute type therefore names one specific object; count > 1
            with one is not meaningful and is not supported.) */
         const char *src = d->abs ? (const char *) (uintptr_t) d->disp[b]
                                  : base + d->disp[b];
         memcpy(out + off, src, d->len[b]);
         off += d->len[b];
      }
   }
}

/* 'avail' is what the sender actually put on the wire; a receive may be posted
   with a larger capacity than the message, which is legal in MPI. */
static void unpack(const char *in, void *buf, int count, int dt, size_t avail)
{
   nmpi_dt *d = dt_get(dt);
   int c, b;
   size_t off = 0;

   if (!d)
   {
      size_t want = msg_bytes(count, dt);
      memcpy(buf, in, (avail < want) ? avail : want);
      return;
   }
   for (c = 0; c < count; c++)
   {
      char *base = (char *) buf + (size_t) c * d->size;
      for (b = 0; b < d->nblk; b++)
      {
         size_t n = d->len[b];
         char *dst = d->abs ? (char *) (uintptr_t) d->disp[b]   /* see pack() */
                            : base + d->disp[b];
         if (off + n > avail) { n = (off < avail) ? (avail - off) : 0; }
         if (n) { memcpy(dst, in + off, n); }
         off += d->len[b];
      }
   }
}

/*--------------------------------------------------------------------------
 * Communicators
 *--------------------------------------------------------------------------*/
typedef struct
{
   int              used;
   int              size;
   int             *world;   /* comm rank -> world rank */
   int             *inv;     /* world rank -> comm rank (-1 if not a member) */
   /* sense-reversing barrier */
   pthread_mutex_t  mtx;
   pthread_cond_t   cv;
   int              count;
   int              generation;   /* monotonic; never reset, see comm_alloc */
   int              inited;       /* mtx/cv constructed once per slot */
   /* scratch for collectives, indexed by comm rank */
   const void     **ptr;
   size_t          *len;
   /* root-only arguments republished for scatter-style collectives */
   const void      *rbuf;
   const void      *rcnt;
   const void      *rdsp;
   int        rn;
} nmpi_comm;

static nmpi_comm       g_comm[NMPI_MAX_COMMS];
static pthread_mutex_t g_comm_mtx = PTHREAD_MUTEX_INITIALIZER;

/* Groups: a plain list of world ranks. */
typedef struct { int used; int size; int *world; } nmpi_group;
static nmpi_group      g_group[NMPI_MAX_COMMS];
static pthread_mutex_t g_group_mtx = PTHREAD_MUTEX_INITIALIZER;

static nmpi_group *group_get(MPI_Group g)
{
   if (g < 1 || g >= NMPI_MAX_COMMS || !g_group[g].used) { return NULL; }
   return &g_group[g];
}

static int group_alloc(int size, const int *world_ranks)
{
   int i, h = -1;
   pthread_mutex_lock(&g_group_mtx);
   for (i = 1; i < NMPI_MAX_COMMS; i++) { if (!g_group[i].used) { h = i; break; } }
   if (h < 0)
   {
      pthread_mutex_unlock(&g_group_mtx);
      fprintf(stderr, "nano-mpi: out of group handles\n"); abort();
   }
   g_group[h].used  = 1;
   g_group[h].size  = size;
   g_group[h].world = (int *) malloc(sizeof(int) * (size_t)(size > 0 ? size : 1));
   for (i = 0; i < size; i++) { g_group[h].world[i] = world_ranks[i]; }
   pthread_mutex_unlock(&g_group_mtx);
   return h;
}

/* User-defined reduction ops, from MPI_Op_create. */
typedef void (*nmpi_user_fn)(void *, void *, int *, MPI_Datatype *);
#define NMPI_FIRST_USER_OP 32   /* predefined ops occupy 1..13 */
#define NMPI_MAX_OPS       128
typedef struct { int used; nmpi_user_fn fn; } nmpi_op;
static nmpi_op         g_op[NMPI_MAX_OPS];
static pthread_mutex_t g_op_mtx = PTHREAD_MUTEX_INITIALIZER;

static nmpi_user_fn op_user(int op)
{
   if (op >= NMPI_FIRST_USER_OP && op < NMPI_MAX_OPS && g_op[op].used) { return g_op[op].fn; }
   return NULL;
}

static int             g_nranks = 1;
static int universe_init(int nranks);
static void universe_free(void);
static int             g_universe_up = 0;
/* the universe was created implicitly for a single thread, not by a team */
static int             g_auto_universe = 0;
static pthread_mutex_t g_auto_mtx = PTHREAD_MUTEX_INITIALIZER;
static void ensure_universe(void);
static NANOMPI_TLS int g_myrank = 0;

int  nanompi_rank(void)   { return g_myrank; }
int  nanompi_nranks(void) { return g_nranks; }

/*--------------------------------------------------------------------------
 * Optional deadlock watchdog: set NANOMPI_WATCHDOG=<seconds>. Records what each
 * rank is blocked on so a hang reports itself instead of needing a debugger.
 *--------------------------------------------------------------------------*/
enum { NMPI_ST_RUN = 0, NMPI_ST_BARRIER, NMPI_ST_RECV, NMPI_ST_PROBE };
typedef struct { int op, a, b, c; unsigned long seq; } nmpi_state;
static nmpi_state *g_state = NULL;

#define SET_STATE(o, x, y, z)                                     \
   do {                                                           \
      if (g_state) {                                              \
         nmpi_state *s_ = &g_state[g_myrank];                     \
         s_->op = (o); s_->a = (x); s_->b = (y); s_->c = (z);     \
         s_->seq++;                                               \
      }                                                           \
   } while (0)

static const char *state_name(int op)
{
   switch (op)
   {
      case NMPI_ST_BARRIER: return "barrier";
      case NMPI_ST_RECV:    return "recv";
      case NMPI_ST_PROBE:   return "probe";
      default:              return "running";
   }
}

static nmpi_comm *comm_get(MPI_Comm c)
{
   /* A process is perfectly usable with no team at all: one thread, one rank.
      That thread never called nanompi_run/team_start, so give it a world here
      rather than letting COMM_WORLD report size 0 and rank -1. */
   if (!g_universe_up) { ensure_universe(); }
   if (c <= 0 || c >= NMPI_MAX_COMMS || !g_comm[c].used) { return NULL; }
   return &g_comm[c];
}

/* rank of the calling thread inside comm c (COMM_SELF is per-thread) */
static int comm_rank_of(MPI_Comm c)
{
   nmpi_comm *k;
   if (c == MPI_COMM_SELF) { return 0; }
   k = comm_get(c);
   return k ? k->inv[g_myrank] : -1;
}

static int comm_size_of(MPI_Comm c)
{
   nmpi_comm *k;
   if (c == MPI_COMM_SELF) { return 1; }
   k = comm_get(c);
   return k ? k->size : 0;
}

static int comm_alloc(int size, const int *world_ranks)
{
   int i, h = -1;
   nmpi_comm *k;

   pthread_mutex_lock(&g_comm_mtx);
   for (i = 3; i < NMPI_MAX_COMMS; i++)   /* 0 = NULL, 1 = WORLD, 2 = SELF */
   {
      if (!g_comm[i].used) { h = i; break; }
   }
   if (h < 0)
   {
      pthread_mutex_unlock(&g_comm_mtx);
      fprintf(stderr, "nano-mpi: out of communicator handles\n");
      abort();
   }
   k = &g_comm[h];
   k->used = 1;
   k->size = size;
   k->world = (int *) malloc(sizeof(int) * (size_t) size);
   k->inv   = (int *) malloc(sizeof(int) * (size_t) g_nranks);
   k->ptr   = (const void **) malloc(sizeof(void *) * (size_t) size);
   k->len   = (size_t *) malloc(sizeof(size_t) * (size_t) size);
   for (i = 0; i < g_nranks; i++) { k->inv[i] = -1; }
   for (i = 0; i < size; i++) { k->world[i] = world_ranks[i]; k->inv[world_ranks[i]] = i; }
   if (!k->inited)
   {
      /* re-initialising a live pthread mutex is undefined, and this slot may be
         one that a previous communicator freed */
      pthread_mutex_init(&k->mtx, NULL);
      pthread_cond_init(&k->cv, NULL);
      k->inited = 1;
   }
   /* generation is deliberately NOT reset. A rank can still be spinning on the
      old communicator's barrier when the handle is recycled; a monotonic
      counter guarantees it sees a change and leaves, where a reset to the value
      it captured would leave it spinning forever. */
   k->count = 0;
   pthread_mutex_unlock(&g_comm_mtx);
   return h;
}

/* Sense-reversing barrier on atomics. Collectives here are fine-grained (PCG
   does several Allreduces per iteration), and a condvar broadcast per barrier
   wakes every thread -- far too expensive. Spin briefly, then yield. */
static void comm_barrier(MPI_Comm c)
{
   nmpi_comm *k = comm_get(c);
   int gen, cnt, spins = 0;

   if (c == MPI_COMM_SELF || !k || k->size <= 1) { return; }

#ifdef NMPI_LOCK_BARRIER
   pthread_mutex_lock(&k->mtx);
   gen = k->generation;
   if (++k->count == k->size)
   {
      k->count = 0;
      k->generation++;
      pthread_cond_broadcast(&k->cv);
   }
   else
   {
      while (gen == k->generation) { pthread_cond_wait(&k->cv, &k->mtx); }
   }
   pthread_mutex_unlock(&k->mtx);
   (void) cnt; (void) spins;
#else
   gen = __atomic_load_n(&k->generation, __ATOMIC_ACQUIRE);
   cnt = __atomic_add_fetch(&k->count, 1, __ATOMIC_ACQ_REL);

   if (cnt == k->size)
   {
      __atomic_store_n(&k->count, 0, __ATOMIC_RELAXED);
      __atomic_add_fetch(&k->generation, 1, __ATOMIC_ACQ_REL);
      return;
   }
   while (__atomic_load_n(&k->generation, __ATOMIC_ACQUIRE) == gen)
   {
      if (++spins < 4096)
      {
#if defined(__aarch64__)
         __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
         __builtin_ia32_pause();
#endif
      }
      else { sched_yield(); spins = 4096; }
   }
#endif
}

/*--------------------------------------------------------------------------
 * Point-to-point: per-world-rank inbox of eagerly buffered messages
 *--------------------------------------------------------------------------*/
/* MPI matches an incoming message against receives in the order they were
 * POSTED, not the order they are waited on. Real code depends on it: creating
 * two comm handles and then completes them in the opposite order. So we keep
 * the standard pair of queues per rank -- posted receives, and unexpected
 * messages -- and match at post/send time. */
typedef struct nmpi_msg
{
   struct nmpi_msg *next;
   int              comm;
   int              src;    /* comm rank of sender */
   int              tag;
   char            *data;
   size_t           nbytes;
} nmpi_msg;

typedef struct nmpi_pending
{
   struct nmpi_pending *next;
   int     comm, src, tag;  /* what this receive is waiting for */
   void   *buf;
   int     count, dt;
   int     done;
   int     act_src, act_tag;
   size_t  nbytes;          /* bytes actually delivered */
} nmpi_pending;

typedef struct
{
   pthread_mutex_t mtx;
   pthread_cond_t  cv;
   nmpi_msg       *umq_head, *umq_tail;   /* arrived, not yet matched */
   nmpi_pending   *prq_head, *prq_tail;   /* posted, not yet satisfied */
} nmpi_inbox;

static nmpi_inbox *g_inbox = NULL;

static int match_ok(int p_comm, int p_src, int p_tag, int m_comm, int m_src, int m_tag)
{
   if (p_comm != m_comm) { return 0; }
   if (p_src != MPI_ANY_SOURCE && p_src != m_src) { return 0; }
   if (p_tag != MPI_ANY_TAG    && p_tag != m_tag) { return 0; }
   return 1;
}


/* Diagnostic: dump the matching queues of 'world_rank'. Only called when a
   truncation has already made the run fatal. */
static void dump_queues(int world_rank)
{
   nmpi_inbox   *b = &g_inbox[world_rank];
   nmpi_pending *p;
   nmpi_msg     *m;
   int i;

   fprintf(stderr, "  posted receives on rank %d (in order):\n", world_rank);
   for (i = 0, p = b->prq_head; p; p = p->next, i++)
   {
      fprintf(stderr, "    [%d] comm=%d src=%d tag=%d capacity=%zu bytes\n",
              i, p->comm, p->src, p->tag, msg_bytes(p->count, p->dt));
   }
   if (!i) { fprintf(stderr, "    (none)\n"); }
   fprintf(stderr, "  unexpected messages on rank %d (in order):\n", world_rank);
   for (i = 0, m = b->umq_head; m; m = m->next, i++)
   {
      fprintf(stderr, "    [%d] comm=%d src=%d tag=%d %zu bytes\n",
              i, m->comm, m->src, m->tag, m->nbytes);
   }
   if (!i) { fprintf(stderr, "    (none)\n"); }
}

/* copy an arrived message into a posted receive; caller holds the inbox lock */
static void deliver(nmpi_pending *p, int m_src, int m_tag, const char *data, size_t nbytes)
{
   size_t cap = msg_bytes(p->count, p->dt);

   if (nbytes > cap)
   {
      fprintf(stderr,
              "nano-mpi: MESSAGE TRUNCATED on rank %d: comm=%d src=%d tag=%d "
              "sent=%zu bytes, recv capacity=%zu bytes (count=%d dt=%d)\n",
              g_myrank, p->comm, m_src, m_tag, nbytes, cap, p->count, p->dt);
      dump_queues(g_myrank);
      abort();
   }
   if (nbytes > 0) { unpack(data, p->buf, p->count, p->dt, nbytes); }
   p->act_src = m_src;
   p->act_tag = m_tag;
   p->nbytes  = nbytes;
   p->done    = 1;
}

static void do_send(const void *buf, int count, int dt, int dest, int tag, MPI_Comm comm)
{
   nmpi_comm    *k = comm_get(comm);
   nmpi_inbox   *b;
   nmpi_pending *p, *prev;
   int           dest_world, my_crank;
   size_t        nbytes;

   if (comm == MPI_COMM_SELF) { dest_world = g_myrank; }
   else if (!k) { return; }
   else { dest_world = k->world[dest]; }

   my_crank = comm_rank_of(comm);
   nbytes   = msg_bytes(count, dt);
   b        = &g_inbox[dest_world];

   /* Fast path: if a matching receive is already posted, detach it and copy
      straight into its buffer -- no heap allocation, one copy instead of two.
      This is the common pattern: Irecv posted before the matching Isend. */
#ifndef NMPI_DISABLE_FASTSEND
   pthread_mutex_lock(&b->mtx);
   prev = NULL;
   for (p = b->prq_head; p; prev = p, p = p->next)
   {
      if (match_ok(p->comm, p->src, p->tag, comm, my_crank, tag))
      {
         if (prev) { prev->next = p->next; } else { b->prq_head = p->next; }
         if (b->prq_tail == p) { b->prq_tail = prev; }
         break;
      }
   }
   pthread_mutex_unlock(&b->mtx);

   if (p)
   {
      size_t cap = msg_bytes(p->count, p->dt);
      if (nbytes > cap)
      {
         fprintf(stderr,
                 "nano-mpi: MESSAGE TRUNCATED on rank %d: comm=%d src=%d tag=%d "
                 "sent=%zu bytes, recv capacity=%zu bytes (count=%d dt=%d)\n",
                 dest_world, (int) comm, my_crank, tag, nbytes, cap, p->count, p->dt);
         dump_queues(dest_world);
         abort();
      }
      if (nbytes > 0)
      {
         /* both sides contiguous -> straight memcpy, outside the lock */
         if (!dt_get(dt) && !dt_get(p->dt))
         {
            memcpy(p->buf, buf, nbytes);
         }
         else
         {
            char *tmp = (char *) malloc(nbytes);
            pack(buf, count, dt, tmp);
            unpack(tmp, p->buf, p->count, p->dt, nbytes);
            free(tmp);
         }
      }
      pthread_mutex_lock(&b->mtx);
      p->act_src = my_crank; p->act_tag = tag; p->nbytes = nbytes; p->done = 1;
      pthread_cond_broadcast(&b->cv);
      pthread_mutex_unlock(&b->mtx);
      return;
   }
#endif /* NMPI_DISABLE_FASTSEND */

   /* Slow path: nobody is waiting yet, so buffer the message. */
   {
      char     *data = (nbytes > 0) ? (char *) malloc(nbytes) : NULL;
      nmpi_msg *m    = (nmpi_msg *) malloc(sizeof(nmpi_msg));
      if (nbytes > 0) { pack(buf, count, dt, data); }
      m->next = NULL; m->comm = comm; m->src = my_crank; m->tag = tag;
      m->data = data; m->nbytes = nbytes;

      pthread_mutex_lock(&b->mtx);
      /* re-check: a receive may have been posted while we were packing */
      prev = NULL;
      for (p = b->prq_head; p; prev = p, p = p->next)
      {
         if (match_ok(p->comm, p->src, p->tag, comm, my_crank, tag))
         {
            if (prev) { prev->next = p->next; } else { b->prq_head = p->next; }
            if (b->prq_tail == p) { b->prq_tail = prev; }
            deliver(p, my_crank, tag, m->data, m->nbytes);
            pthread_cond_broadcast(&b->cv);
            pthread_mutex_unlock(&b->mtx);
            free(m->data); free(m);
            return;
         }
      }
      if (b->umq_tail) { b->umq_tail->next = m; } else { b->umq_head = m; }
      b->umq_tail = m;
      pthread_cond_broadcast(&b->cv);
      pthread_mutex_unlock(&b->mtx);
   }
}

/* Post a receive. Returns a pending record (already satisfied if the message
   had arrived early). */
static nmpi_pending *post_recv(void *buf, int count, int dt, int src, int tag,
                               MPI_Comm comm)
{
   nmpi_inbox   *b = &g_inbox[g_myrank];
   nmpi_pending *p = (nmpi_pending *) malloc(sizeof(nmpi_pending));
   nmpi_msg     *m, *prev;

   p->next = NULL; p->comm = comm; p->src = src; p->tag = tag;
   p->buf = buf; p->count = count; p->dt = dt;
   p->done = 0; p->act_src = 0; p->act_tag = 0; p->nbytes = 0;

   pthread_mutex_lock(&b->mtx);
   prev = NULL;
   for (m = b->umq_head; m; prev = m, m = m->next)
   {
      if (match_ok(comm, src, tag, m->comm, m->src, m->tag))
      {
         if (prev) { prev->next = m->next; } else { b->umq_head = m->next; }
         if (b->umq_tail == m) { b->umq_tail = prev; }
         deliver(p, m->src, m->tag, m->data, m->nbytes);
         pthread_mutex_unlock(&b->mtx);
         free(m->data); free(m);
         return p;
      }
   }
   if (b->prq_tail) { b->prq_tail->next = p; } else { b->prq_head = p; }
   b->prq_tail = p;
   pthread_mutex_unlock(&b->mtx);
   return p;
}

static void wait_pending(nmpi_pending *p, MPI_Status *status)
{
   nmpi_inbox *b = &g_inbox[g_myrank];

   SET_STATE(NMPI_ST_RECV, p->comm, p->src, p->tag);
   pthread_mutex_lock(&b->mtx);
   while (!p->done) { pthread_cond_wait(&b->cv, &b->mtx); }
   pthread_mutex_unlock(&b->mtx);
   SET_STATE(NMPI_ST_RUN, 0, 0, 0);

   if (status)
   {
      status->MPI_SOURCE = p->act_src;
      status->MPI_TAG    = p->act_tag;
      status->nanompi_count  = (int) p->nbytes;
   }
   free(p);
}

static void do_recv(void *buf, int count, int dt, int src, int tag,
                    MPI_Comm comm, MPI_Status *status)
{
   wait_pending(post_recv(buf, count, dt, src, tag, comm), status);
}

/* Probe inspects only unmatched arrivals -- it must not consume a posted recv. */
static nmpi_msg *probe_umq(int comm, int src, int tag, int blocking, MPI_Status *status)
{
   nmpi_inbox *b = &g_inbox[g_myrank];
   nmpi_msg *m;

   pthread_mutex_lock(&b->mtx);
   for (;;)
   {
      for (m = b->umq_head; m; m = m->next)
      {
         if (match_ok(comm, src, tag, m->comm, m->src, m->tag))
         {
            if (status)
            {
               status->MPI_SOURCE = m->src;
               status->MPI_TAG    = m->tag;
               status->nanompi_count  = (int) m->nbytes;
            }
            pthread_mutex_unlock(&b->mtx);
            return m;
         }
      }
      if (!blocking) { pthread_mutex_unlock(&b->mtx); return NULL; }
      SET_STATE(NMPI_ST_PROBE, comm, src, tag);
      pthread_cond_wait(&b->cv, &b->mtx);
   }
}

/*--------------------------------------------------------------------------
 * Requests (per-thread pool, so no locking on the fast path)
 *--------------------------------------------------------------------------*/
/* kind: 1 = completed send, 2 = posted recv,
         3 = persistent send, 4 = persistent recv (MPI_*_init + MPI_Startall) */
typedef struct
{
   int           used;
   int           kind;
   int           active;         /* persistent request currently in flight */
   nmpi_pending *p;              /* kinds 2 and 4 */
   void         *buf;            /* persistent parameters, replayed on Start */
   int           count, dt, peer, tag, comm;
} nmpi_req;

static nmpi_req **g_reqs = NULL;   /* [world rank][NMPI_REQS_PER_RANK] */

static MPI_Request req_alloc(void)
{
   nmpi_req *pool = g_reqs[g_myrank];
   int i;
   for (i = 0; i < NMPI_REQS_PER_RANK; i++)
   {
      if (!pool[i].used)
      {
         pool[i].used = 1;
         return (MPI_Request)((size_t) g_myrank * NMPI_REQS_PER_RANK + i + 1);
      }
   }
   fprintf(stderr, "nano-mpi: out of request handles on rank %d\n", g_myrank);
   abort();
}

static nmpi_req *req_get(MPI_Request h)
{
   size_t idx;
   if (h == MPI_REQUEST_NULL) { return NULL; }
   idx = (size_t) h - 1;
   return &g_reqs[idx / NMPI_REQS_PER_RANK][idx % NMPI_REQS_PER_RANK];
}

/*--------------------------------------------------------------------------
 * Reductions
 *--------------------------------------------------------------------------*/
#define NMPI_LOOP(CTYPE, EXPR)                                              \
   {                                                                          \
      CTYPE *d = (CTYPE *) dst; const CTYPE *s = (const CTYPE *) src;          \
      int i;                                                                  \
      for (i = 0; i < count; i++) { EXPR; }                                    \
   }

/* MPI permits these on C integer and floating-point types. */
#define NMPI_REDUCE_ORD(CTYPE)                                               \
   switch (op) {                                                              \
      case MPI_MAX:     NMPI_LOOP(CTYPE, d[i] = (s[i] > d[i]) ? s[i] : d[i]); break; \
      case MPI_MIN:     NMPI_LOOP(CTYPE, d[i] = (s[i] < d[i]) ? s[i] : d[i]); break; \
      case MPI_SUM:     NMPI_LOOP(CTYPE, d[i] = (CTYPE)(d[i] + s[i])); break;  \
      case MPI_PROD:    NMPI_LOOP(CTYPE, d[i] = (CTYPE)(d[i] * s[i])); break;  \
      case MPI_REPLACE: NMPI_LOOP(CTYPE, d[i] = s[i]); break;                  \
      default:          bad_op(op, dt); break;                                 \
   }

/* C integer and logical types additionally take the logical and bitwise ops. */
#define NMPI_REDUCE_INT(CTYPE)                                               \
   switch (op) {                                                              \
      case MPI_LAND: NMPI_LOOP(CTYPE, d[i] = (CTYPE)(d[i] && s[i])); break;    \
      case MPI_LOR:  NMPI_LOOP(CTYPE, d[i] = (CTYPE)(d[i] || s[i])); break;    \
      case MPI_LXOR: NMPI_LOOP(CTYPE, d[i] = (CTYPE)(!d[i] != !s[i])); break;  \
      case MPI_BAND: NMPI_LOOP(CTYPE, d[i] = (CTYPE)(d[i] & s[i])); break;     \
      case MPI_BOR:  NMPI_LOOP(CTYPE, d[i] = (CTYPE)(d[i] | s[i])); break;     \
      case MPI_BXOR: NMPI_LOOP(CTYPE, d[i] = (CTYPE)(d[i] ^ s[i])); break;     \
      default:       NMPI_REDUCE_ORD(CTYPE); break;                            \
   }

/* Complex is stored as adjacent (re, im). Only SUM and PROD are defined for it;
   there is no ordering, so MAX and MIN are errors rather than silent nonsense.
   Kept in terms of the real type so the file needs no <complex.h> and stays
   compilable as C++. */
#define NMPI_REDUCE_CPLX(RTYPE)                                              \
   switch (op) {                                                              \
      case MPI_SUM:                                                           \
         { RTYPE *d = (RTYPE *) dst; const RTYPE *s = (const RTYPE *) src;     \
           int i; for (i = 0; i < 2 * count; i++) { d[i] += s[i]; } } break;   \
      case MPI_PROD:                                                          \
         { RTYPE *d = (RTYPE *) dst; const RTYPE *s = (const RTYPE *) src;     \
           int i;                                                             \
           for (i = 0; i < count; i++) {                                      \
              RTYPE a = d[2*i], b = d[2*i+1], c = s[2*i], e = s[2*i+1];        \
              d[2*i] = a*c - b*e; d[2*i+1] = a*e + b*c; } } break;             \
      case MPI_REPLACE:                                                       \
         memcpy(dst, src, (size_t) count * 2 * sizeof(RTYPE)); break;          \
      default: bad_op(op, dt); break;                                          \
   }

/* MAXLOC / MINLOC. On a tie the smaller index wins, as the standard requires. */
#define NMPI_REDUCE_LOC(STYPE)                                               \
   switch (op) {                                                              \
      case MPI_MAXLOC:                                                        \
         NMPI_LOOP(STYPE, d[i] = (s[i].value > d[i].value ||                   \
                                  (s[i].value == d[i].value &&                 \
                                   s[i].index <  d[i].index)) ? s[i] : d[i]);  \
         break;                                                               \
      case MPI_MINLOC:                                                        \
         NMPI_LOOP(STYPE, d[i] = (s[i].value < d[i].value ||                   \
                                  (s[i].value == d[i].value &&                 \
                                   s[i].index <  d[i].index)) ? s[i] : d[i]);  \
         break;                                                               \
      case MPI_REPLACE: NMPI_LOOP(STYPE, d[i] = s[i]); break;                  \
      default: bad_op(op, dt); break;                                          \
   }

static void bad_op(int op, int dt)
{
   fprintf(stderr, "nano-mpi: reduction op %d is not defined for datatype %d\n",
           op, dt);
   abort();
}

/* dst <- dst OP src */
static void reduce_into(void *dst, const void *src, int count, int dt, int op)
{
   switch (dt)
   {
      case MPI_CHAR:                  NMPI_REDUCE_INT(char);               break;
      case MPI_SIGNED_CHAR:           NMPI_REDUCE_INT(signed char);        break;
      case MPI_UNSIGNED_CHAR:         NMPI_REDUCE_INT(unsigned char);      break;
      case MPI_BYTE:                  NMPI_REDUCE_INT(unsigned char);      break;
      case MPI_SHORT:                 NMPI_REDUCE_INT(short);              break;
      case MPI_UNSIGNED_SHORT:        NMPI_REDUCE_INT(unsigned short);     break;
      case MPI_INT:                   NMPI_REDUCE_INT(int);                break;
      case MPI_UNSIGNED:              NMPI_REDUCE_INT(unsigned int);       break;
      case MPI_LONG:                  NMPI_REDUCE_INT(long);               break;
      case MPI_UNSIGNED_LONG:         NMPI_REDUCE_INT(unsigned long);      break;
      case MPI_LONG_LONG_INT:         NMPI_REDUCE_INT(long long);          break;
      case MPI_UNSIGNED_LONG_LONG:    NMPI_REDUCE_INT(unsigned long long); break;
      case MPI_C_BOOL:                NMPI_REDUCE_INT(bool);               break;
      case MPI_INT8_T:                NMPI_REDUCE_INT(int8_t);             break;
      case MPI_INT16_T:               NMPI_REDUCE_INT(int16_t);            break;
      case MPI_INT32_T:               NMPI_REDUCE_INT(int32_t);            break;
      case MPI_INT64_T:               NMPI_REDUCE_INT(int64_t);            break;
      case MPI_UINT8_T:               NMPI_REDUCE_INT(uint8_t);            break;
      case MPI_UINT16_T:              NMPI_REDUCE_INT(uint16_t);           break;
      case MPI_UINT32_T:              NMPI_REDUCE_INT(uint32_t);           break;
      case MPI_UINT64_T:              NMPI_REDUCE_INT(uint64_t);           break;
      case MPI_AINT:                  NMPI_REDUCE_INT(MPI_Aint);           break;
      case MPI_OFFSET:                NMPI_REDUCE_INT(MPI_Offset);         break;
      case MPI_COUNT:                 NMPI_REDUCE_INT(MPI_Count);          break;

      case MPI_FLOAT:                 NMPI_REDUCE_ORD(float);              break;
      case MPI_DOUBLE:                NMPI_REDUCE_ORD(double);             break;
      case MPI_LONG_DOUBLE:           NMPI_REDUCE_ORD(long double);        break;

      case MPI_C_FLOAT_COMPLEX:       NMPI_REDUCE_CPLX(float);             break;
      case MPI_C_DOUBLE_COMPLEX:      NMPI_REDUCE_CPLX(double);            break;
      case MPI_C_LONG_DOUBLE_COMPLEX: NMPI_REDUCE_CPLX(long double);       break;

      case MPI_FLOAT_INT:             NMPI_REDUCE_LOC(MPI_Float_int);      break;
      case MPI_DOUBLE_INT:            NMPI_REDUCE_LOC(MPI_Double_int);     break;
      case MPI_LONG_INT:              NMPI_REDUCE_LOC(MPI_Long_int);       break;
      case MPI_SHORT_INT:             NMPI_REDUCE_LOC(MPI_Short_int);      break;
      case MPI_2INT:                  NMPI_REDUCE_LOC(MPI_2int);           break;
      case MPI_LONG_DOUBLE_INT:       NMPI_REDUCE_LOC(MPI_Long_double_int); break;

      default:
         fprintf(stderr, "nano-mpi: reduction on datatype %d is not supported "
                         "(user-defined types cannot be reduced)\n", dt);
         abort();
   }
}


/* Combine sz contributions into recvbuf.
 *
 * The association matters: floating-point addition is not associative, so the
 * order in which contributions are combined decides the last bits of the
 * result. Summing linearly (((a0+a1)+a2)+a3) gives different answers from the
 * recursive doubling that MPI implementations use, which is enough to make an
 * AMG residual differ in its last digits and, over a long solve, drift.
 *
 * So use recursive doubling here too: fold the first 2r contributions pairwise
 * into r virtual ranks, shift the rest down, then combine with partner i^d for
 * d = 1, 2, 4, ... This is the standard algorithm; it is also more accurate
 * than a linear sum (error grows as log n rather than n), and it reproduces
 * Open MPI bit-for-bit -- verified over 400 random trials at every rank count
 * from 2 to 16, including non-powers of two.
 *
 * A user-defined op still sees MPI's (invec, inoutvec) contract, so the
 * left operand is passed as invec.
 */
static void reduce_all(void *recvbuf, const void **d, int sz, int count,
                       int dt, int op, size_t nb)
{
   nmpi_user_fn fn = op_user(op);
   const void **v;
   char *work, *tmp = NULL;
   int adjust = 1, r, k, step, i;

   if (sz <= 0) { return; }
   if (sz == 1) { memcpy(recvbuf, d[0], nb); return; }

   while (adjust * 2 <= sz) { adjust *= 2; }
   r = sz - adjust;

   work = (char *) malloc((size_t) adjust * nb);
   v    = (const void **) malloc(sizeof(void *) * (size_t) adjust);
   if (fn) { tmp = (char *) malloc(nb); }

   /* dst = left op right */
   #define NMPI_COMBINE(dst, left, right)                                     \
      do {                                                                    \
         if (fn)                                                              \
         {                                                                    \
            int len_ = (int) count;                               \
            MPI_Datatype dtc_ = (MPI_Datatype) dt;                \
            memcpy(tmp, (left), nb);                                          \
            memcpy((dst), (right), nb);                                       \
            fn((void *) tmp, (dst), &len_, &dtc_);                            \
         }                                                                    \
         else                                                                 \
         {                                                                    \
            if ((const void *)(dst) != (const void *)(left))                  \
            { memcpy((dst), (left), nb); }                                    \
            reduce_into((dst), (right), count, dt, op);                       \
         }                                                                    \
      } while (0)

   /* fold the 2r low contributions into r virtual ranks */
   for (k = 0; k < r; k++)
   {
      char *slot = work + (size_t) k * nb;
      NMPI_COMBINE(slot, d[2 * k], d[2 * k + 1]);
      v[k] = slot;
   }
   for (k = r; k < adjust; k++) { v[k] = d[k + r]; }

   /* recursive doubling over the virtual ranks */
   for (step = 1; step < adjust; step *= 2)
   {
      for (i = 0; i < adjust; i += 2 * step)
      {
         char *slot = work + (size_t) i * nb;
         NMPI_COMBINE(slot, v[i], v[i + step]);
         v[i] = slot;
      }
   }
   #undef NMPI_COMBINE

   memcpy(recvbuf, v[0], nb);
   free(work); free((void *) v); free(tmp);
}

/*--------------------------------------------------------------------------
 * Launch
 *--------------------------------------------------------------------------*/
typedef struct
{
   int    rank;
   int  (*fn)(int, char **, void *);
   int    argc;
   char **argv;
   void  *user;
   int    ret;
} nmpi_thread_arg;

static void *nmpi_watchdog(void *v)
{
   int secs = *(int *) v, i, t;
   unsigned long *prev = (unsigned long *) calloc((size_t) g_nranks, sizeof(unsigned long));

   for (t = 0; ; t++)
   {
      struct timespec ts; ts.tv_sec = 1; ts.tv_nsec = 0;
      nanosleep(&ts, NULL);
      {
         int stuck = 1;
         for (i = 0; i < g_nranks; i++)
         {
            if (g_state[i].seq != prev[i]) { stuck = 0; }
            prev[i] = g_state[i].seq;
         }
         if (!stuck) { t = 0; continue; }
      }
      if (t < secs) { continue; }
      fprintf(stderr, "\n=== TMPI WATCHDOG: no progress for %ds, %d ranks ===\n", secs, g_nranks);
      for (i = 0; i < g_nranks; i++)
      {
         fprintf(stderr, "  rank %3d: %-8s comm=%d a=%d b=%d\n",
                 i, state_name(g_state[i].op), g_state[i].a, g_state[i].b, g_state[i].c);
      }
      fflush(stderr);
      abort();
   }
   return NULL;
}

static void *nmpi_trampoline(void *v)
{
   nmpi_thread_arg *a = (nmpi_thread_arg *) v;
   g_myrank = a->rank;
   a->ret = a->fn(a->argc, a->argv, a->user);
   return NULL;
}


/*--------------------------------------------------------------------------
 * Default rank count: NANOMPI_NUM_RANKS if set, else the number of
 * cores online. Callers who pass a positive count to nanompi_run keep it;
 * this is only consulted when they ask for the default.
 *--------------------------------------------------------------------------*/
int nanompi_default_ranks(void)
{
   const char *e = getenv("NANOMPI_NUM_RANKS");

   if (e && *e)
   {
      char *end = NULL;
      long v = strtol(e, &end, 10);

      if (end != e && *end == '\0' && v >= 1)
      {
         return (int) v;
      }
      fprintf(stderr, "nano-mpi: ignoring invalid NANOMPI_NUM_RANKS=\"%s\" "
                      "(want a positive integer)\n", e);
   }
#if defined(_SC_NPROCESSORS_ONLN)
   {
      long n = sysconf(_SC_NPROCESSORS_ONLN);
      if (n >= 1) { return (int) n; }
   }
#endif
   return 1;
}

/*--------------------------------------------------------------------------
 * The rank universe: inboxes, request pools, COMM_WORLD. Process-wide, so at
 * most one universe exists at a time.
 *--------------------------------------------------------------------------*/

/* Build the implicit one-rank world. Only ever reached before a team exists,
   but locked anyway so two threads racing to first use cannot both build it. */
/* Hand the implicit world back at exit. Nothing else will: a caller that never
   started a team never calls anything that tears one down, so without this the
   one-rank world is a permanent allocation and a leak checker says so. */
static void free_auto_universe(void)
{
   pthread_mutex_lock(&g_auto_mtx);
   if (g_auto_universe) { universe_free(); g_auto_universe = 0; }
   pthread_mutex_unlock(&g_auto_mtx);
}

static void ensure_universe(void)
{
   static int registered = 0;   /* atexit slots are finite; take one, once */

   pthread_mutex_lock(&g_auto_mtx);
   if (!g_universe_up)
   {
      universe_init(1);
      g_auto_universe = 1;
      if (!registered) { atexit(free_auto_universe); registered = 1; }
   }
   pthread_mutex_unlock(&g_auto_mtx);
}

/* A real team supersedes the implicit world: drop it so nranks can grow. */
static void drop_auto_universe(void)
{
   if (g_auto_universe)
   {
      universe_free();
      g_auto_universe = 0;
   }
}

static int universe_init(int nranks)
{
   int *world;
   int i;

   if (g_universe_up)
   {
      fprintf(stderr, "nano-mpi: a rank universe already exists in this process\n");
      return 1;
   }
   g_nranks = nranks;

   g_inbox = (nmpi_inbox *) calloc((size_t) nranks, sizeof(nmpi_inbox));
   g_reqs  = (nmpi_req **) calloc((size_t) nranks, sizeof(nmpi_req *));
   for (i = 0; i < nranks; i++)
   {
      pthread_mutex_init(&g_inbox[i].mtx, NULL);
      pthread_cond_init(&g_inbox[i].cv, NULL);
      g_reqs[i] = (nmpi_req *) calloc(NMPI_REQS_PER_RANK, sizeof(nmpi_req));
   }

   world = (int *) malloc(sizeof(int) * (size_t) nranks);
   for (i = 0; i < nranks; i++) { world[i] = i; }
   {
      nmpi_comm *k = &g_comm[MPI_COMM_WORLD];
      /* a previous universe may have used this slot: free before overwriting */
      free(k->inv); free((void *) k->ptr); free(k->len);
      k->used = 1; k->size = nranks;
      k->world = world;
      k->inv = (int *) malloc(sizeof(int) * (size_t) nranks);
      k->ptr = (const void **) malloc(sizeof(void *) * (size_t) nranks);
      k->len = (size_t *) malloc(sizeof(size_t) * (size_t) nranks);
      for (i = 0; i < nranks; i++) { k->inv[i] = i; }
      if (!k->inited)
      {
         pthread_mutex_init(&k->mtx, NULL);
         pthread_cond_init(&k->cv, NULL);
         k->inited = 1;
      }
      k->count = 0;   /* generation stays monotonic; see comm_alloc */
   }
   g_comm[MPI_COMM_SELF].used = 1;
   g_comm[MPI_COMM_SELF].size = 1;

   g_state = (nmpi_state *) calloc((size_t) nranks, sizeof(nmpi_state));
   {
      const char *wd = getenv("NANOMPI_WATCHDOG");
      if (wd && atoi(wd) > 0)
      {
         static int secs; pthread_t wt;
         secs = atoi(wd);
         pthread_create(&wt, NULL, nmpi_watchdog, &secs);
         pthread_detach(wt);
      }
   }
   g_universe_up = 1;
   return 0;
}

static void universe_free(void)
{
   int i;
   if (!g_universe_up) { return; }
   for (i = 0; i < g_nranks; i++) { free(g_reqs[i]); }
   free(g_reqs);   g_reqs = NULL;
   free(g_inbox);  g_inbox = NULL;
   free(g_state);  g_state = NULL;
   {
      nmpi_comm *k = &g_comm[MPI_COMM_WORLD];
      free(k->world); k->world = NULL;
      free(k->inv);   k->inv   = NULL;
      free((void *) k->ptr); k->ptr = NULL;
      free(k->len);   k->len   = NULL;
      k->used = 0; k->size = 0;
   }
   g_universe_up = 0;
}

int nanompi_run(int nranks, int (*fn)(int, char **, void *),
                   int argc, char **argv, void *user)
{
   pthread_t       *th;
   nmpi_thread_arg *args;
   int i, rc = 0;

   /* nranks <= 0 means "pick for me": NANOMPI_NUM_RANKS, else cores online */
   if (nranks < 1) { nranks = nanompi_default_ranks(); }
   if (nranks < 1) { return 1; }
   drop_auto_universe();
   if (universe_init(nranks)) { return 1; }

   th   = (pthread_t *) malloc(sizeof(pthread_t) * (size_t) nranks);
   args = (nmpi_thread_arg *) malloc(sizeof(nmpi_thread_arg) * (size_t) nranks);
   for (i = 0; i < nranks; i++)
   {
      args[i].rank = i; args[i].fn = fn; args[i].argc = argc;
      args[i].argv = argv; args[i].user = user; args[i].ret = 0;
      if (pthread_create(&th[i], NULL, nmpi_trampoline, &args[i]) != 0)
      {
         fprintf(stderr, "nano-mpi: pthread_create failed for rank %d\n", i);
         return 1;
      }
   }
   for (i = 0; i < nranks; i++) { pthread_join(th[i], NULL); if (args[i].ret) { rc = args[i].ret; } }

   free(th); free(args);
   universe_free();
   return rc;
}

/*--------------------------------------------------------------------------
 * Persistent rank team
 *
 * nanompi_run() spawns, runs one function and joins, which suits a program
 * whose whole life is the parallel region. A library embedding nano-mpi needs the
 * opposite shape: the ranks must outlive individual calls, so that a solver can
 * be constructed, then have factorize() and solve() invoked on it repeatedly,
 * each collectively across the same ranks.
 *
 * The team spawns its threads once and parks them. Each invoke wakes them to
 * run one function and blocks the caller until all have returned. The calling
 * thread is not itself a rank.
 *--------------------------------------------------------------------------*/
struct nanompi_team_struct
{
   pthread_t      *th;
   int             nranks;
   pthread_mutex_t mtx;
   pthread_cond_t  cv_work;    /* ranks wait here for the next call */
   pthread_cond_t  cv_done;    /* the caller waits here for completion */
   int             generation; /* bumped once per invoke */
   int             nfinished;
   int             shutdown;
   int           (*fn)(void *);
   void           *user;
   int             rc;
};

typedef struct { nanompi_team *team; int rank; } nmpi_team_arg;

static void *nmpi_team_worker(void *v)
{
   nmpi_team_arg   *a = (nmpi_team_arg *) v;
   nanompi_team *t = a->team;
   int seen = 0;

   g_myrank = a->rank;

   for (;;)
   {
      int (*fn)(void *); void *user; int rc;

      pthread_mutex_lock(&t->mtx);
      while (!t->shutdown && t->generation == seen)
      {
         pthread_cond_wait(&t->cv_work, &t->mtx);
      }
      if (t->shutdown) { pthread_mutex_unlock(&t->mtx); break; }
      seen = t->generation;
      fn = t->fn; user = t->user;
      pthread_mutex_unlock(&t->mtx);

      rc = fn ? fn(user) : 0;

      pthread_mutex_lock(&t->mtx);
      if (rc && !t->rc) { t->rc = rc; }
      if (++t->nfinished == t->nranks) { pthread_cond_broadcast(&t->cv_done); }
      pthread_mutex_unlock(&t->mtx);
   }
   return NULL;
}

int nanompi_team_create(int nranks, nanompi_team **team_ptr)
{
   nanompi_team *t;
   nmpi_team_arg   *args;
   int i;

   if (!team_ptr) { return 1; }
   *team_ptr = NULL;

   if (nranks < 1) { nranks = nanompi_default_ranks(); }
   if (nranks < 1) { return 1; }
   drop_auto_universe();
   if (universe_init(nranks)) { return 1; }

   t = (nanompi_team *) calloc(1, sizeof(nanompi_team));
   t->nranks = nranks;
   pthread_mutex_init(&t->mtx, NULL);
   pthread_cond_init(&t->cv_work, NULL);
   pthread_cond_init(&t->cv_done, NULL);
   t->th = (pthread_t *) malloc(sizeof(pthread_t) * (size_t) nranks);

   /* the arg blocks must outlive create(), so hang them off the team */
   args = (nmpi_team_arg *) malloc(sizeof(nmpi_team_arg) * (size_t) nranks);
   for (i = 0; i < nranks; i++)
   {
      args[i].team = t; args[i].rank = i;
      if (pthread_create(&t->th[i], NULL, nmpi_team_worker, &args[i]) != 0)
      {
         fprintf(stderr, "nano-mpi: pthread_create failed for rank %d\n", i);
         return 1;
      }
   }
   t->user = NULL;
   *team_ptr = t;
   return 0;
}

int nanompi_team_size(nanompi_team *t) { return t ? t->nranks : 0; }

int nanompi_team_invoke(nanompi_team *t, int (*fn)(void *user), void *user)
{
   int rc;

   if (!t || !fn) { return 1; }

   pthread_mutex_lock(&t->mtx);
   t->fn = fn; t->user = user; t->rc = 0; t->nfinished = 0;
   t->generation++;
   pthread_cond_broadcast(&t->cv_work);
   while (t->nfinished < t->nranks) { pthread_cond_wait(&t->cv_done, &t->mtx); }
   rc = t->rc;
   pthread_mutex_unlock(&t->mtx);

   return rc;
}

/*--------------------------------------------------------------------------
 * Caller-as-rank-0 team.
 *
 * The invoke model above blocks the caller for the duration of each collective
 * call, which suits "run this function on N ranks". A library written in the
 * SPMD style wants the opposite: one driver rank that stays in the
 * application's hands, and workers that live inside a service loop for as long
 * as the object exists, taking commands from the driver.
 *
 * nanompi_team_start() makes the calling thread rank 0 and returns to it
 * immediately, having spawned ranks 1..nranks-1 each running 'worker'. The
 * application then drives as rank 0 -- broadcasting commands the workers
 * service -- and calls nanompi_team_join() once it has told them to stop.
 *--------------------------------------------------------------------------*/
typedef struct { nanompi_team *team; int rank; int (*fn)(void *); void *user; int ret; }
        nmpi_worker_arg;

static void *nmpi_worker_entry(void *v)
{
   nmpi_worker_arg *a = (nmpi_worker_arg *) v;
   g_myrank = a->rank;
   a->ret = a->fn ? a->fn(a->user) : 0;
   return NULL;
}

int nanompi_team_start(int nranks, int (*worker)(void *user), void *user,
                          nanompi_team **team_ptr)
{
   nanompi_team *t;
   nmpi_worker_arg *args;
   int i;

   if (!team_ptr || !worker) { return 1; }
   *team_ptr = NULL;

   if (nranks < 1) { nranks = nanompi_default_ranks(); }
   if (nranks < 1) { return 1; }
   drop_auto_universe();
   if (universe_init(nranks)) { return 1; }

   t = (nanompi_team *) calloc(1, sizeof(nanompi_team));
   t->nranks = nranks;
   pthread_mutex_init(&t->mtx, NULL);
   pthread_cond_init(&t->cv_work, NULL);
   pthread_cond_init(&t->cv_done, NULL);
   t->th = (pthread_t *) calloc((size_t) nranks, sizeof(pthread_t));

   args = (nmpi_worker_arg *) calloc((size_t) nranks, sizeof(nmpi_worker_arg));
   t->user = args;               /* keep them alive until join */

   /* the caller is rank 0 and keeps running the application */
   g_myrank = 0;

   for (i = 1; i < nranks; i++)
   {
      args[i].team = t; args[i].rank = i; args[i].fn = worker; args[i].user = user;
      if (pthread_create(&t->th[i], NULL, nmpi_worker_entry, &args[i]) != 0)
      {
         fprintf(stderr, "nano-mpi: pthread_create failed for rank %d\n", i);
         return 1;
      }
   }
   *team_ptr = t;
   return 0;
}

int nanompi_team_join(nanompi_team *t)
{
   nmpi_worker_arg *args;
   int i, rc = 0;

   if (!t) { return 0; }
   args = (nmpi_worker_arg *) t->user;

   for (i = 1; i < t->nranks; i++)
   {
      pthread_join(t->th[i], NULL);
      if (args && args[i].ret && !rc) { rc = args[i].ret; }
   }

   pthread_mutex_destroy(&t->mtx);
   pthread_cond_destroy(&t->cv_work);
   pthread_cond_destroy(&t->cv_done);
   free(args);
   free(t->th);
   free(t);
   universe_free();
   return rc;
}

int nanompi_team_destroy(nanompi_team *t)
{
   int i;
   if (!t) { return 0; }

   pthread_mutex_lock(&t->mtx);
   t->shutdown = 1;
   pthread_cond_broadcast(&t->cv_work);
   pthread_mutex_unlock(&t->mtx);

   for (i = 0; i < t->nranks; i++) { pthread_join(t->th[i], NULL); }

   pthread_mutex_destroy(&t->mtx);
   pthread_cond_destroy(&t->cv_work);
   pthread_cond_destroy(&t->cv_done);
   free(t->th);
   free(t);
   universe_free();
   return 0;
}

/*==========================================================================
 * MPI_* entry points
 *========================================================================*/

/* MPI_Init does not create the ranks -- nanompi_run/team_start already did, or
   this thread is a one-rank world. What it does is record that the program
   passed through initialisation, so MPI_Initialized answers truthfully. The
   flags are per-rank because each rank initialises itself, exactly as each
   process does under a launcher. */
static NANOMPI_TLS int g_initialized = 0;
static NANOMPI_TLS int g_finalized   = 0;

int MPI_Init( int *argc, char ***argv )
{
   NANOMPI_UNUSED(argc); NANOMPI_UNUSED(argv);
   ensure_universe();
   g_initialized = 1;
   return MPI_SUCCESS;
}

int MPI_Init_thread( int *argc, char ***argv, int required, int *provided )
{
   NANOMPI_UNUSED(required);
   /* A rank IS a thread, so rank identity is thread-local: a helper thread the
      rank spawns has no rank and must not call MPI. That is exactly
      MPI_THREAD_FUNNELED. Asking for more is not an error -- the standard
      allows 'provided' to come back lower than 'required' -- but a caller that
      needs MPI_THREAD_MULTIPLE has to check, so return the truth. */
   if (provided) { *provided = MPI_THREAD_FUNNELED; }
   return MPI_Init(argc, argv);
}

int MPI_Initialized( int *flag ) { *flag = g_initialized; return MPI_SUCCESS; }
int MPI_Finalized( int *flag )   { *flag = g_finalized;   return MPI_SUCCESS; }

int MPI_Query_thread( int *provided )
{
   *provided = MPI_THREAD_FUNNELED;
   return MPI_SUCCESS;
}

/* Every rank thread is the main thread of its own rank. */
int MPI_Is_thread_main( int *flag ) { *flag = 1; return MPI_SUCCESS; }

int MPI_Finalize( void ) { g_finalized = 1; return MPI_SUCCESS; }

int MPI_Get_version( int *version, int *subversion )
{
   if (version)    { *version = MPI_VERSION; }
   if (subversion) { *subversion = MPI_SUBVERSION; }
   return MPI_SUCCESS;
}

int MPI_Get_library_version( char *version, int *resultlen )
{
   int n = snprintf(version, MPI_MAX_LIBRARY_VERSION_STRING,
                    "nano-mpi %d.%d.%d (MPI ranks as threads; see SCOPE.md)",
                    NANOMPI_VERSION_MAJOR, NANOMPI_VERSION_MINOR,
                    NANOMPI_VERSION_PATCH);
   if (resultlen) { *resultlen = (n < 0) ? 0 : n; }
   return MPI_SUCCESS;
}

/* Every rank is in the same process on the same host, so the interesting part
   of the name is the rank, not the machine. */
int MPI_Get_processor_name( char *name, int *resultlen )
{
   char host[128];
   int n;
   if (gethostname(host, sizeof host) != 0) { strcpy(host, "localhost"); }
   host[sizeof host - 1] = 0;
   n = snprintf(name, MPI_MAX_PROCESSOR_NAME, "%s:rank%d", host, g_myrank);
   if (resultlen) { *resultlen = (n < 0) ? 0 : n; }
   return MPI_SUCCESS;
}

int MPI_Error_class( int errorcode, int *errorclass )
{
   /* error codes and classes are the same thing here: there are no
      implementation-specific codes yet */
   *errorclass = errorcode;
   return MPI_SUCCESS;
}

int MPI_Error_string( int errorcode, char *string, int *resultlen )
{
   static const char *const msg[] = {
      "no error", "invalid buffer", "invalid count", "invalid datatype",
      "invalid tag", "invalid communicator", "invalid rank", "invalid request",
      "invalid root", "invalid group", "invalid reduction operation",
      "invalid topology", "invalid dimension argument", "invalid argument",
      "unknown error", "message truncated", "other error", "internal error",
      "error in status", "pending",
      ("process spawning is not supported by nano-mpi: "
       "ranks are threads, there are no processes to spawn"),
      "out of memory", "operation not supported by nano-mpi; see SCOPE.md"
   };
   const char *m = (errorcode >= 0 && errorcode < (int)(sizeof msg / sizeof msg[0]))
                   ? msg[errorcode] : "unknown error code";
   int n = snprintf(string, MPI_MAX_ERROR_STRING, "%s", m);
   if (resultlen) { *resultlen = (n < 0) ? 0 : n; }
   return MPI_SUCCESS;
}

int MPI_Abort( MPI_Comm comm, int errorcode )
{
   NANOMPI_UNUSED(comm);
   fprintf(stderr, "nano-mpi: abort(%d)\n", (int) errorcode);
   abort();
   return 0;
}

double MPI_Wtime( void )
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

double MPI_Wtick( void ) { return 1.0e-9; }

int MPI_Barrier( MPI_Comm comm ) { comm_barrier(comm); return 0; }

int MPI_Comm_size( MPI_Comm comm, int *size )
{
   *size = (int) comm_size_of(comm);
   return 0;
}

int MPI_Comm_rank( MPI_Comm comm, int *rank )
{
   *rank = (int) comm_rank_of(comm);
   return 0;
}

int MPI_Comm_compare( MPI_Comm comm1, MPI_Comm comm2, int *result )
{
   nmpi_comm *a = comm_get(comm1), *b = comm_get(comm2);
   int i;

   if (comm1 == comm2) { *result = MPI_IDENT; return MPI_SUCCESS; }
   if (!a || !b)       { *result = MPI_UNEQUAL; return MPI_SUCCESS; }
   if (a->size != b->size) { *result = MPI_UNEQUAL; return MPI_SUCCESS; }
   /* same members in the same order is CONGRUENT (IDENT is reserved for the
      same handle); same members in a different order is SIMILAR */
   for (i = 0; i < a->size; i++)
   {
      if (a->world[i] != b->world[i]) { break; }
   }
   if (i == a->size) { *result = MPI_CONGRUENT; return MPI_SUCCESS; }
   for (i = 0; i < a->size; i++)
   {
      if (b->inv[a->world[i]] < 0) { *result = MPI_UNEQUAL; return MPI_SUCCESS; }
   }
   *result = MPI_SIMILAR;
   return MPI_SUCCESS;
}

int MPI_Comm_dup( MPI_Comm comm, MPI_Comm *newcomm )
{
   nmpi_comm *k = comm_get(comm);
   int h;

   if (comm == MPI_COMM_SELF || !k) { *newcomm = comm; return 0; }
   /* every member must agree on the new handle: rank 0 allocates, then broadcast */
   comm_barrier(comm);
   if (comm_rank_of(comm) == 0)
   {
      h = comm_alloc(k->size, k->world);
      k->ptr[0] = NULL;
      k->len[0] = (size_t) h;
   }
   comm_barrier(comm);
   h = (int) k->len[0];
   comm_barrier(comm);
   *newcomm = (MPI_Comm) h;
   return 0;
}

int MPI_Comm_free( MPI_Comm *comm )
{
   nmpi_comm *k;
   int me;

   if (!comm) { return MPI_ERR_ARG; }
   k = comm_get(*comm);

   /* WORLD and SELF outlive every caller, and freeing COMM_NULL is a no-op. */
   if (k && *comm > MPI_COMM_SELF)
   {
      /* Read our rank BEFORE the barrier. comm_rank_of() indexes k->inv, which
         rank 0 is about to free -- asking afterwards is a use-after-free that
         only shows up when rank 0 wins the race out of the barrier. */
      me = comm_rank_of(*comm);
      comm_barrier(*comm);
      if (me == 0)
      {
         pthread_mutex_lock(&g_comm_mtx);
         free(k->world); free(k->inv); free((void *) k->ptr); free(k->len);
         k->world = NULL; k->inv = NULL; k->ptr = NULL; k->len = NULL;
         k->used = 0;
         pthread_mutex_unlock(&g_comm_mtx);
      }
   }
   *comm = MPI_COMM_NULL;
   return MPI_SUCCESS;
}

/* group == the set of world ranks; we encode a group as a comm handle */
int MPI_Comm_group( MPI_Comm comm, MPI_Group *group )
{
   nmpi_comm *k = comm_get(comm);
   if (comm == MPI_COMM_SELF || !k)
   {
      int self = g_myrank;
      *group = (MPI_Group) group_alloc(1, &self);
   }
   else
   {
      *group = (MPI_Group) group_alloc(k->size, k->world);
   }
   return 0;
}

int MPI_Group_incl( MPI_Group group, int n, const int *ranks,
                                MPI_Group *newgroup )
{
   nmpi_group *g = group_get(group);
   int *w, i;

   if (!g) { *newgroup = group; return 0; }
   w = (int *) malloc(sizeof(int) * (size_t)(n > 0 ? n : 1));
   for (i = 0; i < n; i++)
   {
      int r = (int) ranks[i];
      w[i] = (r >= 0 && r < g->size) ? g->world[r] : 0;
   }
   *newgroup = (MPI_Group) group_alloc((int) n, w);
   free(w);
   return 0;
}

int MPI_Group_size( MPI_Group group, int *size )
{
   nmpi_group *g = group_get(group);
   if (!g) { *size = 0; return MPI_ERR_GROUP; }
   *size = g->size;
   return MPI_SUCCESS;
}

int MPI_Group_rank( MPI_Group group, int *rank )
{
   nmpi_group *g = group_get(group);
   int i;
   if (!g) { *rank = MPI_UNDEFINED; return MPI_ERR_GROUP; }
   *rank = MPI_UNDEFINED;
   for (i = 0; i < g->size; i++)
   {
      if (g->world[i] == g_myrank) { *rank = i; break; }
   }
   return MPI_SUCCESS;
}

int MPI_Group_free( MPI_Group *group )
{
   nmpi_group *g = group_get(*group);
   if (g)
   {
      pthread_mutex_lock(&g_group_mtx);
      free(g->world); g->world = NULL; g->used = 0;
      pthread_mutex_unlock(&g_group_mtx);
   }
   return 0;
}

/* Collective over 'comm'. Ranks in 'group' get the new communicator; everyone
   else gets COMM_NULL. This is how a coarse-grid sub-communicator gets built,
   so getting the non-member case right is what keeps the other ranks moving. */
int MPI_Comm_create( MPI_Comm comm, MPI_Group group,
                                 MPI_Comm *newcomm )
{
   nmpi_comm  *k = comm_get(comm);
   nmpi_group *g = group_get(group);
   int h, i, member = 0;

   if (comm == MPI_COMM_SELF || !k || !g) { return MPI_Comm_dup(comm, newcomm); }

   comm_barrier(comm);
   if (comm_rank_of(comm) == 0) { k->len[0] = (size_t) comm_alloc(g->size, g->world); }
   comm_barrier(comm);
   h = (int) k->len[0];
   for (i = 0; i < g->size; i++) { if (g->world[i] == g_myrank) { member = 1; break; } }
   comm_barrier(comm);

   *newcomm = member ? (MPI_Comm) h : (MPI_Comm) MPI_COMM_NULL;
   return 0;
}

/* colour = n, key = m */
int MPI_Comm_split( MPI_Comm comm, int n, int m,
                                MPI_Comm *newcomm )
{
   nmpi_comm *k = comm_get(comm);
   int sz, me, i, j, cnt, h = MPI_COMM_NULL;
   int *colours, *keys, *members;

   if (comm == MPI_COMM_SELF || !k) { *newcomm = comm; return 0; }
   sz = k->size;
   me = comm_rank_of(comm);

   colours = (int *) malloc(sizeof(int) * (size_t) sz * 2);
   keys    = colours + sz;

   /* gather colours and keys */
   {
      int cm = n, km = m;
      MPI_Allgather(&cm, 1, MPI_INT, colours, 1, MPI_INT, comm);
      MPI_Allgather(&km, 1, MPI_INT, keys,    1, MPI_INT, comm);
   }

   if (n != MPI_UNDEFINED)
   {
      members = (int *) malloc(sizeof(int) * (size_t) sz);
      cnt = 0;
      for (i = 0; i < sz; i++) { if (colours[i] == n) { members[cnt++] = i; } }
      /* stable sort members by key */
      for (i = 1; i < cnt; i++)
      {
         int v = members[i];
         for (j = i - 1; j >= 0 && keys[members[j]] > keys[v]; j--) { members[j + 1] = members[j]; }
         members[j + 1] = v;
      }
      for (i = 0; i < cnt; i++) { members[i] = k->world[members[i]]; }

      /* the lowest-ranked member of each colour allocates the handle and shares it */
      {
         int *all_h = (int *) malloc(sizeof(int) * (size_t) sz);
         int mine = -1;
         if (k->world[me] == members[0]) { mine = comm_alloc(cnt, members); }
         MPI_Allgather(&mine, 1, MPI_INT, all_h, 1, MPI_INT, comm);
         for (i = 0; i < sz; i++)
         {
            if (colours[i] == n && all_h[i] >= 0) { h = all_h[i]; break; }
         }
         free(all_h);
      }
      free(members);
   }
   else
   {
      int *all_h = (int *) malloc(sizeof(int) * (size_t) sz);
      int mine = -1;
      MPI_Allgather(&mine, 1, MPI_INT, all_h, 1, MPI_INT, comm);
      free(all_h);
   }

   free(colours);
   *newcomm = (MPI_Comm) h;
   return 0;
}

/* every rank is on the same "node" here, so shared-memory split == dup */
int MPI_Comm_split_type( MPI_Comm comm, int split_type, int key,
                                     MPI_Info info, MPI_Comm *newcomm )
{
   NANOMPI_UNUSED(split_type); NANOMPI_UNUSED(key); NANOMPI_UNUSED(info);
   return MPI_Comm_dup(comm, newcomm);
}

int MPI_Address( const void *location, MPI_Aint *address )
{
   *address = (MPI_Aint)(intptr_t) location;
   return 0;
}

int MPI_Get_count( const MPI_Status *status, MPI_Datatype datatype,
                               int *count )
{
   size_t esz = dt_size(datatype);
   *count = (esz > 0) ? (int)((size_t) status->nanompi_count / esz) : 0;
   return 0;
}

/*--------------------------------------------------------------------------
 * Collectives (barrier + shared scratch pointers)
 *--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------
 * MPI_IN_PLACE
 *
 * A rank that passes MPI_IN_PLACE contributes what is already sitting in its
 * recvbuf. That is awkward here, because a collective publishes each rank's
 * send pointer for every other rank to read, and the same recvbuf is about to
 * be overwritten with the result -- so publishing recvbuf directly is a race
 * against the readers. Where the result overwrites the contribution
 * (reductions) we publish a private copy; where it does not (the gathers, whose
 * own slot is written with the bytes already there) we publish the slot and
 * skip copying it onto itself.
 *
 * Scatter, Scatterv and Alltoall do not accept MPI_IN_PLACE here. They say so
 * rather than corrupting memory quietly.
 *--------------------------------------------------------------------------*/
static void no_inplace(const char *fn)
{
   fprintf(stderr, "nano-mpi: %s does not support MPI_IN_PLACE; see SCOPE.md\n", fn);
   abort();
}

int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype,
                           int root, MPI_Comm comm )
{
   nmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm);

   if (comm == MPI_COMM_SELF || !k || k->size == 1) { return 0; }
   if (me == root) { k->ptr[0] = buffer; }
   comm_barrier(comm);
   if (me != root) { memcpy(buffer, k->ptr[0], msg_bytes(count, datatype)); }
   comm_barrier(comm);
   return 0;
}

int MPI_Allreduce( const void *sendbuf, void *recvbuf, int count,
                               MPI_Datatype datatype, MPI_Op op, MPI_Comm comm )
{
   nmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t nb = msg_bytes(count, datatype);
   void *inplace = NULL;

   if (sendbuf == MPI_IN_PLACE)
   {
      inplace = malloc(nb ? nb : 1);
      if (!inplace) { fprintf(stderr, "nano-mpi: out of memory in %s\n", "MPI_Allreduce"); abort(); }
      memcpy(inplace, recvbuf, nb);
      sendbuf = inplace;
   }

   if (comm == MPI_COMM_SELF || !k || k->size == 1)
   {
      if (sendbuf != recvbuf) { memcpy(recvbuf, sendbuf, nb); }
      free(inplace);
      return MPI_SUCCESS;
   }
   sz = k->size;
   /* publish own contribution; every rank then reduces the full set itself so
      the result is bitwise identical across ranks (rank order is fixed) */
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   reduce_all(recvbuf, k->ptr, sz, count, datatype, op, nb);
   comm_barrier(comm);   /* nobody is reading our pointer after this */
   free(inplace);
   NANOMPI_UNUSED(i);
   return MPI_SUCCESS;
}

int MPI_Reduce( const void *sendbuf, void *recvbuf, int count,
                            MPI_Datatype datatype, MPI_Op op, int root,
                            MPI_Comm comm )
{
   nmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t nb = msg_bytes(count, datatype);
   void *inplace = NULL;

   if (sendbuf == MPI_IN_PLACE)
   {
      inplace = malloc(nb ? nb : 1);
      if (!inplace) { fprintf(stderr, "nano-mpi: out of memory in %s\n", "MPI_Reduce"); abort(); }
      memcpy(inplace, recvbuf, nb);
      sendbuf = inplace;
   }

   if (comm == MPI_COMM_SELF || !k || k->size == 1)
   {
      if (sendbuf != recvbuf) { memcpy(recvbuf, sendbuf, nb); }
      free(inplace);
      return MPI_SUCCESS;
   }
   sz = k->size;
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   if (me == root) { reduce_all(recvbuf, k->ptr, sz, count, datatype, op, nb); }
   comm_barrier(comm);
   free(inplace);
   NANOMPI_UNUSED(i);
   return MPI_SUCCESS;
}

int MPI_Scan( const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm )
{
   nmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i;
   size_t nb = msg_bytes(count, datatype);
   void *inplace = NULL;

   if (sendbuf == MPI_IN_PLACE)
   {
      inplace = malloc(nb ? nb : 1);
      if (!inplace) { fprintf(stderr, "nano-mpi: out of memory in %s\n", "MPI_Scan"); abort(); }
      memcpy(inplace, recvbuf, nb);
      sendbuf = inplace;
   }

   if (comm == MPI_COMM_SELF || !k || k->size == 1)
   {
      if (sendbuf != recvbuf) { memcpy(recvbuf, sendbuf, nb); }
      free(inplace);
      return MPI_SUCCESS;
   }
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   reduce_all(recvbuf, k->ptr, me + 1, count, datatype, op, nb);
   comm_barrier(comm);
   free(inplace);
   NANOMPI_UNUSED(i);
   return MPI_SUCCESS;
}

int MPI_Allgather( const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   MPI_Comm comm )
{
   nmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz, inplace = (sendbuf == MPI_IN_PLACE);
   size_t rb = msg_bytes(recvcount, recvtype);
   size_t sb = inplace ? rb : msg_bytes(sendcount, sendtype);

   if (inplace) { sendbuf = (char *) recvbuf + (size_t) me * rb; }

   if (comm == MPI_COMM_SELF || !k || k->size == 1)
   {
      if (!inplace) { memcpy(recvbuf, sendbuf, sb); }
      return MPI_SUCCESS;
   }
   sz = k->size;
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   for (i = 0; i < sz; i++)
   {
      if (i == me && inplace) { continue; }   /* already there; copying it onto
                                                 itself would race the readers */
      memcpy((char *) recvbuf + (size_t) i * rb, k->ptr[i], sb);
   }
   comm_barrier(comm);
   return MPI_SUCCESS;
}

int MPI_Allgatherv( const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                void *recvbuf, const int *recvcounts, const int *displs,
                                MPI_Datatype recvtype, MPI_Comm comm )
{
   nmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t esz = dt_size(recvtype);

   int inplace = (sendbuf == MPI_IN_PLACE);

   if (inplace) { sendbuf = (char *) recvbuf + (size_t) displs[me] * esz; }

   if (comm == MPI_COMM_SELF || !k || k->size == 1)
   {
      if (!inplace)
      {
         memcpy((char *) recvbuf + (size_t) displs[0] * esz, sendbuf,
                msg_bytes(sendcount, sendtype));
      }
      return MPI_SUCCESS;
   }
   sz = k->size;
   k->ptr[me] = sendbuf;
   k->len[me] = inplace ? (size_t) recvcounts[me] * esz
                        : msg_bytes(sendcount, sendtype);
   comm_barrier(comm);
   for (i = 0; i < sz; i++)
   {
      if (i == me && inplace) { continue; }
      memcpy((char *) recvbuf + (size_t) displs[i] * esz, k->ptr[i],
             (size_t) recvcounts[i] * esz);
   }
   comm_barrier(comm);
   return MPI_SUCCESS;
}

int MPI_Gather( const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                            void *recvbuf, int recvcount, MPI_Datatype recvtype,
                            int root, MPI_Comm comm )
{
   nmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t sb = msg_bytes(sendcount, sendtype), rb = 0;

   int inplace = (sendbuf == MPI_IN_PLACE);   /* root only, per the standard */

   if (inplace)
   {
      rb = msg_bytes(recvcount, recvtype);
      sb = rb;
      sendbuf = (char *) recvbuf + (size_t) me * rb;
   }
   if (comm == MPI_COMM_SELF || !k || k->size == 1)
   {
      if (!inplace) { memcpy(recvbuf, sendbuf, sb); }
      return MPI_SUCCESS;
   }
   sz = k->size;
   k->ptr[me] = sendbuf;
   k->len[me] = sb;
   comm_barrier(comm);
   if (me == root)
   {
      rb = msg_bytes(recvcount, recvtype);   /* root-only arguments */
      for (i = 0; i < sz; i++)
      {
         if (i == me && inplace) { continue; }
         memcpy((char *) recvbuf + (size_t) i * rb, k->ptr[i], k->len[i]);
      }
   }
   comm_barrier(comm);
   return MPI_SUCCESS;
}

int MPI_Gatherv( const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, const int *recvcounts, const int *displs,
                             MPI_Datatype recvtype, int root, MPI_Comm comm )
{
   nmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t esz = dt_size(recvtype);

   int inplace = (sendbuf == MPI_IN_PLACE);   /* root only, per the standard */

   if (inplace) { sendbuf = (char *) recvbuf + (size_t) displs[me] * esz; }

   if (comm == MPI_COMM_SELF || !k || k->size == 1)
   {
      if (!inplace)
      {
         memcpy((char *) recvbuf + (size_t) displs[0] * esz, sendbuf,
                msg_bytes(sendcount, sendtype));
      }
      return MPI_SUCCESS;
   }
   sz = k->size;
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   if (me == root)
   {
      for (i = 0; i < sz; i++)
      {
         if (i == me && inplace) { continue; }
         memcpy((char *) recvbuf + (size_t) displs[i] * esz, k->ptr[i],
                (size_t) recvcounts[i] * esz);
      }
   }
   comm_barrier(comm);
   return MPI_SUCCESS;
}

int MPI_Scatter( const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             int root, MPI_Comm comm )
{
   if (sendbuf == MPI_IN_PLACE) { no_inplace("MPI_Scatter"); }
   nmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm);
   size_t sb = msg_bytes(sendcount, sendtype);

   if (comm == MPI_COMM_SELF || !k || k->size == 1)
   {
      memcpy(recvbuf, sendbuf, msg_bytes(recvcount, recvtype));
      return 0;
   }
   if (me == root) { k->rbuf = sendbuf; k->rn = sendcount; }
   comm_barrier(comm);
   /* sendcount is only valid at the root, so use the root's value */
   sb = msg_bytes((int) k->rn, sendtype);
   memcpy(recvbuf, (const char *) k->rbuf + (size_t) me * sb, sb);
   comm_barrier(comm);
   return 0;
}

int MPI_Scatterv( const void *sendbuf, const int *sendcounts, const int *displs,
                              MPI_Datatype sendtype, void *recvbuf, int recvcount,
                              MPI_Datatype recvtype, int root, MPI_Comm comm )
{
   if (sendbuf == MPI_IN_PLACE) { no_inplace("MPI_Scatterv"); }
   nmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm);
   size_t esz = dt_size(sendtype);

   if (comm == MPI_COMM_SELF || !k || k->size == 1)
   {
      memcpy(recvbuf, (const char *) sendbuf + (size_t) displs[0] * esz, msg_bytes(recvcount, recvtype));
      return 0;
   }
   if (me == root) { k->rbuf = sendbuf; k->rcnt = sendcounts; k->rdsp = displs; }
   comm_barrier(comm);
   {
      /* sendcounts/displs are root-only in MPI; read the root's copies */
      const int *sc = (const int *) k->rcnt;
      const int *dp = (const int *) k->rdsp;
      memcpy(recvbuf, (const char *) k->rbuf + (size_t) dp[me] * esz,
             (size_t) sc[me] * esz);
   }
   comm_barrier(comm);
   return 0;
}

int MPI_Alltoall( const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                              void *recvbuf, int recvcount, MPI_Datatype recvtype,
                              MPI_Comm comm )
{
   if (sendbuf == MPI_IN_PLACE) { no_inplace("MPI_Alltoall"); }
   nmpi_comm *k = comm_get(comm);
   int me = comm_rank_of(comm), i, sz;
   size_t sb = msg_bytes(sendcount, sendtype), rb = msg_bytes(recvcount, recvtype);

   if (comm == MPI_COMM_SELF || !k || k->size == 1) { memcpy(recvbuf, sendbuf, sb); return 0; }
   sz = k->size;
   k->ptr[me] = sendbuf;
   comm_barrier(comm);
   for (i = 0; i < sz; i++)
   {
      memcpy((char *) recvbuf + (size_t) i * rb,
             (const char *) k->ptr[i] + (size_t) me * sb, sb);
   }
   comm_barrier(comm);
   return 0;
}

/*--------------------------------------------------------------------------
 * Point-to-point entry points
 *--------------------------------------------------------------------------*/
int MPI_Send( const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm )
{
   if (dest == MPI_PROC_NULL) { return MPI_SUCCESS; }
   do_send(buf, count, datatype, dest, tag, comm);
   return MPI_SUCCESS;
}

int MPI_Recv( void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm,
              MPI_Status *status )
{
   NANOMPI_UNUSED(count); NANOMPI_UNUSED(datatype); NANOMPI_UNUSED(buf);
   if (source == MPI_PROC_NULL)
   {
      if (status)
      {
         status->MPI_SOURCE = MPI_PROC_NULL;
         status->MPI_TAG    = MPI_ANY_TAG;
         status->MPI_ERROR  = MPI_SUCCESS;
         status->nanompi_count = 0;
      }
      return MPI_SUCCESS;
   }
   do_recv(buf, count, datatype, source, tag, comm, status);
   return MPI_SUCCESS;
}

/* Post the receive first so the pair cannot deadlock even when both peers call
   this at the same moment with no eager buffering available. */
int MPI_Sendrecv( const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  int dest, int sendtag,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int source, int recvtag,
                  MPI_Comm comm, MPI_Status *status )
{
   MPI_Request req;
   int ierr;

   ierr = MPI_Irecv(recvbuf, recvcount, recvtype, source, recvtag, comm, &req);
   if (ierr != MPI_SUCCESS) { return ierr; }
   ierr = MPI_Send(sendbuf, sendcount, sendtype, dest, sendtag, comm);
   if (ierr != MPI_SUCCESS) { return ierr; }
   return MPI_Wait(&req, status);
}

int MPI_Isend( const void *buf, int count, MPI_Datatype datatype,
                           int dest, int tag, MPI_Comm comm,
                           MPI_Request *request )
{
   nmpi_req *r;
   do_send(buf, count, datatype, dest, tag, comm);   /* eager: complete on return */
   *request = req_alloc();
   r = req_get(*request);
   r->kind = 1;
   return 0;
}

int MPI_Irsend( const void *buf, int count, MPI_Datatype datatype,
                            int dest, int tag, MPI_Comm comm,
                            MPI_Request *request )
{
   return MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int MPI_Irecv( void *buf, int count, MPI_Datatype datatype,
                           int source, int tag, MPI_Comm comm,
                           MPI_Request *request )
{
   nmpi_req *r;
   /* post immediately so message matching follows posting order */
   nmpi_pending *p = post_recv(buf, count, datatype, source, tag, comm);
   *request = req_alloc();
   r = req_get(*request);
   r->kind = 2; r->p = p;
   return 0;
}

static void req_init_persistent(int kind, void *buf, int count,
                                MPI_Datatype dt, int peer, int tag,
                                MPI_Comm comm, MPI_Request *request)
{
   nmpi_req *r;
   *request = req_alloc();
   r = req_get(*request);
   r->kind = kind; r->active = 0; r->p = NULL;
   r->buf = buf; r->count = (int) count; r->dt = (int) dt;
   r->peer = (int) peer; r->tag = (int) tag; r->comm = (int) comm;
}

int MPI_Send_init( const void *buf, int count, MPI_Datatype datatype,
                               int dest, int tag, MPI_Comm comm,
                               MPI_Request *request )
{
   /* the persistent request only replays the send; it never writes here */
   req_init_persistent(3, (void *) buf, count, datatype, dest, tag, comm, request);
   return 0;
}

int MPI_Recv_init( void *buf, int count, MPI_Datatype datatype,
                               int dest, int tag, MPI_Comm comm,
                               MPI_Request *request )
{
   req_init_persistent(4, buf, count, datatype, dest, tag, comm, request);
   return 0;
}

/* Activate persistent requests. Receives are posted before any send goes out so
   that matching still follows posting order. */
int MPI_Startall( int count, MPI_Request *array_of_requests );

int MPI_Start( MPI_Request *request )
{
   return MPI_Startall(1, request);
}

int MPI_Startall( int count, MPI_Request *array_of_requests )
{
   int i;
   nmpi_req *r;

   for (i = 0; i < count; i++)
   {
      r = req_get(array_of_requests[i]);
      if (r && r->kind == 4 && !r->active)
      {
         r->p = post_recv(r->buf, r->count, r->dt, r->peer, r->tag, r->comm);
         r->active = 1;
      }
   }
   for (i = 0; i < count; i++)
   {
      r = req_get(array_of_requests[i]);
      if (r && r->kind == 3 && !r->active)
      {
         do_send(r->buf, r->count, r->dt, r->peer, r->tag, r->comm);
         r->active = 1;
      }
   }
   return 0;
}

int MPI_Wait( MPI_Request *request, MPI_Status *status )
{
   nmpi_req *r = req_get(*request);
   if (!r) { return 0; }

   if (r->kind == 4)            /* persistent recv: completes, handle stays valid */
   {
      if (r->active) { wait_pending(r->p, status); r->p = NULL; r->active = 0; }
      return 0;
   }
   if (r->kind == 3)            /* persistent send: eager, so already done */
   {
      r->active = 0;
      if (status) { status->MPI_SOURCE = 0; status->MPI_TAG = 0; status->nanompi_count = 0; }
      return 0;
   }
   if (r->kind == 2) { wait_pending(r->p, status); r->p = NULL; }
   else if (status)
   {
      status->MPI_SOURCE = 0; status->MPI_TAG = 0; status->nanompi_count = 0;
   }
   r->used = 0; r->kind = 0;
   *request = MPI_REQUEST_NULL;
   return 0;
}

int MPI_Waitall( int count, MPI_Request *array_of_requests,
                             MPI_Status *array_of_statuses )
{
   int i;
   /* matching already happened at post time, so any completion order is safe */
   for (i = 0; i < count; i++) { MPI_Wait(&array_of_requests[i], NULL); }
   NANOMPI_UNUSED(array_of_statuses);
   return 0;
}

int MPI_Waitany( int count, MPI_Request *array_of_requests,
                             int *index, MPI_Status *status )
{
   int i;
   for (i = 0; i < count; i++)
   {
      if (array_of_requests[i] != MPI_REQUEST_NULL)
      {
         MPI_Wait(&array_of_requests[i], status);
         *index = i;
         return 0;
      }
   }
   *index = MPI_UNDEFINED;
   return 0;
}

int MPI_Test( MPI_Request *request, int *flag, MPI_Status *status )
{
   nmpi_req   *r = req_get(*request);
   nmpi_inbox *b;
   int done;

   if (!r) { *flag = 1; return 0; }
   if (r->kind == 1 || r->kind == 3) { MPI_Wait(request, status); *flag = 1; return 0; }
   if (r->kind == 4 && !r->active) { *flag = 1; return 0; }

   b = &g_inbox[g_myrank];
   pthread_mutex_lock(&b->mtx);
   done = r->p->done;
   pthread_mutex_unlock(&b->mtx);

   if (done) { MPI_Wait(request, status); *flag = 1; }
   else { *flag = 0; }
   return 0;
}

int MPI_Testall( int count, MPI_Request *array_of_requests,
                             int *flag, MPI_Status *array_of_statuses )
{
   int i, f, all = 1;
   for (i = 0; i < count; i++)
   {
      MPI_Test(&array_of_requests[i], &f, NULL);
      if (!f) { all = 0; }
   }
   *flag = all;
   NANOMPI_UNUSED(array_of_statuses);
   return 0;
}

int MPI_Probe( int source, int tag, MPI_Comm comm,
                           MPI_Status *status )
{
   probe_umq(comm, source, tag, 1, status);
   return 0;
}

int MPI_Iprobe( int source, int tag, MPI_Comm comm,
                            int *flag, MPI_Status *status )
{
   *flag = probe_umq(comm, source, tag, 0, status) ? 1 : 0;
   return 0;
}

int MPI_Request_free( MPI_Request *request )
{
   nmpi_req *r = req_get(*request);
   if (r)
   {
      if ((r->kind == 2 || r->kind == 4) && r->p) { wait_pending(r->p, NULL); r->p = NULL; }
      r->used = 0; r->kind = 0; r->active = 0;
   }
   *request = MPI_REQUEST_NULL;
   return 0;
}

/*--------------------------------------------------------------------------
 * Derived datatypes
 *--------------------------------------------------------------------------*/
static int dt_make(int nblk, const size_t *len, const intptr_t *disp, int absolute)
{
   int h = dt_alloc(), b;
   nmpi_dt *d = &g_dt[h];
   d->nblk = nblk;
   d->len  = (size_t *) malloc(sizeof(size_t) * (size_t) nblk);
   d->disp = (intptr_t *) malloc(sizeof(intptr_t) * (size_t) nblk);
   d->abs  = absolute;
   d->size = 0;
   for (b = 0; b < nblk; b++)
   {
      d->len[b] = len[b];
      d->disp[b] = disp[b];
      d->size += len[b];
   }
   return h;
}

int MPI_Type_contiguous( int count, MPI_Datatype oldtype,
                                     MPI_Datatype *newtype )
{
   size_t len = (size_t) count * dt_size(oldtype);
   intptr_t disp = 0;
   *newtype = dt_make(1, &len, &disp, 0);
   return 0;
}

int MPI_Type_vector( int count, int blocklength, int stride,
                                 MPI_Datatype oldtype, MPI_Datatype *newtype )
{
   size_t esz = dt_size(oldtype), *len;
   intptr_t *disp;
   int i;

   len  = (size_t *) malloc(sizeof(size_t) * (size_t) count);
   disp = (intptr_t *) malloc(sizeof(intptr_t) * (size_t) count);
   for (i = 0; i < count; i++)
   {
      len[i]  = (size_t) blocklength * esz;
      disp[i] = (intptr_t)((size_t) i * (size_t) stride * esz);
   }
   *newtype = dt_make((int) count, len, disp, 0);
   free(len); free(disp);
   return 0;
}

int MPI_Type_hvector( int count, int blocklength, MPI_Aint stride,
                                  MPI_Datatype oldtype, MPI_Datatype *newtype )
{
   size_t esz = dt_size(oldtype), *len;
   intptr_t *disp;
   int i;

   len  = (size_t *) malloc(sizeof(size_t) * (size_t) count);
   disp = (intptr_t *) malloc(sizeof(intptr_t) * (size_t) count);
   for (i = 0; i < count; i++)
   {
      len[i]  = (size_t) blocklength * esz;
      disp[i] = (intptr_t)((size_t) i * (size_t) stride);
   }
   *newtype = dt_make((int) count, len, disp, 0);
   free(len); free(disp);
   return 0;
}

/* Built either relatively, or with MPI_Get_address, i.e. absolute addresses
   used together with MPI_BOTTOM. */
int MPI_Type_struct( int count, const int *array_of_blocklengths,
                     const MPI_Aint *array_of_displacements,
                     const MPI_Datatype *array_of_types,
                     MPI_Datatype *newtype )
{
   size_t *len = (size_t *) malloc(sizeof(size_t) * (size_t) count);
   intptr_t *disp = (intptr_t *) malloc(sizeof(intptr_t) * (size_t) count);
   int i;

   for (i = 0; i < count; i++)
   {
      len[i]  = (size_t) array_of_blocklengths[i] * dt_size(array_of_types[i]);
      disp[i] = (intptr_t) array_of_displacements[i];
   }
   *newtype = dt_make((int) count, len, disp, 1);
   free(len); free(disp);
   return 0;
}

int MPI_Type_size( MPI_Datatype datatype, int *size )
{
   size_t n = dt_size(datatype);
   if (n == 0 && datatype != MPI_DATATYPE_NULL && !dt_get(datatype))
   {
      *size = 0;
      return MPI_ERR_TYPE;
   }
   *size = (int) n;
   return MPI_SUCCESS;
}

/* No type here carries an explicit lower bound or a trailing gap, so the extent
   and the size are the same. MPI_UB and MPI_LB are not supported. */
int MPI_Type_extent( MPI_Datatype datatype, MPI_Aint *extent )
{
   *extent = (MPI_Aint) dt_size(datatype);
   return MPI_SUCCESS;
}

int MPI_Get_address( const void *location, MPI_Aint *address )
{
   return MPI_Address(location, address);
}

int MPI_Type_create_hvector( int count, int blocklength, MPI_Aint stride,
                             MPI_Datatype oldtype, MPI_Datatype *newtype )
{
   return MPI_Type_hvector(count, blocklength, stride, oldtype, newtype);
}

int MPI_Type_create_struct( int count, const int *blocklengths,
                            const MPI_Aint *displacements,
                            const MPI_Datatype *types, MPI_Datatype *newtype )
{
   return MPI_Type_struct(count, blocklengths, displacements, types, newtype);
}

int MPI_Type_commit( MPI_Datatype *datatype )
{
   NANOMPI_UNUSED(datatype);
   return 0;
}

int MPI_Type_free( MPI_Datatype *datatype )
{
   nmpi_dt *d = dt_get(*datatype);
   if (d)
   {
      pthread_mutex_lock(&g_dt_mtx);
      free(d->len); free(d->disp);
      d->len = NULL; d->disp = NULL; d->used = 0;
      pthread_mutex_unlock(&g_dt_mtx);
   }
   return 0;
}

int MPI_Op_free( MPI_Op *op )
{
   if (*op >= NMPI_FIRST_USER_OP && *op < NMPI_MAX_OPS)
   {
      pthread_mutex_lock(&g_op_mtx);
      g_op[*op].used = 0; g_op[*op].fn = NULL;
      pthread_mutex_unlock(&g_op_mtx);
   }
   return 0;
}

int MPI_Op_create( MPI_User_function *function, int commute,
                               MPI_Op *op )
{
   int i, h = -1;
   NANOMPI_UNUSED(commute);
   pthread_mutex_lock(&g_op_mtx);
   for (i = NMPI_FIRST_USER_OP; i < NMPI_MAX_OPS; i++)
   {
      if (!g_op[i].used) { h = i; g_op[i].used = 1; g_op[i].fn = (nmpi_user_fn) function; break; }
   }
   pthread_mutex_unlock(&g_op_mtx);
   if (h < 0) { fprintf(stderr, "nano-mpi: out of op handles\n"); abort(); }
   *op = (MPI_Op) h;
   return 0;
}

/*--------------------------------------------------------------------------
 * Shared-memory windows
 *
 * Ranks are threads, so "shared memory" needs no mapping tricks: one rank
 * allocates the whole block and everyone learns the pointer. What still has to
 * be right is the layout MPI promises -- the window is the concatenation of
 * every rank's segment, in rank order, contiguously -- because that is what
 * MPI_Win_shared_query is asked about and what real code walks.
 *
 * There is no MPI_Put/Get/Accumulate. You have the pointer; use it.
 *--------------------------------------------------------------------------*/
#define NMPI_MAX_WINS 256

typedef struct
{
   int       used;
   char     *base;    /* whole block; NULL when every segment is empty */
   size_t    total;
   MPI_Comm  comm;
   int       size;    /* number of ranks in comm */
   size_t   *off;     /* byte offset of each rank's segment */
   size_t   *len;     /* bytes in each rank's segment */
   int      *disp;    /* each rank's disp_unit */
   int       owner;   /* comm rank that allocated, and will free */
} nmpi_win;

static nmpi_win        g_win[NMPI_MAX_WINS];
static pthread_mutex_t g_win_mtx = PTHREAD_MUTEX_INITIALIZER;

static nmpi_win *win_get(MPI_Win w)
{
   if (w <= 0 || w >= NMPI_MAX_WINS || !g_win[w].used) { return NULL; }
   return &g_win[w];
}

static int win_alloc(void)
{
   int i, h = -1;
   pthread_mutex_lock(&g_win_mtx);
   for (i = 1; i < NMPI_MAX_WINS; i++) { if (!g_win[i].used) { h = i; break; } }
   if (h >= 0) { g_win[h].used = 1; }
   pthread_mutex_unlock(&g_win_mtx);
   if (h < 0) { fprintf(stderr, "nano-mpi: out of window handles\n"); abort(); }
   return h;
}

typedef struct { size_t bytes; int disp; } nmpi_win_seg;

int MPI_Win_allocate_shared( MPI_Aint size, int disp_unit, MPI_Info info,
                             MPI_Comm comm, void *baseptr, MPI_Win *win )
{
   int me = comm_rank_of(comm), sz = comm_size_of(comm), i;
   nmpi_win_seg  mine, *segs;
   nmpi_win     *w;
   struct { void *base; int handle; } hdr;

   NANOMPI_UNUSED(info);
   if (size < 0 || sz <= 0) { return MPI_ERR_ARG; }

   mine.bytes = (size_t) size;
   mine.disp  = disp_unit;
   segs = (nmpi_win_seg *) malloc(sizeof(nmpi_win_seg) * (size_t) sz);
   if (!segs) { return MPI_ERR_NO_MEM; }
   MPI_Allgather(&mine, (int) sizeof mine, MPI_BYTE,
                 segs,  (int) sizeof mine, MPI_BYTE, comm);

   hdr.base = NULL; hdr.handle = MPI_WIN_NULL;
   if (me == 0)
   {
      size_t total = 0;
      for (i = 0; i < sz; i++) { total += segs[i].bytes; }
      /* MPI says a freshly allocated window is not zeroed, but zeroing it turns
         a whole class of "forgot to initialise" bugs into reproducible ones. */
      hdr.base   = total ? calloc(1, total) : NULL;
      if (total && !hdr.base) { free(segs); return MPI_ERR_NO_MEM; }
      hdr.handle = win_alloc();

      w = &g_win[hdr.handle];
      w->base  = (char *) hdr.base;
      w->total = total;
      w->comm  = comm;
      w->size  = sz;
      w->owner = 0;
      w->off   = (size_t *) malloc(sizeof(size_t) * (size_t) sz);
      w->len   = (size_t *) malloc(sizeof(size_t) * (size_t) sz);
      w->disp  = (int *)    malloc(sizeof(int)    * (size_t) sz);
      {
         size_t at = 0;
         for (i = 0; i < sz; i++)
         {
            w->off[i]  = at;
            w->len[i]  = segs[i].bytes;
            w->disp[i] = segs[i].disp;
            at += segs[i].bytes;
         }
      }
   }
   MPI_Bcast(&hdr, (int) sizeof hdr, MPI_BYTE, 0, comm);
   free(segs);

   w = win_get(hdr.handle);
   if (!w) { return MPI_ERR_INTERN; }

   *(void **) baseptr = w->len[me] ? (w->base + w->off[me]) : NULL;
   *win = hdr.handle;
   return MPI_SUCCESS;
}

int MPI_Win_shared_query( MPI_Win win, int rank, MPI_Aint *size,
                          int *disp_unit, void *baseptr )
{
   nmpi_win *w = win_get(win);
   int r = rank;

   if (!w) { return MPI_ERR_ARG; }
   if (r == MPI_PROC_NULL)
   {
      /* the standard's answer for MPI_PROC_NULL: the first non-empty segment */
      for (r = 0; r < w->size && w->len[r] == 0; r++) { }
      if (r == w->size) { r = 0; }
   }
   if (r < 0 || r >= w->size) { return MPI_ERR_RANK; }

   if (size)      { *size      = (MPI_Aint) w->len[r]; }
   if (disp_unit) { *disp_unit = w->disp[r]; }
   *(void **) baseptr = w->len[r] ? (w->base + w->off[r]) : NULL;
   return MPI_SUCCESS;
}

int MPI_Win_free( MPI_Win *win )
{
   nmpi_win *w;
   int me;

   if (!win) { return MPI_ERR_ARG; }
   w = win_get(*win);
   if (!w) { *win = MPI_WIN_NULL; return MPI_SUCCESS; }

   me = comm_rank_of(w->comm);
   comm_barrier(w->comm);          /* nobody may still be reading the block */
   if (me == w->owner)
   {
      pthread_mutex_lock(&g_win_mtx);
      free(w->base); free(w->off); free(w->len); free(w->disp);
      w->base = NULL; w->off = NULL; w->len = NULL; w->disp = NULL;
      w->used = 0;
      pthread_mutex_unlock(&g_win_mtx);
   }
   comm_barrier(w->comm);          /* and nobody may free it twice */
   *win = MPI_WIN_NULL;
   return MPI_SUCCESS;
}

/* The memory is genuinely shared, so these carry no data. What they do carry is
   ordering, which still matters: a store by one rank is only guaranteed visible
   to another after a synchronisation point. */
int MPI_Win_fence( int assert_, MPI_Win win )
{
   nmpi_win *w = win_get(win);
   NANOMPI_UNUSED(assert_);
   if (!w) { return MPI_ERR_ARG; }
   __atomic_thread_fence(__ATOMIC_SEQ_CST);
   comm_barrier(w->comm);
   return MPI_SUCCESS;
}

int MPI_Win_sync( MPI_Win win )
{
   if (!win_get(win)) { return MPI_ERR_ARG; }
   __atomic_thread_fence(__ATOMIC_SEQ_CST);
   return MPI_SUCCESS;
}

int MPI_Win_lock_all( int assert_, MPI_Win win )
{
   NANOMPI_UNUSED(assert_);
   return win_get(win) ? MPI_SUCCESS : MPI_ERR_ARG;
}

int MPI_Win_unlock_all( MPI_Win win )
{
   return MPI_Win_sync(win);
}

int MPI_Info_create( MPI_Info *info ) { *info = 0; return 0; }
int MPI_Info_free( MPI_Info *info ) { NANOMPI_UNUSED(info); return 0; }


