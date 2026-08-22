/*****************************************************************************
 * nano-mpi -- MPI ranks as threads of one process.
 *
 * This header intentionally carries the name and the spelling of the standard
 * MPI C interface, so that code using the supported subset compiles unchanged.
 * The supported subset, and everything deliberately left out, is specified in
 * SCOPE.md. Read section 5 of it before porting anything: ranks are threads,
 * so your file-scope mutable state is shared, not per-rank.
 *
 * Handles are int. That is source-compatible with the standard and ABI
 * compatible with nothing. You must recompile to switch implementations.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *****************************************************************************/

#ifndef NANOMPI_MPI_H
#define NANOMPI_MPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NANOMPI_VERSION_MAJOR 0
#define NANOMPI_VERSION_MINOR 1
#define NANOMPI_VERSION_PATCH 0

/* The level of the standard whose spelling we follow. This is not a
 * conformance claim; see SCOPE.md section 11. */
#define MPI_VERSION    3
#define MPI_SUBVERSION 1

/*--------------------------------------------------------------------------
 * Handle types
 *
 * Zero is the null handle for every kind, so a zero-initialised handle is
 * null rather than accidentally MPI_COMM_WORLD.
 *--------------------------------------------------------------------------*/
typedef int MPI_Comm;
typedef int MPI_Group;
typedef int MPI_Datatype;
typedef int MPI_Request;
typedef int MPI_Op;
typedef int MPI_Info;
typedef int MPI_Errhandler;

typedef intptr_t  MPI_Aint;
typedef long long MPI_Offset;
typedef long long MPI_Count;

typedef struct
{
   int MPI_SOURCE;
   int MPI_TAG;
   int MPI_ERROR;
   /* not part of the standard interface; MPI_Get_count reads it */
   int nanompi_count;
} MPI_Status;

typedef void (MPI_User_function)(void *invec, void *inoutvec,
                                 int *len, MPI_Datatype *datatype);
typedef int (MPI_Copy_function)(MPI_Comm, int, void *, void *, void *, int *);
typedef int (MPI_Delete_function)(MPI_Comm, int, void *, void *);

/*--------------------------------------------------------------------------
 * Communicators and groups
 *--------------------------------------------------------------------------*/
#define MPI_COMM_NULL   0
#define MPI_COMM_WORLD  1
#define MPI_COMM_SELF   2

#define MPI_GROUP_NULL  0
#define MPI_GROUP_EMPTY 1

/* MPI_Comm_compare results */
#define MPI_IDENT     0
#define MPI_CONGRUENT 1
#define MPI_SIMILAR   2
#define MPI_UNEQUAL   3

/* MPI_Comm_split_type */
#define MPI_COMM_TYPE_SHARED 1

/*--------------------------------------------------------------------------
 * Predefined datatypes
 *--------------------------------------------------------------------------*/
#define MPI_DATATYPE_NULL          0
#define MPI_CHAR                   1
#define MPI_SIGNED_CHAR            2
#define MPI_UNSIGNED_CHAR          3
#define MPI_BYTE                   4
#define MPI_WCHAR                  5
#define MPI_SHORT                  6
#define MPI_UNSIGNED_SHORT         7
#define MPI_INT                    8
#define MPI_UNSIGNED               9
#define MPI_LONG                  10
#define MPI_UNSIGNED_LONG         11
#define MPI_LONG_LONG_INT         12
#define MPI_LONG_LONG             MPI_LONG_LONG_INT
#define MPI_UNSIGNED_LONG_LONG    13
#define MPI_FLOAT                 14
#define MPI_DOUBLE                15
#define MPI_LONG_DOUBLE           16
#define MPI_C_BOOL                17
#define MPI_INT8_T                18
#define MPI_INT16_T               19
#define MPI_INT32_T               20
#define MPI_INT64_T               21
#define MPI_UINT8_T               22
#define MPI_UINT16_T              23
#define MPI_UINT32_T              24
#define MPI_UINT64_T              25
#define MPI_C_FLOAT_COMPLEX       26
#define MPI_C_COMPLEX             MPI_C_FLOAT_COMPLEX
#define MPI_C_DOUBLE_COMPLEX      27
#define MPI_C_LONG_DOUBLE_COMPLEX 28
#define MPI_AINT                  29
#define MPI_OFFSET                30
#define MPI_COUNT                 31
#define MPI_PACKED                32
/* pair types, for MPI_MAXLOC and MPI_MINLOC */
#define MPI_FLOAT_INT             33
#define MPI_DOUBLE_INT            34
#define MPI_LONG_INT              35
#define MPI_SHORT_INT             36
#define MPI_2INT                  37
#define MPI_LONG_DOUBLE_INT       38

/* struct layouts the pair types above describe */
typedef struct { float       value; int index; } MPI_Float_int;
typedef struct { double      value; int index; } MPI_Double_int;
typedef struct { long        value; int index; } MPI_Long_int;
typedef struct { short       value; int index; } MPI_Short_int;
typedef struct { int         value; int index; } MPI_2int;
typedef struct { long double value; int index; } MPI_Long_double_int;

/*--------------------------------------------------------------------------
 * Reduction operations
 *--------------------------------------------------------------------------*/
#define MPI_OP_NULL  0
#define MPI_MAX      1
#define MPI_MIN      2
#define MPI_SUM      3
#define MPI_PROD     4
#define MPI_LAND     5
#define MPI_BAND     6
#define MPI_LOR      7
#define MPI_BOR      8
#define MPI_LXOR     9
#define MPI_BXOR    10
#define MPI_MINLOC  11
#define MPI_MAXLOC  12
#define MPI_REPLACE 13

/*--------------------------------------------------------------------------
 * Sentinels
 *--------------------------------------------------------------------------*/
#define MPI_ANY_SOURCE  (-1)
#define MPI_ANY_TAG     (-2)
#define MPI_PROC_NULL   (-3)
#define MPI_ROOT        (-4)
#define MPI_UNDEFINED   (-32766)

#define MPI_BOTTOM           ((void *) 0)
#define MPI_IN_PLACE         ((void *) -1)
#define MPI_STATUS_IGNORE    ((MPI_Status *) 0)
#define MPI_STATUSES_IGNORE  ((MPI_Status *) 0)
#define MPI_ARGV_NULL        ((char **) 0)
#define MPI_ERRCODES_IGNORE  ((int *) 0)

#define MPI_REQUEST_NULL     0
#define MPI_INFO_NULL        0
#define MPI_ERRHANDLER_NULL  0

/* Thread support levels; see SCOPE.md section 7 */
#define MPI_THREAD_SINGLE     0
#define MPI_THREAD_FUNNELED   1
#define MPI_THREAD_SERIALIZED 2
#define MPI_THREAD_MULTIPLE   3

/*--------------------------------------------------------------------------
 * Sizes
 *--------------------------------------------------------------------------*/
#define MPI_MAX_PROCESSOR_NAME        256
#define MPI_MAX_ERROR_STRING          256
#define MPI_MAX_LIBRARY_VERSION_STRING 256
#define MPI_MAX_OBJECT_NAME           128
#define MPI_MAX_INFO_KEY               64
#define MPI_MAX_INFO_VAL              256

/*--------------------------------------------------------------------------
 * Error classes
 *--------------------------------------------------------------------------*/
#define MPI_SUCCESS         0
#define MPI_ERR_BUFFER      1
#define MPI_ERR_COUNT       2
#define MPI_ERR_TYPE        3
#define MPI_ERR_TAG         4
#define MPI_ERR_COMM        5
#define MPI_ERR_RANK        6
#define MPI_ERR_REQUEST     7
#define MPI_ERR_ROOT        8
#define MPI_ERR_GROUP       9
#define MPI_ERR_OP         10
#define MPI_ERR_TOPOLOGY   11
#define MPI_ERR_DIMS       12
#define MPI_ERR_ARG        13
#define MPI_ERR_UNKNOWN    14
#define MPI_ERR_TRUNCATE   15
#define MPI_ERR_OTHER      16
#define MPI_ERR_INTERN     17
#define MPI_ERR_IN_STATUS  18
#define MPI_ERR_PENDING    19
#define MPI_ERR_SPAWN      20
#define MPI_ERR_NO_MEM     21
#define MPI_ERR_UNSUPPORTED_OPERATION 22
#define MPI_ERR_LASTCODE   23

/*--------------------------------------------------------------------------
 * Startup, shutdown, environment
 *--------------------------------------------------------------------------*/
int MPI_Init(int *argc, char ***argv);
int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);
int MPI_Initialized(int *flag);
int MPI_Finalize(void);
int MPI_Finalized(int *flag);
int MPI_Query_thread(int *provided);
int MPI_Is_thread_main(int *flag);
int MPI_Abort(MPI_Comm comm, int errorcode);
int MPI_Get_version(int *version, int *subversion);
int MPI_Get_library_version(char *version, int *resultlen);
int MPI_Get_processor_name(char *name, int *resultlen);
int MPI_Error_string(int errorcode, char *string, int *resultlen);
int MPI_Error_class(int errorcode, int *errorclass);

double MPI_Wtime(void);
double MPI_Wtick(void);

/*--------------------------------------------------------------------------
 * Point-to-point
 *--------------------------------------------------------------------------*/
int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm);
int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status *status);
int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status);

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Irsend(const void *buf, int count, MPI_Datatype datatype,
               int dest, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request);

int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status);
int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);

int MPI_Wait(MPI_Request *request, MPI_Status *status);
int MPI_Waitall(int count, MPI_Request *requests, MPI_Status *statuses);
int MPI_Waitany(int count, MPI_Request *requests, int *index, MPI_Status *status);
int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
int MPI_Testall(int count, MPI_Request *requests, int *flag, MPI_Status *statuses);
int MPI_Request_free(MPI_Request *request);

int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype,
                  int dest, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype,
                  int source, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Start(MPI_Request *request);
int MPI_Startall(int count, MPI_Request *requests);

/*--------------------------------------------------------------------------
 * Collectives
 *--------------------------------------------------------------------------*/
int MPI_Barrier(MPI_Comm comm);
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, int root, MPI_Comm comm);
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                  MPI_Op op, MPI_Comm comm);
int MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
             MPI_Op op, MPI_Comm comm);
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm);
int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, const int *recvcounts, const int *displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm);
int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, const int *recvcounts, const int *displs,
                   MPI_Datatype recvtype, MPI_Comm comm);
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm);
int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                 MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype, int root, MPI_Comm comm);
int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Op_create(MPI_User_function *function, int commute, MPI_Op *op);
int MPI_Op_free(MPI_Op *op);

/*--------------------------------------------------------------------------
 * Communicators and groups
 *--------------------------------------------------------------------------*/
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
int MPI_Comm_free(MPI_Comm *comm);
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key,
                        MPI_Info info, MPI_Comm *newcomm);
int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);
int MPI_Comm_group(MPI_Comm comm, MPI_Group *group);
int MPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int *result);

int MPI_Group_size(MPI_Group group, int *size);
int MPI_Group_rank(MPI_Group group, int *rank);
int MPI_Group_incl(MPI_Group group, int n, const int *ranks, MPI_Group *newgroup);
int MPI_Group_free(MPI_Group *group);

/*--------------------------------------------------------------------------
 * Datatypes
 *--------------------------------------------------------------------------*/
int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_vector(int count, int blocklength, int stride,
                    MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_hvector(int count, int blocklength, MPI_Aint stride,
                     MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride,
                            MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_struct(int count, const int *blocklengths,
                    const MPI_Aint *displacements, const MPI_Datatype *types,
                    MPI_Datatype *newtype);
int MPI_Type_create_struct(int count, const int *blocklengths,
                           const MPI_Aint *displacements, const MPI_Datatype *types,
                           MPI_Datatype *newtype);
int MPI_Type_commit(MPI_Datatype *datatype);
int MPI_Type_free(MPI_Datatype *datatype);
int MPI_Type_size(MPI_Datatype datatype, int *size);
int MPI_Type_extent(MPI_Datatype datatype, MPI_Aint *extent);
int MPI_Address(const void *location, MPI_Aint *address);
int MPI_Get_address(const void *location, MPI_Aint *address);

/*--------------------------------------------------------------------------
 * Info
 *--------------------------------------------------------------------------*/
int MPI_Info_create(MPI_Info *info);
int MPI_Info_free(MPI_Info *info);

#ifdef __cplusplus
}
#endif

#endif /* NANOMPI_MPI_H */
