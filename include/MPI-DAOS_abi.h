#ifndef MPIDAOS_ABI_H
#define MPIDAOS_ABI_H

#include <stdint.h>
#include <stddef.h>
#include <daos.h>
#include <daos_array.h>
#include <daos_kv.h>
#include <pthread.h>

/* ABI-neutral types: 64-bit intptr_t holds both MPICH int handles and OpenMPI pointers */
typedef intptr_t  U_MPI_Comm;
typedef intptr_t  U_MPI_Datatype;
typedef intptr_t  U_MPI_Info;
typedef intptr_t  U_MPI_File;
typedef int64_t   U_MPI_Offset;
typedef intptr_t  U_MPI_Aint;
typedef int       U_MPI_Fint;

/* Standard MPI return codes (identical across MPICH and OpenMPI) */
#define U_MPI_SUCCESS       0
#define U_MPI_ERR_BUFFER    1
#define U_MPI_ERR_COUNT     2
#define U_MPI_ERR_TYPE      3
#define U_MPI_ERR_COMM      5
#define U_MPI_ERR_INFO      28
#define U_MPI_ERR_NO_MEM    34
#define U_MPI_ERR_FILE      27
#define U_MPI_ERR_BAD_FILE  20
#define U_MPI_ERR_IO        32

/* File open modes (universal POSIX-mapped bitmasks) */
#define U_MPI_MODE_RDONLY   2
#define U_MPI_MODE_RDWR     8
#define U_MPI_MODE_WRONLY   4
#define U_MPI_MODE_CREATE   1
#define U_MPI_SEEK_SET      600
#define U_MPI_SEEK_CUR      602
#define U_MPI_SEEK_END      604

/* Runtime-resolved MPI symbols, constants, and sentinel values */
typedef struct {
    int            is_openmpi;
    U_MPI_Comm     comm_world;
    U_MPI_Datatype dtype_byte;
    U_MPI_Datatype dtype_int;
    U_MPI_Datatype dtype_uint64;
    U_MPI_Datatype dtype_int64;
    U_MPI_Info     info_null;
    U_MPI_File     file_null;
    void          *status_ignore;
    U_MPI_Fint    *f_status_ignore;
    int            op_sum;

    /* Combiner constants (differ between MPICH and OpenMPI) */
    int            combiner_named;
    int            combiner_dup;
    int            combiner_contiguous;
    int            combiner_vector;
    int            combiner_hvector;
    int            combiner_indexed;
    int            combiner_hindexed;
    int            combiner_struct;
    int            combiner_resized;

    /* Resolved PMPI function pointers */
    int (*Init)(int *, char ***);
    int (*Init_thread)(int *, char ***, int, int *);
    int (*Finalize)(void);
    int (*Comm_rank)(U_MPI_Comm, int *);
    int (*Comm_size)(U_MPI_Comm, int *);
    int (*Comm_dup)(U_MPI_Comm, U_MPI_Comm *);
    int (*Comm_free)(U_MPI_Comm *);
    int (*Barrier)(U_MPI_Comm);
    int (*Bcast)(void *, int, U_MPI_Datatype, int, U_MPI_Comm);
    int (*Exscan)(const void *, void *, int, U_MPI_Datatype, int, U_MPI_Comm);
    int (*Type_size)(U_MPI_Datatype, int *);
    int (*Type_get_extent)(U_MPI_Datatype, U_MPI_Aint *, U_MPI_Aint *);
    int (*Type_get_envelope)(U_MPI_Datatype, int *, int *, int *, int *);
    int (*Type_get_contents)(U_MPI_Datatype, int, int, int, int *, U_MPI_Aint *, U_MPI_Datatype *);
    int (*Type_dup)(U_MPI_Datatype, U_MPI_Datatype *);
    int (*Type_free)(U_MPI_Datatype *);
    int (*Status_set_elements)(void *, U_MPI_Datatype, int);

    /* Fortran handle converters */
    U_MPI_Comm     (*Comm_f2c)(U_MPI_Fint);
    U_MPI_Datatype (*Type_f2c)(U_MPI_Fint);
    U_MPI_Info     (*Info_f2c)(U_MPI_Fint);
    int            (*Status_c2f)(const void *, U_MPI_Fint *);
} mpi_abi_t;

extern mpi_abi_t g_mpi;
void mpidaos_resolve_mpi_abi(void);

#endif /* MPIDAOS_ABI_H */
