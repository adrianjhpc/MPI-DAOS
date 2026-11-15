#ifndef MPIDAOS_INTERNAL_H
#define MPIDAOS_INTERNAL_H

#include "MPI-DAOS_abi.h"
#include <daos.h>
#include <daos_array.h>
#include <daos_kv.h>
#include <pthread.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    daos_handle_t   poh;
    daos_handle_t   coh;
    daos_handle_t   oh;
    daos_obj_id_t   oid;
    daos_handle_t   kv_oh;
    daos_obj_id_t   kv_oid;
    U_MPI_Comm        comm;
    U_MPI_Offset      disp;
    U_MPI_Datatype    etype;
    U_MPI_Datatype    filetype;
    U_MPI_Offset      indiv_fp;
    int             amode;
    pthread_mutex_t fp_lock;
} daos_file_ctx_t;

typedef struct {
    U_MPI_Aint offset;
    U_MPI_Aint length;
} flat_seg_t;

typedef struct {
    flat_seg_t *segs;
    size_t      count;
    size_t      cap;
} flat_list_t;

/* Handle table operations */
U_MPI_File         handle_table_alloc_fh(void);
int              handle_table_insert(U_MPI_File fh, daos_file_ctx_t *ctx);
daos_file_ctx_t *handle_table_lookup(U_MPI_File fh);
daos_file_ctx_t *handle_table_remove(U_MPI_File fh);

/* Datatype flattening & DAOS descriptor builders */
void flat_list_init(flat_list_t *list);
void flat_list_free(flat_list_t *list);
void flatten_datatype(U_MPI_Datatype dtype, int count, U_MPI_Aint base_offset, flat_list_t *out);
void build_daos_sgl(const void *buf, const flat_list_t *mem_flat, d_sg_list_t *sgl);
void build_daos_file_iod(const daos_file_ctx_t *ctx, U_MPI_Offset offset,
                         size_t total_bytes, daos_array_iod_t *iod);

/* Fortran <-> C U_MPI_File handle bridge (avoids OpenMPI internal f2c table) */
U_MPI_Fint mpidaos_file_c2f(U_MPI_File fh);
U_MPI_File mpidaos_file_f2c(U_MPI_Fint f_fh);

/* Public C MPI-I/O entry points implemented in mpi_file_operations.c */
int MPI_File_open(U_MPI_Comm comm, const char *filename, int amode,
                  U_MPI_Info info, U_MPI_File *fh);
int MPI_File_close(U_MPI_File *fh);
int MPI_File_set_view(U_MPI_File fh, U_MPI_Offset disp, U_MPI_Datatype etype,
                      U_MPI_Datatype filetype, const char *datarep, U_MPI_Info info);
int MPI_File_write_at(U_MPI_File fh, U_MPI_Offset offset, const void *buf,
                      int count, U_MPI_Datatype datatype, void *status);
int MPI_File_read_at(U_MPI_File fh, U_MPI_Offset offset, void *buf,
                     int count, U_MPI_Datatype datatype, void *status);
int MPI_File_write_at_all(U_MPI_File fh, U_MPI_Offset offset, const void *buf,
                          int count, U_MPI_Datatype datatype, void *status);
int MPI_File_read_at_all(U_MPI_File fh, U_MPI_Offset offset, void *buf,
                         int count, U_MPI_Datatype datatype, void *status);
int MPI_File_write(U_MPI_File fh, const void *buf, int count,
                   U_MPI_Datatype datatype, void *status);
int MPI_File_read(U_MPI_File fh, void *buf, int count,
                  U_MPI_Datatype datatype, void *status);
int MPI_File_write_shared(U_MPI_File fh, const void *buf, int count,
                          U_MPI_Datatype datatype, void *status);
int MPI_File_read_shared(U_MPI_File fh, void *buf, int count,
                         U_MPI_Datatype datatype, void *status);
int MPI_File_seek_shared(U_MPI_File fh, U_MPI_Offset offset, int whence);
int MPI_File_get_position_shared(U_MPI_File fh, U_MPI_Offset *offset);
int MPI_File_write_ordered(U_MPI_File fh, const void *buf, int count,
                           U_MPI_Datatype datatype, void *status);
int MPI_File_read_ordered(U_MPI_File fh, void *buf, int count,
                          U_MPI_Datatype datatype, void *status);

#endif /* MPIDAOS_INTERNAL_H */
