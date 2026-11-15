#ifndef MPIDAOS_INTERNAL_H
#define MPIDAOS_INTERNAL_H

#include <mpi.h>
#include <daos.h>
#include <daos_array.h>
#include <pthread.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    daos_handle_t   poh;
    daos_handle_t   coh;
    daos_handle_t   oh;
    daos_obj_id_t   oid;
    MPI_Comm        comm;
    MPI_Offset      disp;
    MPI_Datatype    etype;
    MPI_Datatype    filetype;
    MPI_Offset      indiv_fp;
    int             amode;
    pthread_mutex_t fp_lock;
} daos_file_ctx_t;

typedef struct {
    MPI_Aint offset;
    MPI_Aint length;
} flat_seg_t;

typedef struct {
    flat_seg_t *segs;
    size_t      count;
    size_t      cap;
} flat_list_t;

/* Handle table operations */
MPI_File         handle_table_alloc_fh(void);
int              handle_table_insert(MPI_File fh, daos_file_ctx_t *ctx);
daos_file_ctx_t *handle_table_lookup(MPI_File fh);
daos_file_ctx_t *handle_table_remove(MPI_File fh);

/* Datatype flattening & DAOS descriptor builders */
void flat_list_init(flat_list_t *list);
void flat_list_free(flat_list_t *list);
void flatten_datatype(MPI_Datatype dtype, int count, MPI_Aint base_offset, flat_list_t *out);
void build_daos_sgl(const void *buf, const flat_list_t *mem_flat, d_sg_list_t *sgl);
void build_daos_file_iod(const daos_file_ctx_t *ctx, MPI_Offset offset,
                         size_t total_bytes, daos_array_iod_t *iod);

#endif /* MPIDAOS_INTERNAL_H */
