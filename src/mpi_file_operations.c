#include "MPI-DAOS.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Initialize / Finalize DAOS runtime alongside MPI */
int MPI_Init(int *argc, char ***argv) {
    mpidaos_resolve_mpi_abi();
    int rc = g_mpi.Init(argc, argv);
    if (rc == U_MPI_SUCCESS) {
        daos_init();
    }
    return rc;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    mpidaos_resolve_mpi_abi();
    int rc = g_mpi.Init_thread(argc, argv, required, provided);
    if (rc == U_MPI_SUCCESS) {
        daos_init();
    }
    return rc;
}

int MPI_Finalize(void) {
    daos_fini();
    return g_mpi.Finalize();
}

#define SHARED_FP_KEY "shared_fp"

/* Derive paired KV metadata OID (OC_S1 routes all updates to one target) */
static daos_obj_id_t filename_to_kv_oid(const char *filename) {
    uint64_t h1 = 14695981039346656037ULL ^ 0x5A5A5A5A5A5A5A5AULL;
    uint64_t h2 = 1099511628211ULL ^ 0xA5A5A5A5A5A5A5A5ULL;
    for (const char *p = filename; *p; p++) {
        h1 ^= (uint8_t)(*p);
        h1 *= 1099511628211ULL;
        h2 = (h2 << 5) + h2 + (uint8_t)(*p);
    }
    daos_obj_id_t oid = { .lo = h1, .hi = h2 };
    daos_obj_generate_oid(DAOS_HDL_INVAL, &oid, DAOS_OT_KV_HASHED, OC_S1, 0, 0);
    return oid;
}

/* Helper: derive deterministic 128-bit DAOS OID from filename */
static daos_obj_id_t filename_to_oid(const char *filename) {
    uint64_t h1 = 14695981039346656037ULL;
    uint64_t h2 = 1099511628211ULL;
    for (const char *p = filename; *p; p++) {
        h1 ^= (uint8_t)(*p);
        h1 *= 1099511628211ULL;
        h2 = (h2 << 5) + h2 + (uint8_t)(*p);
    }
    daos_obj_id_t oid = { .lo = h1, .hi = h2 };
    daos_array_generate_oid(DAOS_HDL_INVAL, &oid, true, OC_SX, 0, 0);
    return oid;
}

/* Broadcast open pool/container handle from Rank 0 to all ranks.
 * Pass DAOS_HDL_INVAL for poh when broadcasting the pool handle itself. */
static int bcast_daos_handle(daos_handle_t poh, daos_handle_t *hdl,
                             int is_pool, int rank, U_MPI_Comm comm) {
    d_iov_t glob = {0};
    int rc = 0;

    if (rank == 0) {
        rc = is_pool ? daos_pool_local2global(*hdl, &glob)
                     : daos_cont_local2global(*hdl, &glob);
        if (rc != 0) return rc;

        glob.iov_buf = malloc(glob.iov_buf_len);
        if (!glob.iov_buf) return -DER_NOMEM;

        rc = is_pool ? daos_pool_local2global(*hdl, &glob)
                     : daos_cont_local2global(*hdl, &glob);
        if (rc != 0) {
            free(glob.iov_buf);
            return rc;
        }
    }

    uint64_t buf_len = (rank == 0) ? glob.iov_buf_len : 0;
    g_mpi.Bcast(&buf_len, 1, g_mpi.dtype_uint64, 0, comm);

    if (rank != 0) {
        glob.iov_buf_len = buf_len;
        glob.iov_len     = buf_len;
        glob.iov_buf     = malloc(buf_len);
        if (!glob.iov_buf) return -DER_NOMEM;
    }

    g_mpi.Bcast(glob.iov_buf, (int)buf_len, g_mpi.dtype_byte, 0, comm);

    if (rank != 0) {
        rc = is_pool ? daos_pool_global2local(glob, hdl)
                     : daos_cont_global2local(poh, glob, hdl);
    }

    free(glob.iov_buf);
    return rc;
}
int MPI_File_open(U_MPI_Comm comm, const char *filename, int amode,
                  U_MPI_Info info, U_MPI_File *fh) {
    const char *pool_str = getenv("DAOS_POOL");
    const char *cont_str = getenv("DAOS_CONT");
    if (!pool_str || !cont_str) {
        return U_MPI_ERR_BAD_FILE;
    }

    int rank;
    g_mpi.Comm_rank(comm, &rank);

    daos_file_ctx_t *ctx = calloc(1, sizeof(daos_file_ctx_t));
    if (!ctx) return U_MPI_ERR_NO_MEM;

    g_mpi.Comm_dup(comm, &ctx->comm);
    ctx->amode    = amode;
    ctx->disp     = 0;
    ctx->indiv_fp = 0;
    ctx->etype    = g_mpi.dtype_byte;
    ctx->filetype = g_mpi.dtype_byte;
    pthread_mutex_init(&ctx->fp_lock, NULL);

    int rc = 0;
    if (rank == 0) {
        unsigned int flags = (amode & U_MPI_MODE_RDONLY) ? DAOS_PC_RO : DAOS_PC_RW;
        rc = daos_pool_connect(pool_str, NULL, flags, &ctx->poh, NULL, NULL);
        if (rc == 0) {
            unsigned int cflags = (amode & U_MPI_MODE_RDONLY) ? DAOS_COO_RO : DAOS_COO_RW;
            rc = daos_cont_open(ctx->poh, cont_str, cflags, &ctx->coh, NULL, NULL);
        }
    }

    g_mpi.Bcast(&rc, 1, g_mpi.dtype_int, 0, ctx->comm);
    if (rc != 0) goto err_free;

    if (bcast_daos_handle(DAOS_HDL_INVAL, &ctx->poh, 1, rank, ctx->comm) != 0 ||
        bcast_daos_handle(ctx->poh,       &ctx->coh, 0, rank, ctx->comm) != 0) {
        goto err_free;
    }    

    /* Strip leading directory path if present and map filename to DAOS array OID */
    const char *base = strrchr(filename, '/');
    base = base ? base + 1 : filename;
    ctx->oid = filename_to_oid(base);

    if (rank == 0 && (amode & U_MPI_MODE_CREATE)) {
        /* 1 MiB chunk size, 1-byte cell size */
        rc = daos_array_create(ctx->coh, ctx->oid, DAOS_TX_NONE, 1, 1048576, &ctx->oh, NULL);
        if (rc == -DER_EXIST) {
            rc = daos_array_open(ctx->coh, ctx->oid, DAOS_TX_NONE, DAOS_OO_RW, NULL, NULL, &ctx->oh, NULL);
        }
    }
    g_mpi.Barrier(ctx->comm);

    if (rank != 0 || !(amode & U_MPI_MODE_CREATE)) {
        unsigned int oflags = (amode & U_MPI_MODE_RDONLY) ? DAOS_OO_RO : DAOS_OO_RW;
        daos_size_t cell_size, chunk_size;
        rc = daos_array_open(ctx->coh, ctx->oid, DAOS_TX_NONE, oflags,
                             &cell_size, &chunk_size, &ctx->oh, NULL);
    }

    /* Open the dedicated KV metadata object for shared file pointer state */
    ctx->kv_oid = filename_to_kv_oid(base);
    unsigned int kv_flags = (amode & U_MPI_MODE_RDONLY) ? DAOS_OO_RO : DAOS_OO_RW;
    rc = daos_kv_open(ctx->coh, ctx->kv_oid, kv_flags, &ctx->kv_oh, NULL);
    if (rc != 0) goto err_free;

    if (rank == 0 && (amode & U_MPI_MODE_CREATE)) {
        int64_t init_fp = 0;
        rc = daos_kv_put(ctx->kv_oh, DAOS_TX_NONE, 0, SHARED_FP_KEY,
                         sizeof(init_fp), &init_fp, NULL);
        if (rc != 0) goto err_free;
    }
    g_mpi.Barrier(ctx->comm);

    if (rc != 0) goto err_free;

    *fh = handle_table_alloc_fh();
    handle_table_insert(*fh, ctx);
    (void)info;
    return U_MPI_SUCCESS;

err_free:
    g_mpi.Comm_free(&ctx->comm);
    free(ctx);
    return U_MPI_ERR_IO;
}

int MPI_File_close(U_MPI_File *fh) {
    if (!fh) return U_MPI_ERR_FILE;
    daos_file_ctx_t *ctx = handle_table_remove(*fh);
    if (!ctx) return U_MPI_ERR_FILE;

    g_mpi.Barrier(ctx->comm);
    daos_kv_close(ctx->kv_oh, NULL);
    daos_array_close(ctx->oh, NULL);
    daos_cont_close(ctx->coh, NULL);
    daos_pool_disconnect(ctx->poh, NULL);

    if (ctx->etype != g_mpi.dtype_byte)    g_mpi.Type_free(&ctx->etype);
    if (ctx->filetype != g_mpi.dtype_byte) g_mpi.Type_free(&ctx->filetype);
    g_mpi.Comm_free(&ctx->comm);
    pthread_mutex_destroy(&ctx->fp_lock);
    free(ctx);

    *fh = g_mpi.file_null;
    return U_MPI_SUCCESS;
}

int MPI_File_set_view(U_MPI_File fh, U_MPI_Offset disp, U_MPI_Datatype etype,
                      U_MPI_Datatype filetype, const char *datarep, U_MPI_Info info) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return U_MPI_ERR_FILE;

    if (ctx->etype != g_mpi.dtype_byte)    g_mpi.Type_free(&ctx->etype);
    if (ctx->filetype != g_mpi.dtype_byte) g_mpi.Type_free(&ctx->filetype);

    ctx->disp     = disp;
    ctx->indiv_fp = 0;

    if (etype == g_mpi.dtype_byte) {
        ctx->etype = g_mpi.dtype_byte;
    } else {
        g_mpi.Type_dup(etype, &ctx->etype);
    }

    if (filetype == g_mpi.dtype_byte) {
        ctx->filetype = g_mpi.dtype_byte;
    } else {
        g_mpi.Type_dup(filetype, &ctx->filetype);
    }

    (void)datarep;
    (void)info;
    return U_MPI_SUCCESS;
}

static int do_daos_io(daos_file_ctx_t *ctx, U_MPI_Offset offset, void *buf,
                      int count, U_MPI_Datatype datatype, void *status, int is_write) {
    int dtype_size;
    g_mpi.Type_size(datatype, &dtype_size);
    size_t total_bytes = (size_t)count * (size_t)dtype_size;
    if (total_bytes == 0) {
        if (status != g_mpi.status_ignore) g_mpi.Status_set_elements(status, datatype, 0);
        return U_MPI_SUCCESS;
    }

    flat_list_t mem_flat;
    flat_list_init(&mem_flat);
    flatten_datatype(datatype, count, 0, &mem_flat);

    d_sg_list_t sgl;
    build_daos_sgl(buf, &mem_flat, &sgl);

    daos_array_iod_t iod;
    build_daos_file_iod(ctx, offset, total_bytes, &iod);

    int rc = is_write ? daos_array_write(ctx->oh, DAOS_TX_NONE, &iod, &sgl, NULL)
                      : daos_array_read(ctx->oh, DAOS_TX_NONE, &iod, &sgl, NULL);

    flat_list_free(&mem_flat);
    free(sgl.sg_iovs);
    free(iod.arr_rgs);

    if (rc != 0) return U_MPI_ERR_IO;

    if (status != g_mpi.status_ignore) {
        g_mpi.Status_set_elements(status, datatype, count);
    }
    return U_MPI_SUCCESS;
}

/* Explicit-offset independent & collective I/O */
int MPI_File_write_at(U_MPI_File fh, U_MPI_Offset offset, const void *buf,
                      int count, U_MPI_Datatype datatype, void *status) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return U_MPI_ERR_FILE;
    return do_daos_io(ctx, offset, (void *)buf, count, datatype, status, 1);
}

int MPI_File_read_at(U_MPI_File fh, U_MPI_Offset offset, void *buf,
                     int count, U_MPI_Datatype datatype, void *status) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return U_MPI_ERR_FILE;
    return do_daos_io(ctx, offset, buf, count, datatype, status, 0);
}

int MPI_File_write_at_all(U_MPI_File fh, U_MPI_Offset offset, const void *buf,
                          int count, U_MPI_Datatype datatype, void *status) {
    return MPI_File_write_at(fh, offset, buf, count, datatype, status);
}

int MPI_File_read_at_all(U_MPI_File fh, U_MPI_Offset offset, void *buf,
                         int count, U_MPI_Datatype datatype, void *status) {
    return MPI_File_read_at(fh, offset, buf, count, datatype, status);
}

/* Individual file-pointer I/O */
int MPI_File_write(U_MPI_File fh, const void *buf, int count,
                   U_MPI_Datatype datatype, void *status) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return U_MPI_ERR_FILE;

    int dtype_size, etype_size;
    g_mpi.Type_size(datatype, &dtype_size);
    g_mpi.Type_size(ctx->etype, &etype_size);
    U_MPI_Offset etype_advance = ((U_MPI_Offset)count * dtype_size) / etype_size;

    pthread_mutex_lock(&ctx->fp_lock);
    U_MPI_Offset off = ctx->indiv_fp;
    ctx->indiv_fp += etype_advance;
    pthread_mutex_unlock(&ctx->fp_lock);

    return do_daos_io(ctx, off, (void *)buf, count, datatype, status, 1);
}

int MPI_File_read(U_MPI_File fh, void *buf, int count,
                  U_MPI_Datatype datatype, void *status) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return U_MPI_ERR_FILE;

    int dtype_size, etype_size;
    g_mpi.Type_size(datatype, &dtype_size);
    g_mpi.Type_size(ctx->etype, &etype_size);
    U_MPI_Offset etype_advance = ((U_MPI_Offset)count * dtype_size) / etype_size;

    pthread_mutex_lock(&ctx->fp_lock);
    U_MPI_Offset off = ctx->indiv_fp;
    ctx->indiv_fp += etype_advance;
    pthread_mutex_unlock(&ctx->fp_lock);

    return do_daos_io(ctx, off, buf, count, datatype, status, 0);
}

/* Atomically fetch-and-add 'delta' (in etype units) to the DAOS KV shared pointer.
 * Returns the previous offset in *old_fp. */
static int kv_shared_fp_fetch_add(daos_file_ctx_t *ctx, int64_t delta, int64_t *old_fp) {
    daos_handle_t th;
    int rc = daos_tx_open(ctx->coh, &th, 0, NULL);
    if (rc != 0) return rc;

    while (1) {
        int64_t cur_fp = 0;
        daos_size_t val_size = sizeof(cur_fp);

        rc = daos_kv_get(ctx->kv_oh, th, 0, SHARED_FP_KEY, &val_size, &cur_fp, NULL);
        if (rc == -DER_NONEXIST) {
            cur_fp = 0;
            rc = 0;
        }
        if (rc != 0) break;

        if (old_fp) *old_fp = cur_fp;

        if (delta != 0) {
            int64_t new_fp = cur_fp + delta;
            rc = daos_kv_put(ctx->kv_oh, th, 0, SHARED_FP_KEY,
                             sizeof(new_fp), &new_fp, NULL);
            if (rc != 0) break;
        }

        rc = daos_tx_commit(th, NULL);
        if (rc == 0) break;

        if (rc == -DER_TX_RESTART) {
            /* Another rank updated the shared pointer concurrently; retry */
            rc = daos_tx_restart(th, NULL);
            if (rc != 0) break;
            continue;
        }
        break;
    }

    daos_tx_close(th, NULL);
    return rc;
}

/* Atomically set the shared pointer to an absolute offset (in etype units) */
static int kv_shared_fp_set(daos_file_ctx_t *ctx, int64_t new_fp) {
    int rc = daos_kv_put(ctx->kv_oh, DAOS_TX_NONE, 0, SHARED_FP_KEY,
                         sizeof(new_fp), &new_fp, NULL);
    return rc;
}

/* Independent shared-pointer I/O */
int MPI_File_write_shared(U_MPI_File fh, const void *buf, int count,
                          U_MPI_Datatype datatype, void *status) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return U_MPI_ERR_FILE;

    int dtype_size, etype_size;
    g_mpi.Type_size(datatype, &dtype_size);
    g_mpi.Type_size(ctx->etype, &etype_size);
    int64_t etype_advance = ((int64_t)count * dtype_size) / etype_size;

    int64_t target_fp = 0;
    if (kv_shared_fp_fetch_add(ctx, etype_advance, &target_fp) != 0) {
        return U_MPI_ERR_IO;
    }

    return do_daos_io(ctx, (U_MPI_Offset)target_fp, (void *)buf, count, datatype, status, 1);
}

int MPI_File_read_shared(U_MPI_File fh, void *buf, int count,
                         U_MPI_Datatype datatype, void *status) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return U_MPI_ERR_FILE;

    int dtype_size, etype_size;
    g_mpi.Type_size(datatype, &dtype_size);
    g_mpi.Type_size(ctx->etype, &etype_size);
    int64_t etype_advance = ((int64_t)count * dtype_size) / etype_size;

    int64_t target_fp = 0;
    if (kv_shared_fp_fetch_add(ctx, etype_advance, &target_fp) != 0) {
        return U_MPI_ERR_IO;
    }

    return do_daos_io(ctx, (U_MPI_Offset)target_fp, buf, count, datatype, status, 0);
}

/* Collective shared-pointer query and seek */
int MPI_File_get_position_shared(U_MPI_File fh, U_MPI_Offset *offset) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return U_MPI_ERR_FILE;

    int64_t cur_fp = 0;
    if (kv_shared_fp_fetch_add(ctx, 0, &cur_fp) != 0) {
        return U_MPI_ERR_IO;
    }
    *offset = (U_MPI_Offset)cur_fp;
    return U_MPI_SUCCESS;
}

int MPI_File_seek_shared(U_MPI_File fh, U_MPI_Offset offset, int whence) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return U_MPI_ERR_FILE;

    int rank;
    g_mpi.Comm_rank(ctx->comm, &rank);

    int rc = 0;
    if (rank == 0) {
        int64_t new_fp = 0;
        if (whence == U_MPI_SEEK_SET) {
            new_fp = (int64_t)offset;
        } else if (whence == U_MPI_SEEK_CUR) {
            int64_t cur_fp = 0;
            rc = kv_shared_fp_fetch_add(ctx, 0, &cur_fp);
            new_fp = cur_fp + (int64_t)offset;
        } else if (whence == U_MPI_SEEK_END) {
            daos_size_t byte_size = 0;
            int etype_size;
            g_mpi.Type_size(ctx->etype, &etype_size);
            rc = daos_array_get_size(ctx->oh, DAOS_TX_NONE, &byte_size, NULL);
            new_fp = (int64_t)((byte_size - ctx->disp) / etype_size) + (int64_t)offset;
        } else {
            rc = -DER_INVAL;
        }

        if (rc == 0 && new_fp < 0) rc = -DER_INVAL;
        if (rc == 0) rc = kv_shared_fp_set(ctx, new_fp);
    }

    g_mpi.Bcast(&rc, 1, g_mpi.dtype_int, 0, ctx->comm);
    return (rc == 0) ? U_MPI_SUCCESS : U_MPI_ERR_IO;
}

/* Collective ordered I/O: uses MPI_Exscan to avoid N-way DAOS transaction contention */
static int do_daos_ordered_io(U_MPI_File fh, void *buf, int count,
                              U_MPI_Datatype datatype, void *status, int is_write) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return U_MPI_ERR_FILE;

    int rank, size;
    g_mpi.Comm_rank(ctx->comm, &rank);
    g_mpi.Comm_size(ctx->comm, &size);

    int dtype_size, etype_size;
    g_mpi.Type_size(datatype, &dtype_size);
    g_mpi.Type_size(ctx->etype, &etype_size);

    int64_t my_etypes = ((int64_t)count * dtype_size) / etype_size;
    int64_t prefix_etypes = 0;
    g_mpi.Exscan(&my_etypes, &prefix_etypes, 1, g_mpi.dtype_int64, g_mpi.op_sum, ctx->comm);
    if (rank == 0) prefix_etypes = 0;

    int64_t base_fp = 0;
    if (rank == 0) {
        if (kv_shared_fp_fetch_add(ctx, 0, &base_fp) != 0) base_fp = -1;
    }
    g_mpi.Bcast(&base_fp, 1, g_mpi.dtype_int64, 0, ctx->comm);
    if (base_fp < 0) return U_MPI_ERR_IO;

    /* Last rank holds the total etype count across all ranks; update shared pointer once */
    if (rank == size - 1) {
        int64_t final_fp = base_fp + prefix_etypes + my_etypes;
        if (kv_shared_fp_set(ctx, final_fp) != 0) return U_MPI_ERR_IO;
    }

    int rc = do_daos_io(ctx, (U_MPI_Offset)(base_fp + prefix_etypes),
                        buf, count, datatype, status, is_write);
    g_mpi.Barrier(ctx->comm);
    return rc;
}

int MPI_File_write_ordered(U_MPI_File fh, const void *buf, int count,
                           U_MPI_Datatype datatype, void *status) {
    return do_daos_ordered_io(fh, (void *)buf, count, datatype, status, 1);
}

int MPI_File_read_ordered(U_MPI_File fh, void *buf, int count,
                          U_MPI_Datatype datatype, void *status) {
    return do_daos_ordered_io(fh, buf, count, datatype, status, 0);
}
