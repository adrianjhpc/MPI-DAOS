#include "MPI-DAOS.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Initialize / Finalize DAOS runtime alongside MPI */
int MPI_Init(int *argc, char ***argv) {
    int rc = PMPI_Init(argc, argv);
    if (rc == MPI_SUCCESS) {
        daos_init();
    }
    return rc;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    int rc = PMPI_Init_thread(argc, argv, required, provided);
    if (rc == MPI_SUCCESS) {
        daos_init();
    }
    return rc;
}

int MPI_Finalize(void) {
    daos_fini();
    return PMPI_Finalize();
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

/* Broadcast open pool/container handle from Rank 0 to all ranks */
static int bcast_daos_handle(daos_handle_t *hdl, int is_pool, int rank, MPI_Comm comm) {
    d_iov_t glob = {0};
    int rc = 0;

    if (rank == 0) {
        rc = is_pool ? daos_pool_local2global(*hdl, &glob)
                     : daos_cont_local2global(*hdl, &glob);
        if (rc != 0) return rc;
        glob.iov_buf = malloc(glob.iov_buf_len);
        rc = is_pool ? daos_pool_local2global(*hdl, &glob)
                     : daos_cont_local2global(*hdl, &glob);
    }

    uint64_t buf_len = (rank == 0) ? glob.iov_buf_len : 0;
    PMPI_Bcast(&buf_len, 1, MPI_UINT64_T, 0, comm);

    if (rank != 0) {
        glob.iov_buf_len = buf_len;
        glob.iov_len     = buf_len;
        glob.iov_buf     = malloc(buf_len);
    }

    PMPI_Bcast(glob.iov_buf, (int)buf_len, MPI_BYTE, 0, comm);

    if (rank != 0) {
        rc = is_pool ? daos_pool_global2local(glob, hdl)
                     : daos_cont_global2local(glob, hdl);
    }

    free(glob.iov_buf);
    return rc;
}

int MPI_File_open(MPI_Comm comm, const char *filename, int amode,
                  MPI_Info info, MPI_File *fh) {
    const char *pool_str = getenv("DAOS_POOL");
    const char *cont_str = getenv("DAOS_CONT");
    if (!pool_str || !cont_str) {
        return MPI_ERR_BAD_FILE;
    }

    int rank;
    PMPI_Comm_rank(comm, &rank);

    daos_file_ctx_t *ctx = calloc(1, sizeof(daos_file_ctx_t));
    if (!ctx) return MPI_ERR_NO_MEM;

    PMPI_Comm_dup(comm, &ctx->comm);
    ctx->amode    = amode;
    ctx->disp     = 0;
    ctx->indiv_fp = 0;
    ctx->etype    = MPI_BYTE;
    ctx->filetype = MPI_BYTE;
    pthread_mutex_init(&ctx->fp_lock, NULL);

    int rc = 0;
    if (rank == 0) {
        unsigned int flags = (amode & MPI_MODE_RDONLY) ? DAOS_PC_RO : DAOS_PC_RW;
        rc = daos_pool_connect(pool_str, NULL, flags, &ctx->poh, NULL, NULL);
        if (rc == 0) {
            unsigned int cflags = (amode & MPI_MODE_RDONLY) ? DAOS_COO_RO : DAOS_COO_RW;
            rc = daos_cont_open(ctx->poh, cont_str, cflags, &ctx->coh, NULL, NULL);
        }
    }

    PMPI_Bcast(&rc, 1, MPI_INT, 0, ctx->comm);
    if (rc != 0) goto err_free;

    if (bcast_daos_handle(&ctx->poh, 1, rank, ctx->comm) != 0 ||
        bcast_daos_handle(&ctx->coh, 0, rank, ctx->comm) != 0) {
        goto err_free;
    }

    /* Strip leading directory path if present and map filename to DAOS array OID */
    const char *base = strrchr(filename, '/');
    base = base ? base + 1 : filename;
    ctx->oid = filename_to_oid(base);

    if (rank == 0 && (amode & MPI_MODE_CREATE)) {
        /* 1 MiB chunk size, 1-byte cell size */
        rc = daos_array_create(ctx->coh, ctx->oid, DAOS_TX_NONE, 1, 1048576, &ctx->oh, NULL);
        if (rc == -DER_EXIST) {
            rc = daos_array_open(ctx->coh, ctx->oid, DAOS_TX_NONE, DAOS_OO_RW, NULL, NULL, &ctx->oh, NULL);
        }
    }
    PMPI_Barrier(ctx->comm);

    if (rank != 0 || !(amode & MPI_MODE_CREATE)) {
        unsigned int oflags = (amode & MPI_MODE_RDONLY) ? DAOS_OO_RO : DAOS_OO_RW;
        daos_size_t cell_size, chunk_size;
        rc = daos_array_open(ctx->coh, ctx->oid, DAOS_TX_NONE, oflags,
                             &cell_size, &chunk_size, &ctx->oh, NULL);
    }

    if (rc != 0) goto err_free;

    *fh = handle_table_alloc_fh();
    handle_table_insert(*fh, ctx);
    (void)info;
    return MPI_SUCCESS;

err_free:
    PMPI_Comm_free(&ctx->comm);
    free(ctx);
    return MPI_ERR_IO;
}

int MPI_File_close(MPI_File *fh) {
    if (!fh) return MPI_ERR_FILE;
    daos_file_ctx_t *ctx = handle_table_remove(*fh);
    if (!ctx) return MPI_ERR_FILE;

    PMPI_Barrier(ctx->comm);
    daos_array_close(ctx->oh, NULL);
    daos_cont_close(ctx->coh, NULL);
    daos_pool_disconnect(ctx->poh, NULL);

    if (ctx->etype != MPI_BYTE)    PMPI_Type_free(&ctx->etype);
    if (ctx->filetype != MPI_BYTE) PMPI_Type_free(&ctx->filetype);
    PMPI_Comm_free(&ctx->comm);
    pthread_mutex_destroy(&ctx->fp_lock);
    free(ctx);

    *fh = MPI_FILE_NULL;
    return MPI_SUCCESS;
}

int MPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype,
                      MPI_Datatype filetype, const char *datarep, MPI_Info info) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return MPI_ERR_FILE;

    if (ctx->etype != MPI_BYTE)    PMPI_Type_free(&ctx->etype);
    if (ctx->filetype != MPI_BYTE) PMPI_Type_free(&ctx->filetype);

    ctx->disp     = disp;
    ctx->indiv_fp = 0;

    if (etype == MPI_BYTE) {
        ctx->etype = MPI_BYTE;
    } else {
        PMPI_Type_dup(etype, &ctx->etype);
    }

    if (filetype == MPI_BYTE) {
        ctx->filetype = MPI_BYTE;
    } else {
        PMPI_Type_dup(filetype, &ctx->filetype);
    }

    (void)datarep;
    (void)info;
    return MPI_SUCCESS;
}

static int do_daos_io(daos_file_ctx_t *ctx, MPI_Offset offset, void *buf,
                      int count, MPI_Datatype datatype, MPI_Status *status, int is_write) {
    int dtype_size;
    PMPI_Type_size(datatype, &dtype_size);
    size_t total_bytes = (size_t)count * (size_t)dtype_size;
    if (total_bytes == 0) {
        if (status != MPI_STATUS_IGNORE) PMPI_Status_set_elements(status, datatype, 0);
        return MPI_SUCCESS;
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

    if (rc != 0) return MPI_ERR_IO;

    if (status != MPI_STATUS_IGNORE) {
        PMPI_Status_set_elements(status, datatype, count);
    }
    return MPI_SUCCESS;
}

/* Explicit-offset independent & collective I/O */
int MPI_File_write_at(MPI_File fh, MPI_Offset offset, const void *buf,
                      int count, MPI_Datatype datatype, MPI_Status *status) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return MPI_ERR_FILE;
    return do_daos_io(ctx, offset, (void *)buf, count, datatype, status, 1);
}

int MPI_File_read_at(MPI_File fh, MPI_Offset offset, void *buf,
                     int count, MPI_Datatype datatype, MPI_Status *status) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return MPI_ERR_FILE;
    return do_daos_io(ctx, offset, buf, count, datatype, status, 0);
}

int MPI_File_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf,
                          int count, MPI_Datatype datatype, MPI_Status *status) {
    return MPI_File_write_at(fh, offset, buf, count, datatype, status);
}

int MPI_File_read_at_all(MPI_File fh, MPI_Offset offset, void *buf,
                         int count, MPI_Datatype datatype, MPI_Status *status) {
    return MPI_File_read_at(fh, offset, buf, count, datatype, status);
}

/* Individual file-pointer I/O */
int MPI_File_write(MPI_File fh, const void *buf, int count,
                   MPI_Datatype datatype, MPI_Status *status) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return MPI_ERR_FILE;

    int dtype_size, etype_size;
    PMPI_Type_size(datatype, &dtype_size);
    PMPI_Type_size(ctx->etype, &etype_size);
    MPI_Offset etype_advance = ((MPI_Offset)count * dtype_size) / etype_size;

    pthread_mutex_lock(&ctx->fp_lock);
    MPI_Offset off = ctx->indiv_fp;
    ctx->indiv_fp += etype_advance;
    pthread_mutex_unlock(&ctx->fp_lock);

    return do_daos_io(ctx, off, (void *)buf, count, datatype, status, 1);
}

int MPI_File_read(MPI_File fh, void *buf, int count,
                  MPI_Datatype datatype, MPI_Status *status) {
    daos_file_ctx_t *ctx = handle_table_lookup(fh);
    if (!ctx) return MPI_ERR_FILE;

    int dtype_size, etype_size;
    PMPI_Type_size(datatype, &dtype_size);
    PMPI_Type_size(ctx->etype, &etype_size);
    MPI_Offset etype_advance = ((MPI_Offset)count * dtype_size) / etype_size;

    pthread_mutex_lock(&ctx->fp_lock);
    MPI_Offset off = ctx->indiv_fp;
    ctx->indiv_fp += etype_advance;
    pthread_mutex_unlock(&ctx->fp_lock);

    return do_daos_io(ctx, off, buf, count, datatype, status, 0);
}
