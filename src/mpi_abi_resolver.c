#define _GNU_SOURCE
#include "MPI-DAOS_abi.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

mpi_abi_t g_mpi = {0};

static void *resolve_mpi_symbol(const char *sym_name) {
    /* 1. Standard RTLD_NEXT lookup (works when libmpi.so is linked into the app) */
    void *sym = dlsym(RTLD_NEXT, sym_name);
    if (sym) return sym;

    /* 2. Fallback to RTLD_DEFAULT (handles cases where libmpi.so was dlopened) */
    sym = dlsym(RTLD_DEFAULT, sym_name);
    if (sym) return sym;

    fprintf(stderr, "[MPI-DAOS] Fatal: failed to resolve %s\n", sym_name);
    abort();
}

#define RESOLVE_SYM(field, sym_name) do { \
    g_mpi.field = resolve_mpi_symbol(sym_name); \
} while (0)

/* Do NOT use __attribute__((constructor)) — only run when MPI_Init is called */
void mpidaos_resolve_mpi_abi(void) {
    if (g_mpi.Init != NULL) return;

    void *ompi_world = dlsym(RTLD_NEXT, "ompi_mpi_comm_world");
    if (!ompi_world) {
        ompi_world = dlsym(RTLD_DEFAULT, "ompi_mpi_comm_world");
    }

    if (ompi_world != NULL) {
        /* OpenMPI ABI */
        g_mpi.is_openmpi          = 1;
        g_mpi.comm_world          = (U_MPI_Comm)ompi_world;
        g_mpi.dtype_byte          = (U_MPI_Datatype)resolve_mpi_symbol("ompi_mpi_byte");
        g_mpi.dtype_int           = (U_MPI_Datatype)resolve_mpi_symbol("ompi_mpi_int");
        g_mpi.dtype_uint64        = (U_MPI_Datatype)resolve_mpi_symbol("ompi_mpi_uint64_t");
        g_mpi.dtype_int64         = (U_MPI_Datatype)resolve_mpi_symbol("ompi_mpi_int64_t");
        g_mpi.info_null           = (U_MPI_Info)resolve_mpi_symbol("ompi_mpi_info_null");
        g_mpi.file_null           = (U_MPI_File)resolve_mpi_symbol("ompi_mpi_file_null");
        g_mpi.status_ignore       = NULL;
        U_MPI_Fint **f_ign        = (U_MPI_Fint **)resolve_mpi_symbol("MPI_F_STATUS_IGNORE");
        g_mpi.f_status_ignore     = f_ign ? *f_ign : NULL;
        g_mpi.op_sum = (U_MPI_Op)resolve_mpi_symbol("ompi_mpi_op_sum");

        g_mpi.combiner_named      = 0;
        g_mpi.combiner_dup        = 1;
        g_mpi.combiner_contiguous = 2;
        g_mpi.combiner_vector     = 3;
        g_mpi.combiner_hvector    = 4;
        g_mpi.combiner_indexed    = 6;
        g_mpi.combiner_hindexed   = 7;
        g_mpi.combiner_struct     = 10;
        g_mpi.combiner_resized    = 17;
        g_mpi.err_no_mem          = 39;
        g_mpi.err_file            = 30;
        g_mpi.err_bad_file        = 25;
        g_mpi.err_io              = 37;
    } else {
        /* MPICH / Cray MPICH / Intel MPI ABI */
        g_mpi.is_openmpi          = 0;
        g_mpi.comm_world          = 0x44000000;
        g_mpi.dtype_byte          = 0x4c00010d;
        g_mpi.dtype_int           = 0x4c000405;
        g_mpi.dtype_uint64        = 0x4c00081c;
        g_mpi.dtype_int64         = 0x4c00081b;
        g_mpi.info_null           = 0x1c000000;
        g_mpi.file_null           = 0;
        g_mpi.status_ignore       = (void *)1;
        g_mpi.f_status_ignore     = (U_MPI_Fint *)dlsym(RTLD_NEXT, "MPI_F_STATUS_IGNORE");
        g_mpi.op_sum              = 0x58000003;

        g_mpi.combiner_named      = 1;
        g_mpi.combiner_dup        = 2;
        g_mpi.combiner_contiguous = 3;
        g_mpi.combiner_vector     = 4;
        g_mpi.combiner_hvector    = 5;
        g_mpi.combiner_indexed    = 6;
        g_mpi.combiner_hindexed   = 7;
        g_mpi.combiner_struct     = 11;
        g_mpi.combiner_resized    = 18;
        g_mpi.err_no_mem          = 34;
        g_mpi.err_file            = 27;
        g_mpi.err_bad_file        = 20;
        g_mpi.err_io              = 32;
    }

    RESOLVE_SYM(Init,                "PMPI_Init");
    RESOLVE_SYM(Init_thread,         "PMPI_Init_thread");
    RESOLVE_SYM(Finalize,            "PMPI_Finalize");
    RESOLVE_SYM(Comm_rank,           "PMPI_Comm_rank");
    RESOLVE_SYM(Comm_size,           "PMPI_Comm_size");
    RESOLVE_SYM(Comm_dup,            "PMPI_Comm_dup");
    RESOLVE_SYM(Comm_free,           "PMPI_Comm_free");
    RESOLVE_SYM(Barrier,             "PMPI_Barrier");
    RESOLVE_SYM(Bcast,               "PMPI_Bcast");
    RESOLVE_SYM(Exscan,              "PMPI_Exscan");
    RESOLVE_SYM(Type_size,           "PMPI_Type_size");
    RESOLVE_SYM(Type_get_extent,     "PMPI_Type_get_extent");
    RESOLVE_SYM(Type_get_envelope,   "PMPI_Type_get_envelope");
    RESOLVE_SYM(Type_get_contents,   "PMPI_Type_get_contents");
    RESOLVE_SYM(Type_dup,            "PMPI_Type_dup");
    RESOLVE_SYM(Type_free,           "PMPI_Type_free");
    RESOLVE_SYM(Status_set_elements, "PMPI_Status_set_elements");

    RESOLVE_SYM(Comm_f2c,   "PMPI_Comm_f2c");
    RESOLVE_SYM(Type_f2c,   "PMPI_Type_f2c");
    RESOLVE_SYM(Info_f2c,   "PMPI_Info_f2c");
    RESOLVE_SYM(Status_c2f, "PMPI_Status_c2f");
}
