#include "MPI-DAOS.h"
#include <stdlib.h>
#include <string.h>

/* Helper: copy non-null-terminated Fortran string into a C string */
static char *fstr_to_cstr(const char *fstr, int len) {
    while (len > 0 && fstr[len - 1] == ' ') {
        len--; /* Trim trailing Fortran padding spaces */
    }
    char *cstr = malloc((size_t)len + 1);
    if (!cstr) return NULL;
    memcpy(cstr, fstr, (size_t)len);
    cstr[len] = '\0';
    return cstr;
}

/* Helper: populate Fortran status array if not g_mpi.status_ignore */
static inline void update_f_status(void *c_status, U_MPI_Fint *f_status) {
    if (f_status != g_mpi.f_status_ignore) {
        g_mpi.Status_c2f(c_status, f_status);
    }
}

/* ---------------- File Lifecycle & View ---------------- */

void mpi_file_open_(U_MPI_Fint *comm, const char *filename, U_MPI_Fint *amode,
                    U_MPI_Fint *info, U_MPI_Fint *fh, U_MPI_Fint *ierr, int filename_len) {
    char *c_filename = fstr_to_cstr(filename, filename_len);
    if (!c_filename) {
        *ierr = U_MPI_ERR_NO_MEM;
        return;
    }

    U_MPI_Comm c_comm = g_mpi.Comm_f2c(*comm);
    U_MPI_Info c_info = g_mpi.Info_f2c(*info);
    U_MPI_File c_fh   = g_mpi.file_null;

    *ierr = MPI_File_open(c_comm, c_filename, *amode, c_info, &c_fh);
    *fh   = (*ierr == U_MPI_SUCCESS) ? mpidaos_file_c2f(c_fh) : 0;

    free(c_filename);
}

void mpi_file_close_(U_MPI_Fint *fh, U_MPI_Fint *ierr) {
    U_MPI_File c_fh = mpidaos_file_f2c(*fh);
    *ierr = MPI_File_close(&c_fh);
    if (*ierr == U_MPI_SUCCESS) {
        *fh = 0;
    }
}

void mpi_file_set_view_(U_MPI_Fint *fh, U_MPI_Offset *disp, U_MPI_Fint *etype,
                        U_MPI_Fint *filetype, const char *datarep,
                        U_MPI_Fint *info, U_MPI_Fint *ierr, int datarep_len) {
    char *c_datarep = fstr_to_cstr(datarep, datarep_len);
    if (!c_datarep) {
        *ierr = U_MPI_ERR_NO_MEM;
        return;
    }

    U_MPI_File     c_fh       = mpidaos_file_f2c(*fh);
    U_MPI_Datatype c_etype    = g_mpi.Type_f2c(*etype);
    U_MPI_Datatype c_filetype = g_mpi.Type_f2c(*filetype);
    U_MPI_Info     c_info     = g_mpi.Info_f2c(*info);

    *ierr = MPI_File_set_view(c_fh, *disp, c_etype, c_filetype, c_datarep, c_info);
    free(c_datarep);
}

/* ---------------- Explicit-Offset I/O ---------------- */

void mpi_file_write_at_(U_MPI_Fint *fh, U_MPI_Offset *offset, const void *buf,
                        U_MPI_Fint *count, U_MPI_Fint *datatype,
                        U_MPI_Fint *status, U_MPI_Fint *ierr) {
    uint64_t c_status[16] = {0};
    void *st_ptr = (status == g_mpi.f_status_ignore) ? g_mpi.status_ignore : c_status;

    *ierr = MPI_File_write_at(mpidaos_file_f2c(*fh), *offset, buf,
                              *count, g_mpi.Type_f2c(*datatype), st_ptr);
    if (*ierr == U_MPI_SUCCESS) update_f_status(c_status, status);
}

void mpi_file_read_at_(U_MPI_Fint *fh, U_MPI_Offset *offset, void *buf,
                       U_MPI_Fint *count, U_MPI_Fint *datatype,
                       U_MPI_Fint *status, U_MPI_Fint *ierr) {
    uint64_t c_status[16] = {0};
    void *st_ptr = (status == g_mpi.f_status_ignore) ? g_mpi.status_ignore : c_status;

    *ierr = MPI_File_read_at(mpidaos_file_f2c(*fh), *offset, buf,
                             *count, g_mpi.Type_f2c(*datatype), st_ptr);
    if (*ierr == U_MPI_SUCCESS) update_f_status(c_status, status);
}

void mpi_file_write_at_all_(U_MPI_Fint *fh, U_MPI_Offset *offset, const void *buf,
                            U_MPI_Fint *count, U_MPI_Fint *datatype,
                            U_MPI_Fint *status, U_MPI_Fint *ierr) {
    mpi_file_write_at_(fh, offset, buf, count, datatype, status, ierr);
}

void mpi_file_read_at_all_(U_MPI_Fint *fh, U_MPI_Offset *offset, void *buf,
                           U_MPI_Fint *count, U_MPI_Fint *datatype,
                           U_MPI_Fint *status, U_MPI_Fint *ierr) {
    mpi_file_read_at_(fh, offset, buf, count, datatype, status, ierr);
}

/* ---------------- Individual Pointer I/O ---------------- */

void mpi_file_write_(U_MPI_Fint *fh, const void *buf, U_MPI_Fint *count,
                     U_MPI_Fint *datatype, U_MPI_Fint *status, U_MPI_Fint *ierr) {
    uint64_t c_status[16] = {0};
    void *st_ptr = (status == g_mpi.f_status_ignore) ? g_mpi.status_ignore : c_status;

    *ierr = MPI_File_write(mpidaos_file_f2c(*fh), buf,
                           *count, g_mpi.Type_f2c(*datatype), st_ptr);
    if (*ierr == U_MPI_SUCCESS) update_f_status(c_status, status);
}

void mpi_file_read_(U_MPI_Fint *fh, void *buf, U_MPI_Fint *count,
                    U_MPI_Fint *datatype, U_MPI_Fint *status, U_MPI_Fint *ierr) {
    uint64_t c_status[16] = {0};
    void *st_ptr = (status == g_mpi.f_status_ignore) ? g_mpi.status_ignore : c_status;

    *ierr = MPI_File_read(mpidaos_file_f2c(*fh), buf,
                          *count, g_mpi.Type_f2c(*datatype), st_ptr);
    if (*ierr == U_MPI_SUCCESS) update_f_status(c_status, status);
}

/* ---------------- Shared Pointer & Ordered I/O ---------------- */

void mpi_file_write_shared_(U_MPI_Fint *fh, const void *buf, U_MPI_Fint *count,
                            U_MPI_Fint *datatype, U_MPI_Fint *status, U_MPI_Fint *ierr) {
    uint64_t c_status[16] = {0};
    void *st_ptr = (status == g_mpi.f_status_ignore) ? g_mpi.status_ignore : c_status;

    *ierr = MPI_File_write_shared(mpidaos_file_f2c(*fh), buf,
                                  *count, g_mpi.Type_f2c(*datatype), st_ptr);
    if (*ierr == U_MPI_SUCCESS) update_f_status(c_status, status);
}

void mpi_file_read_shared_(U_MPI_Fint *fh, void *buf, U_MPI_Fint *count,
                           U_MPI_Fint *datatype, U_MPI_Fint *status, U_MPI_Fint *ierr) {
    uint64_t c_status[16] = {0};
    void *st_ptr = (status == g_mpi.f_status_ignore) ? g_mpi.status_ignore : c_status;

    *ierr = MPI_File_read_shared(mpidaos_file_f2c(*fh), buf,
                                 *count, g_mpi.Type_f2c(*datatype), st_ptr);
    if (*ierr == U_MPI_SUCCESS) update_f_status(c_status, status);
}

void mpi_file_seek_shared_(U_MPI_Fint *fh, U_MPI_Offset *offset,
                           U_MPI_Fint *whence, U_MPI_Fint *ierr) {
    *ierr = MPI_File_seek_shared(mpidaos_file_f2c(*fh), *offset, *whence);
}

void mpi_file_get_position_shared_(U_MPI_Fint *fh, U_MPI_Offset *offset, U_MPI_Fint *ierr) {
    *ierr = MPI_File_get_position_shared(mpidaos_file_f2c(*fh), offset);
}

void mpi_file_write_ordered_(U_MPI_Fint *fh, const void *buf, U_MPI_Fint *count,
                             U_MPI_Fint *datatype, U_MPI_Fint *status, U_MPI_Fint *ierr) {
    uint64_t c_status[16] = {0};
    void *st_ptr = (status == g_mpi.f_status_ignore) ? g_mpi.status_ignore : c_status;

    *ierr = MPI_File_write_ordered(mpidaos_file_f2c(*fh), buf,
                                   *count, g_mpi.Type_f2c(*datatype), st_ptr);
    if (*ierr == U_MPI_SUCCESS) update_f_status(c_status, status);
}

void mpi_file_read_ordered_(U_MPI_Fint *fh, void *buf, U_MPI_Fint *count,
                            U_MPI_Fint *datatype, U_MPI_Fint *status, U_MPI_Fint *ierr) {
    uint64_t c_status[16] = {0};
    void *st_ptr = (status == g_mpi.f_status_ignore) ? g_mpi.status_ignore : c_status;

    *ierr = MPI_File_read_ordered(mpidaos_file_f2c(*fh), buf,
                                  *count, g_mpi.Type_f2c(*datatype), st_ptr);
    if (*ierr == U_MPI_SUCCESS) update_f_status(c_status, status);
}

/* ---------------- OpenMPI mpi_f08 Specific Suffixes ---------------- */
/* OpenMPI's mpi_f08 module sometimes binds to *_f08 symbols instead of *_ */
void mpi_file_open_f08_(U_MPI_Fint *comm, const char *filename, U_MPI_Fint *amode,
                        U_MPI_Fint *info, U_MPI_Fint *fh, U_MPI_Fint *ierr, int filename_len) {
    mpi_file_open_(comm, filename, amode, info, fh, ierr, filename_len);
}

void mpi_file_close_f08_(U_MPI_Fint *fh, U_MPI_Fint *ierr) {
    mpi_file_close_(fh, ierr);
}

void mpi_file_set_view_f08_(U_MPI_Fint *fh, U_MPI_Offset *disp, U_MPI_Fint *etype,
                            U_MPI_Fint *filetype, const char *datarep,
                            U_MPI_Fint *info, U_MPI_Fint *ierr, int datarep_len) {
    mpi_file_set_view_(fh, disp, etype, filetype, datarep, info, ierr, datarep_len);
}

void mpi_file_write_at_f08_(U_MPI_Fint *fh, U_MPI_Offset *offset, const void *buf,
                            U_MPI_Fint *count, U_MPI_Fint *datatype,
                            U_MPI_Fint *status, U_MPI_Fint *ierr) {
    mpi_file_write_at_(fh, offset, buf, count, datatype, status, ierr);
}

void mpi_file_read_at_f08_(U_MPI_Fint *fh, U_MPI_Offset *offset, void *buf,
                           U_MPI_Fint *count, U_MPI_Fint *datatype,
                           U_MPI_Fint *status, U_MPI_Fint *ierr) {
    mpi_file_read_at_(fh, offset, buf, count, datatype, status, ierr);
}

void mpi_file_write_shared_f08_(U_MPI_Fint *fh, const void *buf, U_MPI_Fint *count,
                                U_MPI_Fint *datatype, U_MPI_Fint *status, U_MPI_Fint *ierr) {
    mpi_file_write_shared_(fh, buf, count, datatype, status, ierr);
}

void mpi_file_read_shared_f08_(U_MPI_Fint *fh, void *buf, U_MPI_Fint *count,
                               U_MPI_Fint *datatype, U_MPI_Fint *status, U_MPI_Fint *ierr) {
    mpi_file_read_shared_(fh, buf, count, datatype, status, ierr);
}

void mpi_file_seek_shared_f08_(U_MPI_Fint *fh, U_MPI_Offset *offset,
                               U_MPI_Fint *whence, U_MPI_Fint *ierr) {
    mpi_file_seek_shared_(fh, offset, whence, ierr);
}

void mpi_file_write_ordered_f08_(U_MPI_Fint *fh, const void *buf, U_MPI_Fint *count,
                                 U_MPI_Fint *datatype, U_MPI_Fint *status, U_MPI_Fint *ierr) {
    mpi_file_write_ordered_(fh, buf, count, datatype, status, ierr);
}

void mpi_file_read_ordered_f08_(U_MPI_Fint *fh, void *buf, U_MPI_Fint *count,
                                U_MPI_Fint *datatype, U_MPI_Fint *status, U_MPI_Fint *ierr) {
    mpi_file_read_ordered_(fh, buf, count, datatype, status, ierr);
}
