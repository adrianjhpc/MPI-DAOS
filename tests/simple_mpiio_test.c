#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define CHECK_MPI(call) do {                                        \
    int _err = (call);                                              \
    if (_err != MPI_SUCCESS) {                                      \
        char _errstr[MPI_MAX_ERROR_STRING];                         \
        int _len;                                                   \
        MPI_Error_string(_err, _errstr, &_len);                     \
        fprintf(stderr, "Rank %d failed at %s:%d: %s\n",            \
                rank, __FILE__, __LINE__, _errstr);                 \
        MPI_Abort(MPI_COMM_WORLD, _err);                            \
    }                                                               \
} while (0)

/* Test 1: Explicit-offset contiguous read/write */
static void test_explicit_offset(int rank, int size) {
    MPI_File fh;
    const int count = 16;
    int *wbuf = malloc(count * sizeof(int));
    int *rbuf = calloc(count, sizeof(int));

    for (int i = 0; i < count; i++) {
        wbuf[i] = (rank * 1000) + i;
    }

    CHECK_MPI(MPI_File_open(MPI_COMM_WORLD, "test_contig.dat",
                            MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh));

    MPI_Offset offset = (MPI_Offset)rank * count * (MPI_Offset)sizeof(int);
    CHECK_MPI(MPI_File_write_at(fh, offset, wbuf, count, MPI_INT, MPI_STATUS_IGNORE));
    MPI_Barrier(MPI_COMM_WORLD);

    /* Read back neighbor rank's data to verify cross-rank visibility */
    int target_rank = (rank + 1) % size;
    MPI_Offset read_offset = (MPI_Offset)target_rank * count * (MPI_Offset)sizeof(int);
    CHECK_MPI(MPI_File_read_at(fh, read_offset, rbuf, count, MPI_INT, MPI_STATUS_IGNORE));

    for (int i = 0; i < count; i++) {
        int expected = (target_rank * 1000) + i;
        if (rbuf[i] != expected) {
            fprintf(stderr, "[Test 1 FAIL] Rank %d index %d: got %d, expected %d\n",
                    rank, i, rbuf[i], expected);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    CHECK_MPI(MPI_File_close(&fh));
    free(wbuf);
    free(rbuf);
    if (rank == 0) printf("[PASS] Test 1: Explicit-offset contiguous read/write\n");
}

/* Test 2: Non-contiguous derived datatype + strided file view */
static void test_derived_view(int rank, int size) {
    MPI_File fh;
    /* Each rank owns a strided pattern in the file: 1 int every 'size' ints */
    MPI_Datatype filetype;
    MPI_Type_vector(4, 1, size, MPI_INT, &filetype);
    MPI_Type_commit(&filetype);

    /* Non-contiguous memory buffer: 2 blocks of 2 ints with stride 4 */
    MPI_Datatype memtype;
    MPI_Type_vector(2, 2, 4, MPI_INT, &memtype);
    MPI_Type_commit(&memtype);

    int wbuf[8] = {0};
    int rbuf[8] = {0};
    /* Populate the active strided slots in wbuf: indices 0,1 and 4,5 */
    wbuf[0] = rank * 10 + 1;
    wbuf[1] = rank * 10 + 2;
    wbuf[4] = rank * 10 + 3;
    wbuf[5] = rank * 10 + 4;

    CHECK_MPI(MPI_File_open(MPI_COMM_WORLD, "test_view.dat",
                            MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh));

    MPI_Offset disp = (MPI_Offset)rank * (MPI_Offset)sizeof(int);
    CHECK_MPI(MPI_File_set_view(fh, disp, MPI_INT, filetype, "native", MPI_INFO_NULL));

    /* Write 1 memtype (4 MPI_INTs total) through the strided file view */
    CHECK_MPI(MPI_File_write_at(fh, 0, wbuf, 1, memtype, MPI_STATUS_IGNORE));
    MPI_Barrier(MPI_COMM_WORLD);

    /* Read back into strided rbuf */
    CHECK_MPI(MPI_File_read_at(fh, 0, rbuf, 1, memtype, MPI_STATUS_IGNORE));

    int check_indices[4] = {0, 1, 4, 5};
    for (int k = 0; k < 4; k++) {
        int idx = check_indices[k];
        if (rbuf[idx] != wbuf[idx]) {
            fprintf(stderr, "[Test 2 FAIL] Rank %d slot %d: got %d, expected %d\n",
                    rank, idx, rbuf[idx], wbuf[idx]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    CHECK_MPI(MPI_File_close(&fh));
    MPI_Type_free(&memtype);
    MPI_Type_free(&filetype);
    if (rank == 0) printf("[PASS] Test 2: Non-contiguous derived datatype & file view\n");
}

/* Test 3: Shared file pointers (atomic KV metadata object) & Ordered collectives */
static void test_shared_and_ordered(int rank, int size) {
    MPI_File fh;
    CHECK_MPI(MPI_File_open(MPI_COMM_WORLD, "test_shared.dat",
                            MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh));

    /* 3a. Concurrent uncoordinated shared writes (tests DAOS TX retry loop) */
    int my_val = 500 + rank;
    CHECK_MPI(MPI_File_write_shared(fh, &my_val, 1, MPI_INT, MPI_STATUS_IGNORE));
    MPI_Barrier(MPI_COMM_WORLD);

    /* Rewind shared pointer to 0 and read all values back on Rank 0 */
    CHECK_MPI(MPI_File_seek_shared(fh, 0, MPI_SEEK_SET));
    if (rank == 0) {
        int *all_vals = calloc(size, sizeof(int));
        int *seen = calloc(size, sizeof(int));
        CHECK_MPI(MPI_File_read_at(fh, 0, all_vals, size, MPI_INT, MPI_STATUS_IGNORE));

        for (int i = 0; i < size; i++) {
            int r = all_vals[i] - 500;
            if (r < 0 || r >= size || seen[r]) {
                fprintf(stderr, "[Test 3a FAIL] Duplicate or corrupt entry %d at slot %d\n",
                        all_vals[i], i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            seen[r] = 1;
        }
        free(all_vals);
        free(seen);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* 3b. Collective ordered write & read (tests MPI_Exscan path) */
    CHECK_MPI(MPI_File_seek_shared(fh, 0, MPI_SEEK_SET));
    int ordered_w[2] = { rank * 2, rank * 2 + 1 };
    int ordered_r[2] = { -1, -1 };

    CHECK_MPI(MPI_File_write_ordered(fh, ordered_w, 2, MPI_INT, MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_File_seek_shared(fh, 0, MPI_SEEK_SET));
    CHECK_MPI(MPI_File_read_ordered(fh, ordered_r, 2, MPI_INT, MPI_STATUS_IGNORE));

    if (ordered_r[0] != ordered_w[0] || ordered_r[1] != ordered_w[1]) {
        fprintf(stderr, "[Test 3b FAIL] Rank %d got (%d, %d), expected (%d, %d)\n",
                rank, ordered_r[0], ordered_r[1], ordered_w[0], ordered_w[1]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    CHECK_MPI(MPI_File_close(&fh));
    if (rank == 0) printf("[PASS] Test 3: Shared file pointers & ordered collectives\n");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    test_explicit_offset(rank, size);
    test_derived_view(rank, size);
    test_shared_and_ordered(rank, size);

    if (rank == 0) {
        printf("\nAll MPI-DAOS tests completed successfully across %d ranks.\n", size);
    }

    MPI_Finalize();
    return 0;
}
