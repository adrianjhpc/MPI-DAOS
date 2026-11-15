program test_mpidaos
    use mpi_f08
    use, intrinsic :: iso_fortran_env, only : int32, int64
    implicit none

    integer :: rank, nprocs, ierr

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

    call test_explicit_offset(rank, nprocs)
    call test_derived_view(rank, nprocs)
    call test_shared_and_ordered(rank, nprocs)

    if (rank == 0) then
        write(*, '(/,A,I0,A)') 'All Fortran MPI-DAOS tests completed successfully across ', &
                               nprocs, ' ranks.'
    end if

    call MPI_Finalize(ierr)

contains

    subroutine check_mpi(ierr, rank, line)
        integer, intent(in) :: ierr, rank, line
        character(len=MPI_MAX_ERROR_STRING) :: errstr
        integer :: reslen, abort_err

        if (ierr /= MPI_SUCCESS) then
            call MPI_Error_string(ierr, errstr, reslen, abort_err)
            write(*, '(A,I0,A,I0,A,A)') 'Rank ', rank, ' failed at line ', line, ': ', trim(errstr)
            call MPI_Abort(MPI_COMM_WORLD, ierr, abort_err)
        end if
    end subroutine check_mpi

    ! Test 1: Explicit-offset contiguous read/write
    subroutine test_explicit_offset(rank, nprocs)
        integer, intent(in) :: rank, nprocs
        type(MPI_File) :: fh
        integer, parameter :: count = 16
        integer(int32) :: wbuf(count), rbuf(count)
        integer(kind=MPI_OFFSET_KIND) :: offset, read_offset
        integer :: i, target_rank, int_bytes, ierr, abort_err

        call MPI_Type_size(MPI_INTEGER, int_bytes, ierr)

        do i = 1, count
            wbuf(i) = (rank * 1000) + (i - 1)
        end do
        rbuf = 0

        call MPI_File_open(MPI_COMM_WORLD, 'test_contig_f.dat', &
                           IOR(MPI_MODE_CREATE, MPI_MODE_RDWR), MPI_INFO_NULL, fh, ierr)
        call check_mpi(ierr, rank, __LINE__)

        offset = int(rank, MPI_OFFSET_KIND) * int(count, MPI_OFFSET_KIND) * int(int_bytes, MPI_OFFSET_KIND)
        call MPI_File_write_at(fh, offset, wbuf, count, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
        call check_mpi(ierr, rank, __LINE__)

        call MPI_Barrier(MPI_COMM_WORLD, ierr)

        ! Read neighbor rank's data to verify cross-rank visibility
        target_rank = mod(rank + 1, nprocs)
        read_offset = int(target_rank, MPI_OFFSET_KIND) * int(count, MPI_OFFSET_KIND) * int(int_bytes, MPI_OFFSET_KIND)
        call MPI_File_read_at(fh, read_offset, rbuf, count, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
        call check_mpi(ierr, rank, __LINE__)

        do i = 1, count
            if (rbuf(i) /= (target_rank * 1000) + (i - 1)) then
                write(*, '(A,I0,A,I0,A,I0,A,I0)') '[Test 1 FAIL] Rank ', rank, ' index ', i, &
                    ': got ', rbuf(i), ', expected ', (target_rank * 1000) + (i - 1)
                call MPI_Abort(MPI_COMM_WORLD, 1, abort_err)
            end if
        end do

        call MPI_File_close(fh, ierr)
        call check_mpi(ierr, rank, __LINE__)

        if (rank == 0) write(*, '(A)') '[PASS] Test 1: Explicit-offset contiguous read/write'
    end subroutine test_explicit_offset

    ! Test 2: Non-contiguous derived datatype + strided file view
    subroutine test_derived_view(rank, nprocs)
        integer, intent(in) :: rank, nprocs
        type(MPI_File) :: fh
        type(MPI_Datatype) :: filetype, memtype
        integer(int32) :: wbuf(8), rbuf(8)
        integer(kind=MPI_OFFSET_KIND) :: disp
        integer :: check_indices(4) = [1, 2, 5, 6]
        integer :: k, idx, int_bytes, ierr, abort_err

        call MPI_Type_size(MPI_INTEGER, int_bytes, ierr)

        ! Each rank owns 1 integer every 'nprocs' integers in the file
        call MPI_Type_vector(4, 1, nprocs, MPI_INTEGER, filetype, ierr)
        call MPI_Type_commit(filetype, ierr)

        ! Non-contiguous memory buffer: 2 blocks of 2 integers with stride 4 (1-based: 1,2 and 5,6)
        call MPI_Type_vector(2, 2, 4, MPI_INTEGER, memtype, ierr)
        call MPI_Type_commit(memtype, ierr)

        wbuf = 0
        rbuf = 0
        wbuf(1) = rank * 10 + 1
        wbuf(2) = rank * 10 + 2
        wbuf(5) = rank * 10 + 3
        wbuf(6) = rank * 10 + 4

        call MPI_File_open(MPI_COMM_WORLD, 'test_view_f.dat', &
                           IOR(MPI_MODE_CREATE, MPI_MODE_RDWR), MPI_INFO_NULL, fh, ierr)
        call check_mpi(ierr, rank, __LINE__)

        disp = int(rank, MPI_OFFSET_KIND) * int(int_bytes, MPI_OFFSET_KIND)
        call MPI_File_set_view(fh, disp, MPI_INTEGER, filetype, 'native', MPI_INFO_NULL, ierr)
        call check_mpi(ierr, rank, __LINE__)

        call MPI_File_write_at(fh, 0_MPI_OFFSET_KIND, wbuf, 1, memtype, MPI_STATUS_IGNORE, ierr)
        call check_mpi(ierr, rank, __LINE__)

        call MPI_Barrier(MPI_COMM_WORLD, ierr)

        call MPI_File_read_at(fh, 0_MPI_OFFSET_KIND, rbuf, 1, memtype, MPI_STATUS_IGNORE, ierr)
        call check_mpi(ierr, rank, __LINE__)

        do k = 1, 4
            idx = check_indices(k)
            if (rbuf(idx) /= wbuf(idx)) then
                write(*, '(A,I0,A,I0,A,I0,A,I0)') '[Test 2 FAIL] Rank ', rank, ' slot ', idx, &
                    ': got ', rbuf(idx), ', expected ', wbuf(idx)
                call MPI_Abort(MPI_COMM_WORLD, 1, abort_err)
            end if
        end do

        call MPI_File_close(fh, ierr)
        call check_mpi(ierr, rank, __LINE__)

        call MPI_Type_free(memtype, ierr)
        call MPI_Type_free(filetype, ierr)

        if (rank == 0) write(*, '(A)') '[PASS] Test 2: Non-contiguous derived datatype & file view'
    end subroutine test_derived_view

    ! Test 3: Shared file pointers (atomic KV metadata object) & Ordered collectives
    subroutine test_shared_and_ordered(rank, nprocs)
        integer, intent(in) :: rank, nprocs
        type(MPI_File) :: fh
        integer(int32) :: my_val, ordered_w(2), ordered_r(2)
        integer(int32), allocatable :: all_vals(:), seen(:)
        integer :: i, r, ierr, abort_err

        call MPI_File_open(MPI_COMM_WORLD, 'test_shared_f.dat', &
                           IOR(MPI_MODE_CREATE, MPI_MODE_RDWR), MPI_INFO_NULL, fh, ierr)
        call check_mpi(ierr, rank, __LINE__)

        ! 3a. Concurrent uncoordinated shared writes
        my_val = 500 + rank
        call MPI_File_write_shared(fh, my_val, 1, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
        call check_mpi(ierr, rank, __LINE__)

        call MPI_Barrier(MPI_COMM_WORLD, ierr)

        call MPI_File_seek_shared(fh, 0_MPI_OFFSET_KIND, MPI_SEEK_SET, ierr)
        call check_mpi(ierr, rank, __LINE__)

        if (rank == 0) then
            allocate(all_vals(nprocs), seen(0:nprocs-1))
            all_vals = 0
            seen = 0

            call MPI_File_read_at(fh, 0_MPI_OFFSET_KIND, all_vals, nprocs, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
            call check_mpi(ierr, rank, __LINE__)

            do i = 1, nprocs
                r = all_vals(i) - 500
                if (r < 0 .or. r >= nprocs) then
                    write(*, '(A,I0,A,I0)') '[Test 3a FAIL] Corrupt entry ', all_vals(i), ' at slot ', i
                    call MPI_Abort(MPI_COMM_WORLD, 1, abort_err)
                end if
                if (seen(r) /= 0) then
                    write(*, '(A,I0,A,I0)') '[Test 3a FAIL] Duplicate entry ', all_vals(i), ' at slot ', i
                    call MPI_Abort(MPI_COMM_WORLD, 1, abort_err)
                end if
                seen(r) = 1
            end do
            deallocate(all_vals, seen)
        end if

        call MPI_Barrier(MPI_COMM_WORLD, ierr)

        ! 3b. Collective ordered write & read
        call MPI_File_seek_shared(fh, 0_MPI_OFFSET_KIND, MPI_SEEK_SET, ierr)
        ordered_w = [rank * 2, rank * 2 + 1]
        ordered_r = [-1, -1]

        call MPI_File_write_ordered(fh, ordered_w, 2, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
        call check_mpi(ierr, rank, __LINE__)

        call MPI_File_seek_shared(fh, 0_MPI_OFFSET_KIND, MPI_SEEK_SET, ierr)
        call MPI_File_read_ordered(fh, ordered_r, 2, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
        call check_mpi(ierr, rank, __LINE__)

        if (ordered_r(1) /= ordered_w(1) .or. ordered_r(2) /= ordered_w(2)) then
            write(*, '(A,I0,A,I0,A,I0,A,I0,A,I0)') '[Test 3b FAIL] Rank ', rank, &
                ' got (', ordered_r(1), ',', ordered_r(2), '), expected (', ordered_w(1), ',', ordered_w(2), ')'
            call MPI_Abort(MPI_COMM_WORLD, 1, abort_err)
        end if

        call MPI_File_close(fh, ierr)
        call check_mpi(ierr, rank, __LINE__)

        if (rank == 0) write(*, '(A)') '[PASS] Test 3: Shared file pointers & ordered collectives'
    end subroutine test_shared_and_ordered

end program test_mpidaos
