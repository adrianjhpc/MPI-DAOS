#include "MPI-DAOS.h"
#include <stdlib.h>

void flat_list_init(flat_list_t *list) {
    list->cap = 64;
    list->count = 0;
    list->segs = malloc(list->cap * sizeof(flat_seg_t));
}

void flat_list_free(flat_list_t *list) {
    free(list->segs);
    list->segs = NULL;
    list->count = 0;
    list->cap = 0;
}

static void flat_list_append(flat_list_t *list, U_MPI_Aint offset, U_MPI_Aint length) {
    if (length == 0) return;

    if (list->count > 0) {
        flat_seg_t *last = &list->segs[list->count - 1];
        if (last->offset + last->length == offset) {
            last->length += length;
            return;
        }
    }

    if (list->count == list->cap) {
        list->cap *= 2;
        list->segs = realloc(list->segs, list->cap * sizeof(flat_seg_t));
    }
    list->segs[list->count].offset = offset;
    list->segs[list->count].length = length;
    list->count++;
}

void flatten_datatype(U_MPI_Datatype dtype, int count, U_MPI_Aint base_offset, flat_list_t *out) {
    int n_ints, n_addrs, n_dtypes, combiner;
    g_mpi.Type_get_envelope(dtype, &n_ints, &n_addrs, &n_dtypes, &combiner);

    if (combiner == g_mpi.combiner_named) {
        int size;
        g_mpi.Type_size(dtype, &size);
        flat_list_append(out, base_offset, (U_MPI_Aint)size * count);
        return;
    }

    int          *ints   = n_ints   ? malloc(n_ints   * sizeof(int))          : NULL;
    U_MPI_Aint     *addrs  = n_addrs  ? malloc(n_addrs  * sizeof(U_MPI_Aint))     : NULL;
    U_MPI_Datatype *dtypes = n_dtypes ? malloc(n_dtypes * sizeof(U_MPI_Datatype)) : NULL;

    g_mpi.Type_get_contents(dtype, n_ints, n_addrs, n_dtypes, ints, addrs, dtypes);

    U_MPI_Aint lb, extent;
    g_mpi.Type_get_extent(dtype, &lb, &extent);

    for (int rep = 0; rep < count; rep++) {
        U_MPI_Aint rep_base = base_offset + (rep * extent);

        if (0) {
            } else if (combiner == g_mpi.combiner_dup || combiner == g_mpi.combiner_resized) {
                flatten_datatype(dtypes[0], 1, rep_base, out);

            } else if (combiner == g_mpi.combiner_contiguous) {
                flatten_datatype(dtypes[0], ints[0], rep_base, out);

            } else if (combiner == g_mpi.combiner_vector || combiner == g_mpi.combiner_hvector) { {
                int vec_count = ints[0];
                int blklen    = ints[1];
                U_MPI_Aint sub_lb, sub_ext;
                g_mpi.Type_get_extent(dtypes[0], &sub_lb, &sub_ext);
                U_MPI_Aint stride_bytes = (combiner == g_mpi.combiner_vector)
                                        ? (U_MPI_Aint)ints[2] * sub_ext
                                        : addrs[0];

                for (int i = 0; i < vec_count; i++) {
                    flatten_datatype(dtypes[0], blklen, rep_base + i * stride_bytes, out);
                }
            }

            } else if (combiner == g_mpi.combiner_indexed || combiner == g_mpi.combiner_hindexed) { {
                int idx_count = ints[0];
                U_MPI_Aint sub_lb, sub_ext;
                g_mpi.Type_get_extent(dtypes[0], &sub_lb, &sub_ext);

                for (int i = 0; i < idx_count; i++) {
                    int blklen = ints[1 + i];
                    U_MPI_Aint disp_bytes = (combiner == g_mpi.combiner_indexed)
                                          ? (U_MPI_Aint)ints[1 + idx_count + i] * sub_ext
                                          : addrs[i];
                    flatten_datatype(dtypes[0], blklen, rep_base + disp_bytes, out);
                }
            }

            } else if (combiner == g_mpi.combiner_struct) { {
                int struct_count = ints[0];
                for (int i = 0; i < struct_count; i++) {
                    flatten_datatype(dtypes[i], ints[1 + i], rep_base + addrs[i], out);
                }
            }

            } else {
        }
    }

    for (int i = 0; i < n_dtypes; i++) {
        int tmp_i, tmp_a, tmp_d, sub_comb;
        g_mpi.Type_get_envelope(dtypes[i], &tmp_i, &tmp_a, &tmp_d, &sub_comb);
        if (sub_comb != g_mpi.combiner_named) {
            g_mpi.Type_free(&dtypes[i]);
        }
    }

    free(ints);
    free(addrs);
    free(dtypes);
}

void build_daos_sgl(const void *buf, const flat_list_t *mem_flat, d_sg_list_t *sgl) {
    sgl->sg_nr     = mem_flat->count;
    sgl->sg_nr_out = 0;
    sgl->sg_iovs   = malloc(mem_flat->count * sizeof(d_iov_t));

    const uint8_t *base_ptr = (const uint8_t *)buf;
    for (size_t i = 0; i < mem_flat->count; i++) {
        void *seg_addr = (void *)(base_ptr + mem_flat->segs[i].offset);
        d_iov_set(&sgl->sg_iovs[i], seg_addr, (size_t)mem_flat->segs[i].length);
    }
}

void build_daos_file_iod(const daos_file_ctx_t *ctx, U_MPI_Offset offset,
                         size_t total_bytes, daos_array_iod_t *iod) {
    int etype_size, ft_size;
    U_MPI_Aint ft_lb, ft_extent;

    g_mpi.Type_size(ctx->etype, &etype_size);
    g_mpi.Type_size(ctx->filetype, &ft_size);
    g_mpi.Type_get_extent(ctx->filetype, &ft_lb, &ft_extent);

    flat_list_t ft_flat;
    flat_list_init(&ft_flat);
    flatten_datatype(ctx->filetype, 1, 0, &ft_flat);

    size_t view_byte_offset = (size_t)offset * (size_t)etype_size;
    size_t ft_tile_idx      = view_byte_offset / (size_t)ft_size;
    size_t bytes_into_tile  = view_byte_offset % (size_t)ft_size;

    size_t max_ranges = ((total_bytes / (size_t)ft_size) + 2) * ft_flat.count;
    daos_range_t *rgs = malloc(max_ranges * sizeof(daos_range_t));

    size_t nr = 0;
    size_t remaining = total_bytes;
    size_t skipped = 0;

    while (remaining > 0) {
        U_MPI_Aint tile_base = ctx->disp + (U_MPI_Aint)(ft_tile_idx * ft_extent);

        for (size_t i = 0; i < ft_flat.count && remaining > 0; i++) {
            size_t seg_off = (size_t)ft_flat.segs[i].offset;
            size_t seg_len = (size_t)ft_flat.segs[i].length;

            if (skipped + seg_len <= bytes_into_tile) {
                skipped += seg_len;
                continue;
            }
            if (skipped < bytes_into_tile) {
                size_t trim = bytes_into_tile - skipped;
                seg_off += trim;
                seg_len -= trim;
                skipped = bytes_into_tile;
            }

            size_t take = (seg_len < remaining) ? seg_len : remaining;
            uint64_t abs_idx = (uint64_t)(tile_base + seg_off);

            if (nr > 0 && rgs[nr - 1].rg_idx + rgs[nr - 1].rg_len == abs_idx) {
                rgs[nr - 1].rg_len += take;
            } else {
                rgs[nr].rg_idx = abs_idx;
                rgs[nr].rg_len = take;
                nr++;
            }
            remaining -= take;
        }
        ft_tile_idx++;
    }

    flat_list_free(&ft_flat);
    iod->arr_nr  = nr;
    iod->arr_rgs = rgs;
}
