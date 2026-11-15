#include "MPI-DAOS.h"
#include <stdatomic.h>
#include <stdlib.h>

#define TABLE_SIZE 256

typedef struct handle_entry {
    uint32_t key;
    daos_file_ctx_t     *ctx;
    struct handle_entry *next;
} handle_entry_t;

static handle_entry_t  *g_table[TABLE_SIZE] = {0};
static pthread_rwlock_t g_table_lock = PTHREAD_RWLOCK_INITIALIZER;
/* Start IDs at 1 so g_mpi.file_null (0) is never collided with */
static atomic_uint g_next_fh_id = 1;

static inline size_t hash_fh(U_MPI_File fh) {
    uint32_t k = (uint32_t)fh;
    return (k ^ (k >> 16)) & (TABLE_SIZE - 1);
}

U_MPI_File handle_table_alloc_fh(void) {
    uint32_t id = atomic_fetch_add(&g_next_fh_id, 1);
    return (U_MPI_File)(uintptr_t)id;
}

U_MPI_Fint mpidaos_file_c2f(U_MPI_File fh) {
    if (fh == g_mpi.file_null) return 0;
    return (U_MPI_Fint)(uintptr_t)fh;
}

U_MPI_File mpidaos_file_f2c(U_MPI_Fint f_fh) {
    if (f_fh == 0) return g_mpi.file_null;
    return (U_MPI_File)(uintptr_t)f_fh;
}

int handle_table_insert(U_MPI_File fh, daos_file_ctx_t *ctx) {
    size_t idx = hash_fh(fh);
    handle_entry_t *entry = malloc(sizeof(handle_entry_t));
    if (!entry) return -1;

    entry->key = (uint32_t)fh;
    entry->ctx = ctx;

    pthread_rwlock_wrlock(&g_table_lock);
    entry->next = g_table[idx];
    g_table[idx] = entry;
    pthread_rwlock_unlock(&g_table_lock);
    return 0;
}

daos_file_ctx_t *handle_table_lookup(U_MPI_File fh) {
    size_t idx = hash_fh(fh);
    uint32_t k = (uint32_t)fh;
    daos_file_ctx_t *ctx = NULL;

    pthread_rwlock_rdlock(&g_table_lock);
    for (handle_entry_t *e = g_table[idx]; e != NULL; e = e->next) {
        if (e->key == k) {
            ctx = e->ctx;
            break;
        }
    }
    pthread_rwlock_unlock(&g_table_lock);
    return ctx;
}

daos_file_ctx_t *handle_table_remove(U_MPI_File fh) {
    size_t idx = hash_fh(fh);
    uint32_t k = (uint32_t)fh;
    daos_file_ctx_t *ctx = NULL;

    pthread_rwlock_wrlock(&g_table_lock);
    handle_entry_t **curr = &g_table[idx];
    while (*curr) {
        if ((*curr)->key == k) {
            handle_entry_t *victim = *curr;
            ctx = victim->ctx;
            *curr = victim->next;
            free(victim);
            break;
        }
        curr = &(*curr)->next;
    }
    pthread_rwlock_unlock(&g_table_lock);
    return ctx;
}
