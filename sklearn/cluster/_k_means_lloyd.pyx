# Licence: BSD 3 clause
# cython: boundscheck=False, wraparound=False


import sys
from cython cimport floating
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, calloc, free
from libc.stddef cimport size_t
from libc.string cimport memset
from libc.float cimport DBL_MAX, FLT_MAX
from libc.stdio cimport printf, fflush, FILE, stdout

from ..utils._openmp_helpers cimport omp_lock_t
from ..utils._openmp_helpers cimport omp_init_lock
from ..utils._openmp_helpers cimport omp_destroy_lock
from ..utils._openmp_helpers cimport omp_set_lock
from ..utils._openmp_helpers cimport omp_unset_lock
from ..utils._openmp_helpers cimport omp_get_thread_num
from ..utils.extmath import row_norms
from ..utils._cython_blas cimport _gemm
from ..utils._cython_blas cimport RowMajor, Trans, NoTrans
from ._k_means_common import CHUNK_SIZE
from ._k_means_common cimport _relocate_empty_clusters_dense
from ._k_means_common cimport _relocate_empty_clusters_sparse
from ._k_means_common cimport _average_centers, _center_shift

cdef extern from "stdlib.h" nogil:
    int posix_memalign(void **memptr, size_t alignment, size_t size) noexcept

cdef void * aligned_alloc(size_t alignment, size_t size) noexcept nogil:
    cdef void * ptr = NULL

    # Allocate aligned memory
    if posix_memalign(&ptr, alignment, size) != 0:
        pass
    # Initialize allocated memory to zero
    memset(ptr, 0, size)
    
    return ptr

cdef void * aligned_malloc(size_t alignment, size_t size) noexcept nogil:
    cdef void * ptr = NULL

    # Allocate aligned memory
    if posix_memalign(&ptr, alignment, size) != 0:
        pass
    return ptr


def lloyd_iter_chunked_dense(
        const floating[:, ::1] X,            # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # OUT
        floating[::1] weight_in_clusters,    # OUT
        int[::1] labels,                     # OUT
        floating[::1] center_shift,          # OUT
        int n_threads,
        bint use_assign_centroids,
        bint use_assign_centroids_gemm,
        int chunk_size,
        bint update_centers=True):
    """Single iteration of K-means lloyd algorithm with dense input.

    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The observations to cluster.

    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.

    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration. `centers_new` can be `None` if
        `update_centers` is False.

    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center. `weight_in_clusters` can be `None` if `update_centers`
        is False.

    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.

    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.

    n_threads : int
        The number of threads to be used by openmp.

    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_old.shape[0]

    if n_samples == 0:
        # An empty array was passed, do nothing and return early (before
        # attempting to compute n_chunks). This can typically happen when
        # calling the prediction function of a bisecting k-means model with a
        # large fraction of outiers.
        return

    cdef:
        # hard-coded number of samples per chunk. Appeared to be close to
        # optimal in all situations.
        int n_samples_chunk = chunk_size if n_samples > chunk_size else n_samples
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_rem = n_samples % n_samples_chunk
        int chunk_idx
        int start, end

        int j, k

        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

        floating *centers_new_chunk
        floating *weight_in_clusters_chunk
        floating *pairwise_distances_chunk

        omp_lock_t lock


    if use_assign_centroids:

        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))

        assign_centroids(
                        X,
                        centers_old,
                        sample_weight,
                        labels,
                        centers_new,
                        weight_in_clusters,
                        n_samples,
                        n_features,
                        n_clusters,
                        n_threads,
                        chunk_size)

        _relocate_empty_clusters_dense(
            X, sample_weight, centers_old, centers_new, weight_in_clusters, labels
        )

        _average_centers(centers_new, weight_in_clusters)
        _center_shift(centers_old, centers_new, center_shift)


    elif use_assign_centroids_gemm:

        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))

        assign_centroids_gemm(
                        X,
                        centers_old,
                        labels,
                        centers_squared_norms,
                        n_samples,
                        n_features,
                        n_clusters,
                        n_threads)

        omp_init_lock(&lock)

        update_centroids(
                        X,
                        centers_new,
                        labels,
                        sample_weight,
                        weight_in_clusters,
                        n_samples,
                        n_features,
                        n_clusters,
                        n_threads,
                        lock)

        omp_destroy_lock(&lock)

        _relocate_empty_clusters_dense(
                X, sample_weight, centers_old, centers_new, weight_in_clusters, labels
            )

        _average_centers(centers_new, weight_in_clusters)
        _center_shift(centers_old, centers_new, center_shift)

        
    else:

        # count remainder chunk in total number of chunks
        n_chunks += n_samples != n_chunks * n_samples_chunk

        # number of threads should not be bigger than number of chunks
        n_threads = min(n_threads, n_chunks)

        if update_centers:
            memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
            memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))
            omp_init_lock(&lock)

        with nogil, parallel(num_threads=n_threads):
            # thread local buffers
            centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
            weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))
            pairwise_distances_chunk = <floating*> malloc(n_samples_chunk * n_clusters * sizeof(floating))

            for chunk_idx in prange(n_chunks, schedule='static'):
                start = chunk_idx * n_samples_chunk
                if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                    end = start + n_samples_rem
                else:
                    end = start + n_samples_chunk

                _update_chunk_dense(
                    X[start: end],
                    sample_weight[start: end],
                    centers_old,
                    centers_squared_norms,
                    labels[start: end],
                    centers_new_chunk,
                    weight_in_clusters_chunk,
                    pairwise_distances_chunk,
                    update_centers)

            # reduction from local buffers.
            if update_centers:
                # The lock is necessary to avoid race conditions when aggregating
                # info from different thread-local buffers.
                omp_set_lock(&lock)
                for j in range(n_clusters):
                    weight_in_clusters[j] += weight_in_clusters_chunk[j]
                    for k in range(n_features):
                        centers_new[j, k] += centers_new_chunk[j * n_features + k]

                omp_unset_lock(&lock)

            free(centers_new_chunk)
            free(weight_in_clusters_chunk)
            free(pairwise_distances_chunk)

        if update_centers:
            omp_destroy_lock(&lock)
            _relocate_empty_clusters_dense(
                X, sample_weight, centers_old, centers_new, weight_in_clusters, labels
            )

            _average_centers(centers_new, weight_in_clusters)
            _center_shift(centers_old, centers_new, center_shift)


cdef void distance_calculation(
    const floating[:, ::1] X,
    const floating[:, ::1] centers_old,
    const floating[::1] sample_weight,
    int[::1] labels,
    floating* centers_new_partial,
    floating* weight_in_clusters_partial,
    const int n_samples_chunk,
    const int n_clusters,
    const int n_features,
    const int n_features_unrolled,
    const int n_iter_rem,
    floating max_val) noexcept nogil:

    cdef:
        int point, cluster, feature, label, row_offset
        floating distance, diff, diff0, diff1, diff2, diff3, min_sq_dist, sample_weight_value

        const floating* X_ptr 
        const floating* centers_old_ptr
        floating* centers_new_partial_ptr

    for point in range(n_samples_chunk):
        label = 0
        min_sq_dist = max_val
        

        for cluster in range(n_clusters):
            distance = 0
            X_ptr = &X[point, 0]
            centers_old_ptr = &centers_old[cluster, 0]

            for feature in range(n_features_unrolled):
                diff0 = X_ptr[0] - centers_old_ptr[0]
                diff1 = X_ptr[1] - centers_old_ptr[1]
                diff2 = X_ptr[2] - centers_old_ptr[2]
                diff3 = X_ptr[3] - centers_old_ptr[3]
                distance += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3

                X_ptr += 4
                centers_old_ptr += 4

            for feature in range(n_iter_rem):
                diff = X_ptr[feature] - centers_old_ptr[feature]
                distance += diff * diff

            if distance < min_sq_dist:
                min_sq_dist = distance
                label = cluster
        
        labels[point] = label
        X_ptr = &X[point, 0]
        centers_new_partial_ptr = &centers_new_partial[label * n_features]

        sample_weight_value = sample_weight[point]
        weight_in_clusters_partial[label] += sample_weight_value

        for feature in range(n_features_unrolled):
            centers_new_partial_ptr[0] += X_ptr[0] * sample_weight_value
            centers_new_partial_ptr[1] += X_ptr[1] * sample_weight_value
            centers_new_partial_ptr[2] += X_ptr[2] * sample_weight_value
            centers_new_partial_ptr[3] += X_ptr[3] * sample_weight_value

            X_ptr += 4
            centers_new_partial_ptr += 4

        for feature in range(n_iter_rem):
            centers_new_partial_ptr[feature] += X_ptr[feature] * sample_weight_value

cdef void assign_centroids(
    const floating[:, ::1] X,                   # IN
    const floating[:, ::1] centers_old,         # IN
    const floating[::1] sample_weight,           # IN
    int[::1] labels,                           # 
    floating[:, ::1] centers_new,
    floating[::1] weight_in_clusters,
    const int n_samples,                        # IN
    const int n_features,                       # IN
    const int n_clusters,                       # In
    const int n_threads,
    int chunk_size):                           # IN 

    cdef int n_chunks = n_samples // chunk_size
    cdef int n_remaining = n_samples % chunk_size

    cdef:
        int j, j_lock, cluster, feature, start, end, chunk, row_offset_lock, samples, byte_alignment = 64, n_features_unrolled
        floating max_val

        omp_lock_t lock

    if floating is float:
        max_val = FLT_MAX
    elif floating is double:
        max_val = DBL_MAX
    else:
        raise TypeError("Unsupported floating type")

    n_features_unrolled = n_features // 4
    n_iter_rem = n_features % 4

    # cdef floating* distances = <floating*> calloc(n_samples * n_clusters, sizeof(floating))
    cdef floating* centers_new_partial
    cdef floating* weight_in_clusters_partial


    omp_init_lock(&lock)

    with nogil, parallel(num_threads = n_threads):

        centers_new_partial = <floating*> aligned_alloc(byte_alignment, n_clusters * n_features * sizeof(floating))
        weight_in_clusters_partial = <floating*> aligned_alloc(byte_alignment, n_clusters * sizeof(floating))

        for chunk in prange(n_chunks, schedule = 'static'):

            start = chunk * chunk_size

            if chunk == n_chunks - 1 and n_remaining > 0:
                end = start + n_remaining
            else:
                end = start + chunk_size

            samples = end - start

            distance_calculation(
                X[start:end],
                centers_old,
                sample_weight[start:end],
                labels[start:end],
                centers_new_partial,
                weight_in_clusters_partial,
                samples,
                n_clusters,
                n_features,
                n_features_unrolled,
                n_iter_rem,
                max_val
                )

        omp_set_lock(&lock)
        for cluster in range(n_clusters):
            weight_in_clusters[cluster] += weight_in_clusters_partial[cluster]
            row_offset_lock = cluster * n_features

            for j_lock in range(n_features):
                centers_new[cluster, j_lock] += centers_new_partial[row_offset_lock + j_lock]

            # if floating is float:
            #    simd_lock_partial_add_float(centers_new[cluster], &centers_new_partial[row_offset_lock], n_features)
            # else:
            #    simd_lock_partial_add_double(centers_new[cluster], &centers_new_partial[row_offset_lock], n_features)

        omp_unset_lock(&lock)

        free(centers_new_partial)
        free(weight_in_clusters_partial)
    
    omp_destroy_lock(&lock)

        
cdef assign_centroids_gemm(
    const floating[:, ::1] X,                   # IN
    const floating[:, ::1] centers_old,          # IN
    int [::1] labels,                           # IN
    const floating[::1] centers_squared_norms,   # IN
    int n_samples,                              # IN
    int n_features,                             # IN
    int n_clusters,                             # In
    int n_threads):                             # IN 

    cdef:
        int i, j, row_offset, label
        floating max_val, min_sq_dist
        floating distance
        floating* pairwise_distances = <floating*> malloc(n_samples * n_clusters * sizeof(floating))

    if floating is float:
        max_val = FLT_MAX
    elif floating is double:
        max_val = DBL_MAX
    else:
        raise TypeError("Unsupported floating type")

    # with boundscheck(False), wraparound(False):
    for i in prange(n_samples, schedule = 'static', num_threads = n_threads, nogil = True):
        row_offset = i * n_clusters
        for j in range(n_clusters):
            pairwise_distances[row_offset + j] = centers_squared_norms[j]


    _gemm(RowMajor, NoTrans, Trans, n_samples, n_clusters, n_features,
        -2.0, &X[0, 0], n_features, &centers_old[0, 0], n_features,
        1.0, pairwise_distances, n_clusters)

    for i in prange(n_samples, schedule = 'static', num_threads = n_threads, nogil = True):
        label = 0
        min_sq_dist = max_val
        row_offset = i * n_clusters

        for j in range(n_clusters):

            distance = pairwise_distances[row_offset + j]

            if distance < min_sq_dist:
                min_sq_dist = distance
                label = j

        labels[i] = label

 
cdef update_centroids (
    const floating[:, ::1] X,                         # IN
    floating[:, ::1] centers_new,                     # OUT
    int[::1] labels,                                  # OUT
    const floating[::1] sample_weight,                # IN
    floating[::1] weight_in_clusters,                  # OUT
    const int n_samples,                              # IN
    const int n_features,                             # IN
    const int n_clusters,                             # In
    const int n_threads,
    omp_lock_t lock):                             # IN 

    cdef:
        int point, j, cluster, label, row_offset, row_offset_lock, j_lock, labels_counts_value
        floating sample_weight_value 
        # int* label_counts = <int*> calloc(n_clusters, sizeof(int))
        # int* label_counts_partial
        floating* cluster_partial
        floating* weight_in_clusters_partial

    # with boundscheck(False), wraparound(False):

    with nogil, parallel(num_threads = n_threads):


        cluster_partial = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        # label_counts_partial = <int*> calloc(n_clusters, sizeof(int))
        weight_in_clusters_partial = <floating*> calloc(n_clusters, sizeof(floating))

        for point in prange(n_samples, schedule = 'static'):

            label = labels[point]
            # label_counts_partial[label] += 1
            sample_weight_value = sample_weight[point]
            weight_in_clusters_partial[label] += sample_weight_value
            row_offset = label * n_features


            for j in range(n_features):

                cluster_partial[row_offset + j] += X[point, j] * sample_weight_value

        omp_set_lock(&lock)
        for cluster in range(n_clusters):
            # label_counts[cluster] += label_counts_partial[cluster]
            weight_in_clusters[cluster] += weight_in_clusters_partial[cluster]

            row_offset_lock = cluster * n_features

            for j_lock in range(n_features):
                centers_new[cluster, j_lock] += cluster_partial[row_offset_lock + j_lock]

        omp_unset_lock(&lock)

        # free(cluster_partial)
        # free(label_counts_partial)
        free(weight_in_clusters_partial)

    omp_destroy_lock(&lock)
    
    # for cluster in range(n_clusters):

        # labels_counts_value = label_counts[cluster]

        # for j in range(n_features):

            # centers_new[cluster, j] /= labels_counts_value

    # free(label_counts)

cdef void _update_chunk_dense(
        const floating[:, ::1] X,                   # IN
        const floating[::1] sample_weight,          # IN
        const floating[:, ::1] centers_old,         # IN
        const floating[::1] centers_squared_norms,  # IN
        int[::1] labels,                            # OUT
        floating *centers_new,                      # OUT
        floating *weight_in_clusters,               # OUT
        floating *pairwise_distances,               # OUT
        bint update_centers) noexcept nogil:
    """K-means combined EM step for one dense data chunk.

    Compute the partial contribution of a single data chunk to the labels and
    centers.
    """
    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating sq_dist, min_sq_dist
        int i, j, k, label

    # Instead of computing the full pairwise squared distances matrix,
    # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to store
    # the - 2 X.C^T + ||C||² term since the argmin for a given sample only
    # depends on the centers.
    # pairwise_distances = ||C||²
    # Only iterates over all the sample in the chunk
    for i in range(n_samples):
        for j in range(n_clusters):
            # adds the ||C||² for every data point as this is independent 
            # the pairwise distance has i * n_cluster points (distance from every point to every cluster)
            pairwise_distances[i * n_clusters + j] = centers_squared_norms[j]

    # pairwise_distances += -2 * X.dot(C.T)
    _gemm(RowMajor, NoTrans, Trans, n_samples, n_clusters, n_features,
          -2.0, &X[0, 0], n_features, &centers_old[0, 0], n_features,
          1.0, pairwise_distances, n_clusters)

    for i in range(n_samples):
        min_sq_dist = pairwise_distances[i * n_clusters]
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pairwise_distances[i * n_clusters + j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(n_features):
                centers_new[label * n_features + k] += X[i, k] * sample_weight[i]


def lloyd_iter_chunked_sparse(
        X,                                   # IN
        const floating[::1] sample_weight,   # IN
        const floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,        # OUT
        floating[::1] weight_in_clusters,    # OUT
        int[::1] labels,                     # OUT
        floating[::1] center_shift,          # OUT
        int n_threads,
        bint update_centers=True):
    """Single iteration of K-means lloyd algorithm with sparse input.

    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features), dtype=floating
        The observations to cluster. Must be in CSR format.

    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.

    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration. `centers_new` can be `None` if
        `update_centers` is False.

    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center. `weight_in_clusters` can be `None` if `update_centers`
        is False.

    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.

    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.

    n_threads : int
        The number of threads to be used by openmp.

    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_old.shape[0]

    if n_samples == 0:
        # An empty array was passed, do nothing and return early (before
        # attempting to compute n_chunks). This can typically happen when
        # calling the prediction function of a bisecting k-means model with a
        # large fraction of outiers.
        return

    cdef:
        # Choose same as for dense. Does not have the same impact since with
        # sparse data the pairwise distances matrix is not precomputed.
        # However, splitting in chunks is necessary to get parallelism.
        int n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_rem = n_samples % n_samples_chunk
        int chunk_idx
        int start = 0, end = 0

        int j, k

        floating[::1] X_data = X.data
        int[::1] X_indices = X.indices
        int[::1] X_indptr = X.indptr

        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

        floating *centers_new_chunk
        floating *weight_in_clusters_chunk

        omp_lock_t lock

    # count remainder chunk in total number of chunks
    n_chunks += n_samples != n_chunks * n_samples_chunk

    # number of threads should not be bigger than number of chunks
    n_threads = min(n_threads, n_chunks)

    if update_centers:
        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))
        omp_init_lock(&lock)

    with nogil, parallel(num_threads=n_threads):
        # thread local buffers
        centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))

        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk

            _update_chunk_sparse(
                X_data[X_indptr[start]: X_indptr[end]],
                X_indices[X_indptr[start]: X_indptr[end]],
                X_indptr[start: end+1],
                sample_weight[start: end],
                centers_old,
                centers_squared_norms,
                labels[start: end],
                centers_new_chunk,
                weight_in_clusters_chunk,
                update_centers)

        # reduction from local buffers.
        if update_centers:
            # The lock is necessary to avoid race conditions when aggregating
            # info from different thread-local buffers.
            omp_set_lock(&lock)
            for j in range(n_clusters):
                weight_in_clusters[j] += weight_in_clusters_chunk[j]
                for k in range(n_features):
                    centers_new[j, k] += centers_new_chunk[j * n_features + k]
            omp_unset_lock(&lock)

        free(centers_new_chunk)
        free(weight_in_clusters_chunk)

    if update_centers:
        omp_destroy_lock(&lock)
        _relocate_empty_clusters_sparse(
            X_data, X_indices, X_indptr, sample_weight,
            centers_old, centers_new, weight_in_clusters, labels)

        _average_centers(centers_new, weight_in_clusters)
        _center_shift(centers_old, centers_new, center_shift)


cdef void _update_chunk_sparse(
        const floating[::1] X_data,                 # IN
        const int[::1] X_indices,                   # IN
        const int[::1] X_indptr,                    # IN
        const floating[::1] sample_weight,          # IN
        const floating[:, ::1] centers_old,         # IN
        const floating[::1] centers_squared_norms,  # IN
        int[::1] labels,                            # OUT
        floating *centers_new,                      # OUT
        floating *weight_in_clusters,               # OUT
        bint update_centers) noexcept nogil:
    """K-means combined EM step for one sparse data chunk.

    Compute the partial contribution of a single data chunk to the labels and
    centers.
    """
    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating sq_dist, min_sq_dist
        int i, j, k, label
        floating max_floating = FLT_MAX if floating is float else DBL_MAX
        int s = X_indptr[0]

    # XXX Precompute the pairwise distances matrix is not worth for sparse
    # currently. Should be tested when BLAS (sparse x dense) matrix
    # multiplication is available.
    for i in range(n_samples):
        min_sq_dist = max_floating
        label = 0

        for j in range(n_clusters):
            sq_dist = 0.0
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                sq_dist += centers_old[j, X_indices[k]] * X_data[k]

            # Instead of computing the full squared distance with each cluster,
            # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to compute
            # the - 2 X.C^T + ||C||² term since the argmin for a given sample
            # only depends on the centers C.
            sq_dist = centers_squared_norms[j] -2 * sq_dist
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j

        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                centers_new[label * n_features + X_indices[k]] += X_data[k] * sample_weight[i]
