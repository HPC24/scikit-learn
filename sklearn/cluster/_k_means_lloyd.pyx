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
from ..utils.extmath import row_norms
from ..utils._cython_blas cimport _gemm
from ..utils._cython_blas cimport RowMajor, Trans, NoTrans
from ._k_means_common import CHUNK_SIZE
from ._k_means_common cimport _relocate_empty_clusters_dense
from ._k_means_common cimport _relocate_empty_clusters_sparse
from ._k_means_common cimport _average_centers, _center_shift

cdef extern from "stdlib.h" nogil:
    int posix_memalign(void **memptr, size_t alignment, size_t size)


cdef extern from "immintrin.h" nogil:
    # AVX-512 vector types
    ctypedef struct __m512:
        pass
    ctypedef struct __m512d:
        pass
    ctypedef struct __m512i:
        pass

    # Load and store functions for aligned data
    __m512 _mm512_load_ps(const float * mem_addr)
    void _mm512_store_ps(float * mem_addr, __m512 a)
    __m512d _mm512_load_pd(const double * mem_addr)
    void _mm512_store_pd(double * mem_addr, __m512d a)
    __m512i _mm512_load_si512(const int * mem_addr)
    void _mm512_store_si512(void * mem_addr, __m512i a)

    # Arithmetic operations
    __m512 _mm512_add_ps(__m512 a, __m512 b)
    __m512 _mm512_sub_ps(__m512 a, __m512 b)
    __m512 _mm512_mul_ps(__m512 a, __m512 b)
    __m512 _mm512_div_ps(__m512 a, __m512 b)
    __m512 _mm512_fmadd_ps(__m512 a, __m512 b, __m512 c)
    __m512 _mm512_setzero_ps()
    __m512 _mm512_set1_ps(float a)
    float _mm512_reduce_add_ps (__m512 a)

    __m512d _mm512_add_pd(__m512d a, __m512d b)
    __m512d _mm512_sub_pd(__m512d a, __m512d b)
    __m512d _mm512_mul_pd(__m512d a, __m512d b)
    __m512d _mm512_div_pd(__m512d a, __m512d b)
    __m512d _mm512_fmadd_pd(__m512d a, __m512d b, __m512d c)
    __m512d _mm512_setzero_pd()
    __m512d _mm512_set1_pd(double a)
    double _mm512_reduce_add_pd(__m512d a)


    # AVX-256 vector types
    ctypedef struct __m256:
        pass
    ctypedef struct __m256d:
        pass
    ctypedef struct __m256i:
        pass
    ctypedef struct __m128:
        pass
    ctypedef struct __m128d:
        pass

    # Load and store functions for aligned data
    __m256 _mm256_load_ps(const void * mem_addr)
    void _mm256_store_ps(void * mem_addr, __m256 a)
    __m256d _mm256_load_pd(const void* mem_addr)
    void _mm256_store_pd(void * mem_addr, __m256d a)
    __m256i _mm256_load_si256(const void * mem_addr)
    void _mm256_store_si256(void * mem_addr, __m256i a)

    # Arithmetic operations
    __m256 _mm256_add_ps(__m256 a, __m256 b)
    __m256 _mm256_sub_ps(__m256 a, __m256 b)
    __m256 _mm256_mul_ps(__m256 a, __m256 b)
    __m256 _mm256_div_ps(__m256 a, __m256 b)
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c)
    __m256 _mm256_setzero_ps()
    __m256 _mm256_set1_ps(float a)

    __m256d _mm256_add_pd(__m256d a, __m256d b)
    __m256d _mm256_sub_pd(__m256d a, __m256d b)
    __m256d _mm256_mul_pd(__m256d a, __m256d b)
    __m256d _mm256_div_pd(__m256d a, __m256d b)
    __m256d _mm256_fmadd_pd(__m256d a, __m256d b, __m256d c)
    __m256d _mm256_setzero_pd()
    __m256d _mm256_set1_pd(double a)

    __m128d _mm256_castpd256_pd128(__m256d a)             # Cast 256-bit __m256d to 128-bit __m128d
    __m128d _mm256_extractf128_pd(__m256d a, int imm8)    # Extract upper 128 bits from 256-bit __m256d
    __m128d _mm_add_pd(__m128d a, __m128d b)              # Add two __m128d vectors element-wise
    __m128d _mm_hadd_pd(__m128d a, __m128d b)             # Horizontal add of two __m128d vectors
    double _mm_cvtsd_f64(__m128d a)                       # Extract the lower double from __m128d

    __m128 _mm256_castps256_ps128(__m256 a)             # Cast 256-bit __m256d to 128-bit __m128d
    __m128 _mm256_extractf128_ps(__m256 a, int imm8)    # Extract upper 128 bits from 256-bit __m256d
    __m128 _mm_add_ps(__m128 a, __m128 b)              # Add two __m128d vectors element-wise
    __m128 _mm_hadd_ps(__m128 a, __m128 b)             # Horizontal add of two __m128d vectors
    float _mm_cvtss_f32(__m128 a)                       # Extract the lower double from __m128d


cdef void * aligned_alloc(size_t alignment, size_t size) noexcept nogil:
    cdef void * ptr = NULL

    # Allocate aligned memory
    if posix_memalign(&ptr, alignment, size) != 0:
        printf("Wrong memory allocation")

    # Initialize allocated memory to zero
    memset(ptr, 0, size)
    
    return ptr

cdef void * aligned_malloc(size_t alignment, size_t size) noexcept nogil:
    cdef void * ptr = NULL

    # Allocate aligned memory
    if posix_memalign(&ptr, alignment, size) != 0:
        printf("Wrong memory allocation")
    
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

cdef void simd_partial_cluster_float(
    const float[::1] X,
    float* centers_new_partial,
    float sample_weight_value,
    int n_features) noexcept nogil:

    cdef: 
        int i, remaining, rem_iterator
        __m512 centers, X_partial, weight_vec
    
    remaining = n_features % 16
    rem_iterator = n_features - remaining

    weight_vec = _mm512_set1_ps(sample_weight_value)

    for i in range(0, rem_iterator, 16):

        centers = _mm512_load_ps(&centers_new_partial[i])   # load 4 values of the partial centers vector into a register
        X_partial = _mm512_load_ps(&X[i])                   # load 4 values of the point that is being processed into a register
        X_partial = _mm512_mul_ps(X_partial, weight_vec)    # multiply the point values with the sample weight
        centers = _mm512_add_ps(centers, X_partial)         # add the point values to the centers
        _mm512_store_ps(&centers_new_partial[i], centers)   # Load values into the old centers_new_partial vector

    for i in range(rem_iterator, n_features):
        centers_new_partial[i] += X[i] * sample_weight_value


cdef float simd_float(
    const float[::1] X,
    const float[::1] centers_old,
    const int n_features) noexcept nogil:

    cdef:
        int i, remaining, rem_iterator
        __m512 sum_vector, X_partial, centers_partial, difference
        float distance, diff

    remaining = n_features % 16
    rem_iterator = n_features - remaining

    sum_vector = _mm512_setzero_ps()

    for i in range(0, rem_iterator, 16):

        X_partial = _mm512_load_ps(&X[i])
        centers_partial = _mm512_load_ps(&centers_old[i])
        difference = _mm512_sub_ps(X_partial, centers_partial)
        difference = _mm512_mul_ps(difference, difference)
        sum_vector = _mm512_add_ps(sum_vector, difference)

    distance = _mm512_reduce_add_ps(sum_vector)

    for i in range(rem_iterator, n_features):
        diff = X[i] - centers_old[i]
        distance += diff * diff

    return distance


cdef void simd_partial_cluster_double(
    const double[::1] X,
    double* centers_new_partial,
    double sample_weight_value,
    int n_features) noexcept nogil:

    cdef: 
        int i, remaining, rem_iterator
        __m512d centers, X_partial, weight_vec
    
    remaining = n_features % 8
    rem_iterator = n_features - remaining

    weight_vec = _mm512_set1_pd(sample_weight_value)

    for i in range(0, rem_iterator, 8):

        centers = _mm512_load_pd(&centers_new_partial[i])   # load 4 values of the partial centers vector into a register
        X_partial = _mm512_load_pd(&X[i])                   # load 4 values of the point that is being processed into a register
        X_partial = _mm512_mul_pd(X_partial, weight_vec)    # multiply the point values with the sample weight
        centers = _mm512_add_pd(centers, X_partial)         # add the point values to the centers
        _mm512_store_pd(&centers_new_partial[i], centers)   # Load values into the old centers_new_partial vector

    for i in range(rem_iterator, n_features):
        centers_new_partial[i] += X[i] * sample_weight_value


cdef double simd_double(
    const double[::1] X,
    const double[::1] centers_old,
    const int n_features) noexcept nogil:

    cdef:
        int i, remaining, rem_iterator, bit_lane = 1
        __m512d sum_vector, X_partial, centers_partial, difference
        double distance, diff

    remaining = n_features % 8
    rem_iterator = n_features - remaining

    sum_vector = _mm512_setzero_pd()

    for i in range(0, rem_iterator, 8):

        X_partial = _mm512_load_pd(&X[i])
        centers_partial = _mm512_load_pd(&centers_old[i])
        difference = _mm512_sub_pd(X_partial, centers_partial)
        difference = _mm512_mul_pd(difference, difference)
        sum_vector = _mm512_add_pd(sum_vector, difference)

    distance = _mm512_reduce_add_pd(sum_vector) 

    for i in range(rem_iterator, n_features):
        diff = X[i] - centers_old[i]
        distance += diff * diff

    return distance


cdef void simd_distance_calculation_float(    
    const float[:, ::1] X,
    const float[:, ::1] centers_old,
    const float[::1] sample_weight,
    int[::1] labels,
    float* centers_new_partial,
    float* weight_in_clusters_partial,
    const int n_samples_chunk,
    const int n_clusters,
    const int n_features,
    float max_val) noexcept nogil:

    
    cdef:
        int point, cluster, feature, label, row_offset
        float distance, diff, min_sq_dist, sample_weight_value


    for point in range(n_samples_chunk):
        label = 0
        min_sq_dist = max_val

        for cluster in range(n_clusters):

            distance = simd_float(X[point], centers_old[cluster], n_features)

            if distance < min_sq_dist:
                min_sq_dist = distance
                label = cluster
        
        labels[point] = label
        row_offset = label * n_features
        sample_weight_value = sample_weight[point]
        weight_in_clusters_partial[label] += sample_weight_value

        simd_partial_cluster_float(X[point], &centers_new_partial[row_offset], sample_weight_value, n_features)


cdef void simd_distance_calculation_double(    
    const double[:, ::1] X,
    const double[:, ::1] centers_old,
    const double[::1] sample_weight,
    int[::1] labels,
    double* centers_new_partial,
    double* weight_in_clusters_partial,
    const int n_samples_chunk,
    const int n_clusters,
    const int n_features,
    double max_val) noexcept nogil:

    
    cdef:
        int point, cluster, feature, label, row_offset
        double distance, diff, min_sq_dist, sample_weight_value


    for point in range(n_samples_chunk):
        label = 0
        min_sq_dist = max_val

        for cluster in range(n_clusters):

            distance = simd_double(X[point], centers_old[cluster], n_features)

            if distance < min_sq_dist:
                min_sq_dist = distance
                label = cluster
        
        labels[point] = label
        row_offset = label * n_features
        sample_weight_value = sample_weight[point]
        weight_in_clusters_partial[label] += sample_weight_value

        simd_partial_cluster_double(X[point], &centers_new_partial[row_offset], sample_weight_value, n_features)


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
    floating max_val) noexcept nogil:

    cdef:
        int point, cluster, feature, label, row_offset
        floating distance, diff, min_sq_dist, sample_weight_value

    for point in range(n_samples_chunk):
        label = 0
        min_sq_dist = max_val

        for cluster in range(n_clusters):
            distance = 0

            for feature in range(n_features):
                diff = X[point, feature] - centers_old[cluster, feature]
                distance += diff * diff

            if distance < min_sq_dist:
                min_sq_dist = distance
                label = cluster
        
        labels[point] = label
        row_offset = label * n_features
        sample_weight_value = sample_weight[point]
        weight_in_clusters_partial[label] += sample_weight_value

        for feature in range(n_features):
            centers_new_partial[row_offset + feature] += X[point, feature] * sample_weight_value

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
        int j, cluster, feature, start, end, chunk, row_offset_lock, samples, byte_alignment = 64
        floating max_val

        omp_lock_t lock

    if floating is float:
        max_val = FLT_MAX
    elif floating is double:
        max_val = DBL_MAX
    else:
        raise TypeError("Unsupported floating type")


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

            if floating is float:

                simd_distance_calculation_float(
                    X[start:end],
                    centers_old,
                    sample_weight[start:end],
                    labels[start:end],
                    centers_new_partial,
                    weight_in_clusters_partial,
                    samples,
                    n_clusters,
                    n_features,
                    max_val
                )

            else:

                simd_distance_calculation_double(
                    X[start:end],
                    centers_old,
                    sample_weight[start:end],
                    labels[start:end],
                    centers_new_partial,
                    weight_in_clusters_partial,
                    samples,
                    n_clusters,
                    n_features,
                    max_val
                )

        omp_set_lock(&lock)
        for cluster in range(n_clusters):
            weight_in_clusters[cluster] += weight_in_clusters_partial[cluster]

            row_offset_lock = cluster * n_features

            for j in range(n_features):
                centers_new[cluster, j] += centers_new_partial[row_offset_lock + j]

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
