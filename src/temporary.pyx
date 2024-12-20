# at the top of assign_centroids function

# number to scale the n_samples_chunk
cdef int n_samples_chunk_scaling = 256

# Buffer that reserves some part of the L1 cache for other memory allocations
cdef int byte_buffer = 2048
# Number of bytes that get occupied by a single sample

cdef int single_sample_size = n_features * sizeof(floating) 

printf("Samples size: %d \n",single_sample_size)

# this is how much space all the centroids need
cdef int centroids_size = single_sample_size * n_clusters

# now calculate how many data points + the clusters fit into the cache

# Assuming 32KB L1 cache
cdef int n_samples_chunk = ((32 * 1024 - byte_buffer) // single_sample_size ) * n_samples_chunk_scaling

printf("Samples per chunk: %d \n",n_samples_chunk)

cdef int n_chunks = n_samples // n_samples_chunk
cdef int n_remaining = n_samples % n_samples_chunk





cdef void simd_partial_cluster_float(
    const float[::1] X,
    float* centers_new_partial,
    float sample_weight_value,
    int n_features) noexcept nogil:

    cdef: 
        int i, remaining, rem_iterator
        __m256 centers, X_partial, weight_vec
    
    remaining = n_features % 8
    rem_iterator = n_features - remaining

    weight_vec = _mm256_set1_ps(sample_weight_value)

    for i in range(0, n_features, 8):

        centers = _mm256_load_ps(&centers_new_partial[i])   # load 4 values of the partial centers vector into a register
        X_partial = _mm256_load_ps(&X[i])                   # load 4 values of the point that is being processed into a register
        X_partial = _mm256_mul_ps(X_partial, weight_vec)    # multiply the point values with the sample weight
        centers = _mm256_add_ps(centers, X_partial)         # add the point values to the centers
        _mm256_store_ps(&centers_new_partial[i], centers)   # Load values into the old centers_new_partial vector

    for i in range(rem_iterator, n_features):
        centers_new_partial[i] += X[i] * sample_weight_value


cdef floating simd_float(
    const floating[::1] X,
    const floating[::1] centers_old,
    const int n_features) noexcept nogil:

    cdef:
        int i, remaining, rem_iterator, bit_lane = 1
        __m256 sum_vector, X_partial, centers_partial, difference
        __m128 upper, lower, result
        double distance, diff

    remaining = n_features % 8
    rem_iterator = n_features - remaining

    sum_vector = _mm256_setzero_ps()

    for i in range(0, n_features, 8):

        X_partial = _mm256_load_ps(&X[i])
        centers_partial = _mm256_load_ps(&centers_old[i])
        difference = _mm256_sub_ps(X_partial, centers_partial)
        difference = _mm256_mul_ps(difference, difference)
        sum_vector = _mm256_add_ps(sum_vector, difference)

    upper = _mm256_castps256_ps128(sum_vector)        # [a0, a1 , a2, a3]
    lower = _mm256_extractf128_ps(sum_vector, bit_lane)   # [a4, a5, a6, a7]
    result = _mm_add_ps(lower, upper)           # [(a0 + a4), (a1 + a5), (a2 + a6), (a3 + a7)]
    result = _mm_hadd_ps(result, result)        # [(a0 + a4 + a1 + a5), (a2 + a6 + a3 + a7), X, X] X = duplicates
    result = _mm_hadd_ps(result, result)        # [(a0 + a4 + a1 + a5 + a2 + a6 + a3 + a7), X, X, X] result is now in the first 32 bits
    distance = _mm_cvtss_f32(result)            # extract the result in the first 32 bits

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
        __m256d centers, X_partial, weight_vec
    
    remaining = n_features % 4
    rem_iterator = n_features - remaining

    weight_vec = _mm256_set1_pd(sample_weight_value)

    for i in range(0, rem_iterator, 4):

        centers = _mm256_load_pd(&centers_new_partial[i])   # load 4 values of the partial centers vector into a register
        X_partial = _mm256_load_pd(&X[i])                   # load 4 values of the point that is being processed into a register
        X_partial = _mm256_mul_pd(X_partial, weight_vec)    # multiply the point values with the sample weight
        centers = _mm256_add_pd(centers, X_partial)         # add the point values to the centers
        _mm256_store_pd(&centers_new_partial[i], centers)   # Load values into the old centers_new_partial vector

    for i in range(rem_iterator, n_features):
        centers_new_partial[i] += X[i] * sample_weight_value


cdef double simd_double(
    const double[::1] X,
    const double[::1] centers_old,
    const int n_features) noexcept nogil:

    cdef:
        int i, remaining, rem_iterator, bit_lane = 1
        __m256d sum_vector, X_partial, centers_partial, difference
        __m128d upper, lower, result
        double distance, diff

    remaining = n_features % 4
    rem_iterator = n_features - remaining

    sum_vector = _mm256_setzero_pd()

    for i in range(0, rem_iterator, 4):

        X_partial = _mm256_load_pd(&X[i])
        centers_partial = _mm256_load_pd(&centers_old[i])
        difference = _mm256_sub_pd(X_partial, centers_partial)
        difference = _mm256_mul_pd(difference, difference)
        sum_vector = _mm256_add_pd(sum_vector, difference)

    upper = _mm256_castpd256_pd128(sum_vector)        # [a0, a1]
    lower = _mm256_extractf128_pd(sum_vector, bit_lane)   # [a2, a3]
    result = _mm_add_pd(lower, upper)           # [(a0 + a2), (a1 + a3)]
    result = _mm_hadd_pd(result, result)        # [(a0 + a2) + (a1 + a3), (a0 + a2) + (a1 + a3)]
    distance = _mm_cvtsd_f64(result) 

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

        printf("Starting SIMD partial cluster summation")
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

    __m512d _mm512_add_pd(__m512 a, __m512 b)
    __m512d _mm512_sub_pd(__m512 a, __m512 b)
    __m512d _mm512_mul_pd(__m512 a, __m512 b)
    __m512d _mm512_div_pd(__m512 a, __m512 b)
    __m512d _mm512_fmadd_pd(__m512 a, __m512 b, __m512 c)
    __m512d _mm512_setzero_pd()


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
        int j, j_lock, cluster, feature, start, end, chunk, row_offset_lock, samples, byte_alignment = 64
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

            for j_lock in range(n_features):
                centers_new[cluster, j_lock] += centers_new_partial[row_offset_lock + j_lock]

            # if floating is float:
            #     simd_lock_partial_add_float(centers_new[row_offset_lock], &centers_new_partial[row_offset_lock], n_features)
            # else:
            #     simd_lock_partial_add_double(centers_new[row_offset_lock], &centers_new_partial[row_offset_lock], n_features)

        omp_unset_lock(&lock)

        free(centers_new_partial)
        free(weight_in_clusters_partial)
    
    omp_destroy_lock(&lock)


    ctypedef struct __m512:
        pass
    ctypedef struct __m512d:
        pass
    ctypedef struct __m512i:
        pass



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
        int j, j_lock, cluster, feature, start, end, chunk, row_offset_lock, samples, byte_alignment = 64
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

            # for j_lock in range(n_features):
            #    centers_new[cluster, j_lock] += centers_new_partial[row_offset_lock + j_lock]

            if floating is float:
                simd_lock_partial_add_float(centers_new[cluster], &centers_new_partial[row_offset_lock], n_features)
            else:
                simd_lock_partial_add_double(centers_new[cluster], &centers_new_partial[row_offset_lock], n_features)

        omp_unset_lock(&lock)

        free(centers_new_partial)
        free(weight_in_clusters_partial)
    
    omp_destroy_lock(&lock)


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
