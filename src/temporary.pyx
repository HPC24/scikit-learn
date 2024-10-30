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