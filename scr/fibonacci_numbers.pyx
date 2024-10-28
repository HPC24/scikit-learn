def n_fibonacci_numbers(int n):


    if n > 1000:
        n = 1000

    cdef int[1000] fibonacci
    cdef int a = 0
    cdef int b = 1
    cdef counter = 1

    fibonacci[0] = 0
    while counter < n:
        a, b = b, a+b
        fibonacci[counter] = a 
        counter += 1 

    fibonacci_list = [i for i in fibonacci[:n]]
    return fibonacci_list


