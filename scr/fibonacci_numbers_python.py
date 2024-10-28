def n_fibonacci_numbers(n) -> list:
    
    fibonacci_numbers = [0]
    
    a = 0
    b = 1
    counter = 1
    
    while counter < n:

        a, b = b, a+b
        fibonacci_numbers.append(a)
        counter += 1
        
    return fibonacci_numbers
        