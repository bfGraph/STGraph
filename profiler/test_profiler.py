import cProfile

def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n-1)

cProfile.run('factorial(500)')