def is_prime(n):
  """Checks if a number is prime."""
  if n <= 1:
    return False
  for i in range(2, int(n**0.5) + 1):
    if n % i == 0:
      return False
  return True

def count_primes(limit):
  """Counts the number of prime numbers up to a given limit."""
  count = 0
  for i in range(2, limit + 1):
    if is_prime(i):
      count += 1
  print(f"Number of prime numbers up to {limit}: {count}")
  return count

count_primes(10000)