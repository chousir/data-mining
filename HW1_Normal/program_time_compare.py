import time
import os

# Apriori
start_time_a = time.time()
os.system('python Apriori.py')
end_time_a = time.time()
execution_time_a = end_time_a - start_time_a
print(f"Apriori: {execution_time_a:.4f} seconds")

# FP_Growth
start_time_b = time.time()
os.system('python FP_Growth.py')
end_time_b = time.time()
execution_time_b = end_time_b - start_time_b
print(f"FP_Growth: {execution_time_b:.4f} seconds")
