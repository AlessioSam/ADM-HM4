import pandas as pd
import math
import numpy as np
import time
import json

# Different implementation
def hash_func_1(password):
    n = len(password)
    # As a start we will set error rate to 0.02 so we can calculate the m
    p = 0.02
    m = round(-(n * math.log(p)) / (math.log(2) ** 2))
    hashed_1 = 0
    for i in range(len(password)):
        hashed_1 += ord(password[i])
    return (hashed_1 % m)  # first hashing function which thake the ord of each char

def hash_func_2(password):
    n = len(password)
    # As a start we will set error rate to 0.02 so we can calculate the m
    p = 0.02
    m = round(-(n * math.log(p)) / (math.log(2) ** 2))
    hashed_2 = 0
    for i in range(len(password)):
        hashed_2 += hashed_2 + (
            ord(password[i])) ** 2  # Second hashing function which thake the ord of each char power 2
    return (hashed_2 % m)


def bloom_filter(password1, password2):
    n = len(password1)
    # As a start we will set error rate to 0.02 so we can calculate the m
    p = 0.02
    m = round(-(n * math.log(p)) / (math.log(2) ** 2))
    bloomTable = ['' for i in range(m)]  # The bloom table which has the size of m filled with empty space

    k = round((m / n) * math.log(2))
    start = time.time()
    duplicates_counter = 0
    not_duplicates = 0

    for j in password1:
        for u in j:
            pass1_1 = hash_func_1(u)
            pass1_2 = hash_func_2(u)
        # Fill the corresponding index in the bloom filter table with one
        if bloomTable[pass1_1] != 1:
            bloomTable[pass1_1] = 1
        if bloomTable[pass1_2] != 1:
            bloomTable[pass1_2] = 1

            # Now we work on the second data set
    for i in password2:
        for y in i:
            pass2_1 = hash_func_1(y)
            pass2_2 = hash_func_2(y)

        if bloomTable[pass2_1] == 1 and bloomTable[
            pass2_2] == 1:  # iF the bloom table index already filled with 1 meaning the hashed value of password1 occupied it
            duplicates_counter += 1
        else:
            not_duplicates += 1

    end = time.time()

    print('Number of hash function used: ',
          k)  # K form the bloom filter formula the nneeded hashing function ,but we used two hashing functions
    # print('Number tot pass ', len(password1), len(password2))
    print('Not_duplicates ', not_duplicates)
    print('Number of duplicates detected: ', duplicates_counter)
    print('Probability of false positives: ', p)
    print('Execution time: ', end - start)

