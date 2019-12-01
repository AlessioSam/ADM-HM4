import matplotlib.pyplot as plt
import time
import numpy as np

def counting_sort(array):
    aux_array = [0]*1000 # make auxilary array of size 1000, we suppose the ordinal are between 0 and 999, however it can be any range
    order = [0]*len(array)
    # calculate the frequencies
    for i in array:
        aux_array[i] += 1
    # calculate the cumulative values of the frequencies
    for i in range(1,len(aux_array)):
        aux_array[i] += aux_array[i-1]
    sorted_array = [None]*len(array)
    # put tha values in array to sorted array on the right position and store the order
    for j,i in enumerate(array):
        order[aux_array[i]-1] = j
        sorted_array[aux_array[i]-1] = i
        aux_array[i] += -1
    return sorted_array, order


def counting_sort_alpha(arr):
    count = [0 for i in range(256)]  # fill counter with zeros 256 index because we are using ord
    alpha_ = ['' for _ in arr]  # to fill the ordered letters

    for a in arr:
        count[ord(a)] += 1  # count occurrence of each character and fill Corresponding ord index with letter counter
        # print(count)
        # print("==================================================")

    for i in range(len(count)):
        count[i] = count[i] + count[i - 1]  # put each letter in its actual Position the count[i]

        # print(count)
    # print("==================================================")

    for i in range(len(arr)):  # for each index number from above minus 1 put the letter in the index number
        count[count[ord(arr[i])] - 1] = arr[i]
        count[ord(arr[i])] -= 1

        # print(count)
    # print("==================================================")
    for i in range(len(arr)):
        alpha_[i] = count[i]
        # print(count[i] )
    return (print(alpha_, end=""))


def plot_count_sort():
    time_tot = list()
    # change m from 10 to 100000 with steps of 10000
    for m in range(10,100000,10000):
        time_ = list()
        let = np.random.randint(ord('A'), ord('z')+1, m)
        start = time.time()
        sorted_words_list = counting_sort(let) # same thing of counting_sort_alpha
        end_time = time.time()
        time_.append(end_time - start)
        time_tot.append(sum(time_))
   #PLOT
    fig = plt.figure(figsize=(16,4))
    plt.plot(range(10,100000,10000),time_tot)
    plt.xlabel('Letter lenght')
    plt.ylabel('Running time (secs)')
    plt.title('Running time for counting sort')


def create_matrix(words):
    m = max(map(len, words))
    n = len(words)
    arrays = np.zeros((n, m + 1), dtype=int)
    arrays[:, -1] = range(n)
    for i, word in enumerate(words):
        arrays[i, range(len(word))] = list(map(ord, word.lower()))

    return arrays

def make_group(array, index):
    problem_list = list()
    problem = [index[0]]
    for i in range(1,len(array)):
        if array[i] == array[i-1]:
            problem.append(index[i])
        else:
            problem_list.append(problem)
            problem = [index[i]]
    problem_list.append(problem)
    return problem_list


def sort_eq_col(arrays, i):
    if arrays.shape[0] == 1:
        return arrays
    else:
        # reorder equal columns (splitted by 'group')
        sorted_array, order = counting_sort(arrays[:, i])
        arrays = arrays[order, :]

        if (i + 1) < arrays.shape[1]:
            list_groups = make_group(sorted_array, list(range(arrays.shape[0])))
            list_array = list()
            for problem in list_groups:
                list_array.append(sort_eq_col(arrays[problem, :], i + 1))
            arrays = np.concatenate(list_array, axis=0)
        return arrays


def alpha_counting_sort():
    words = list(map(str, input("Enter words: ").split()))
    arrays = create_matrix(words)

    sorted_array = sort_eq_col(arrays, 0)
    order = list(map(int, sorted_array[:, -1]))
    return ([words[i] for i in order])

def alpha_counting_sortC(words):
    arrays = create_matrix(words)
    sorted_array = sort_eq_col(arrays, 0)
    order = list(map(int, sorted_array[:,-1]))
    return([words[i] for i in order])


def plot_alpha_sort():
    chr_vec = np.vectorize(chr)
    # we need to fix the number of words
    m = 50
    time_tot = list()
    # change n from 10 to 10000 with steps of 1000
    for n in range(10, 10000, 1000):
        time_ = list()

        ordinals_list = np.random.randint(ord('A'), ord('z') + 1, (m, n))
        chr_lists = chr_vec(ordinals_list)
        words_list = list()
        # convert ordinals to words
        for i in range(chr_lists.shape[0]):
            words_list.append("".join(chr_lists[i, :]))
        start_time = time.time()
        sorted_words_list = alpha_counting_sortC(words_list)
        end_time = time.time()
        time_.append(end_time - start_time)
        time_tot.append(sum(time_))

    # PLOT
    fig = plt.figure(figsize=(16, 4))
    plt.plot(range(10, 100000, 10000), time_tot)
    plt.xlabel('Number of words')
    plt.ylabel('Running time (secs)')
    plt.title('Running time for fixed word lenght')