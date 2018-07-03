# This file contains some basic sorting algorithms implemented in Python. Just for fun.
#
# Author: Sen Jia
 
import pdb
 
def bubble_sort(lst):
    """
    Bubble sort is the most basic sorting algorithm, which is very inefficient especially when the list is in reverse order. 
    Its time complexity is O(n^2) for the worst-case and average. The space complexity is O(1). (Stable)
    """
    assert (lst),"Input list is empty!"
    length = len(lst)
    if length == 1: return lst
    for i in range(0,length):
        for j in range(i+1,length):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
    return lst
 
def cocktail_sort(lst):
    """
    Aka bidirectional bubble sort, stable.
    Time complexity for the worst case is O(n^2), O(n) for average. Space complexity O(1). 
    """
    beginIdx = 0
    endIdx = len(lst)-1
    while beginIdx <= endIdx:
        newBeginIdx = endIdx
        newEndIdx = beginIdx
        for i in range(beginIdx,endIdx):
            if lst[i] > lst[i+1]:
                lst[i], lst[i+1] = lst[i+1], lst[i]
                newEndIdx = i
        endIdx = newEndIdx  
        for i in range(endIdx,beginIdx-1,-1):
            if lst[i] > lst[i+1]:
                lst[i], lst[i+1] = lst[i+1], lst[i]
                newBeginIdx = i
        beginIdx = newBeginIdx
    return lst
 
def odd_even_sort(lst):
    """
    Another variant of bubble sort, designed for multi-threading.
    Processing odd-indexed and even-indexed element simultaneously.
    Time complexity depends on the number of threads. Space complexity O(1). Stable.
    """
    lst_sorted = False
    while not lst_sorted:
        lst_sorted = True
        for i in range(0, len(lst)-1,2):
            if lst[i] > lst[i+1]:
                lst[i],lst[i+1] = lst[i+1],lst[i]
                lst_sorted = False
        for i in range(1, len(lst)-1,2):
            if lst[i] > lst[i+1]:
                lst[i],lst[i+1] = lst[i+1],lst[i]
                lst_sorted = False
 
def quick_sort(lst): 
    """
    Quick sort algorithm. This implementation costs space complexity O(n)
    """
    less, greater = [], []
    if len(lst) <= 1: 
        return lst
    else:
        pivot = lst[-1] # select the last element of the list as the pivot point.
        for elem in lst[:-1]:
            if elem < pivot:
                less.append(elem)
            else:
                greater.append(elem)
    return quick_sort(less) + [pivot] + quick_sort(greater)
 
def quick_sort_inplace_recursive(lst, start, end):
    """
    In-place implementation of quick sort algorithm. Time complexity: O(n^2) for the worst and O(nLogn) for average. 
    Space complexity O(Logn). (Unstable)
    """
    if len(lst) <= 1 or start >= end: 
        return
    pivot = lst[end]
    right = end-1
    left = start
    while left < right:
        while lst[left] < pivot and left < right:
            left += 1
        while lst[right] >= pivot and left < right:
            right -= 1
 
        lst[left], lst[right] = lst[right], lst[left]
 
    if lst[left] >= lst[end]:
        lst[left], lst[end] = lst[end], lst[left]
    else:
        left += 1
    quick_sort_inplace_recursive(lst, start, left-1)
    quick_sort_inplace_recursive(lst, left+1, end)
 
 
def selection_sort(lst):
    """
    Time complexity O(n^2). Space complexity O(1).Unstable(but there is a stable variant.)
    """
    for i in range(len(lst)):
        minimum = lst[i]
        min_index = i
        for j in range(i+1, len(lst)):
            if minimum > lst[j]:
                minimum = lst[j]
                min_index = j
        lst[i],lst[min_index] = lst[min_index], lst[i]
 
def stable_selection_sort(lst):
    """
    Time complexity O(n). Space complexity O(1). 
    This is a stable implementation of selection sort using insertion rather than swapping.
    """
    for i in range(len(lst)):
        minimum = lst[i]
        min_index = i
        for j in range(i+1,len(lst)):
            if minimum > lst[j]:
                minimum = lst[j]
                min_index = j
        del lst[min_index]
        lst.insert(i,minimum)
 
def counting_sort(lst):
    """
    Counting sort is prefered when there are many duplicate elements in a list.Stable.
    Time complexity O(n+k), space complexity O(n+k), where k is the range of elements in a list. 
    This implementation only works for positive integer.
    """
    LOOKUP={}
    k=0
    for x in lst:
        LOOKUP.setdefault(x,0)+1
        LOOKUP[x] += 1
        if x > k:
            k = x
    total=0
    for i in range(k+1):
        try:
            oldCount = LOOKUP[i]
        except: pass
        else:
            LOOKUP[i] = total
            total += oldCount
    ans=[0]*len(lst)
    for x in lst:
        ans[LOOKUP[x]] = x
        LOOKUP[x] += 1
    return ans
 
def insertion_sort(lst):
    """
    Time complexity O(n^2). Space complexity O(1).Stable.
    Addition: the original insertion sort is implemented by swapping 
    two adjacent elements to pop up the current element.(Like the bubble)
    So, there is 1/2(n(n-1)) comparison operations and n-1 assignment operations. That's why O(n^2).
     
    During the sorting process, there is a sublist of sorted elements during sorting (first K)
    Therefore Binary Search can be used to efficiently find out the right place to insert.
     
    If binary search is not applied, a cursor needs to swipe back to insert the element.
    Linked list can be used to save the time for swapping.
    """
    for i in range(1,len(lst)):
        for j in range(i,0,-1):
            if lst[j] < lst[j-1]:
                lst[j], lst[j-1] = lst[j-1], lst[j]
 
def linked_list_insertion_sort(lst):
    """
    Time complexity O(n), linked list required. Space complexity O(1).Stable.
    Python built-in list is array, pretending we have linked list here.
    """
    for i in range(1,len(lst)):
        temp = lst[i]
        lst.pop(i)     # delete the current node from the list (I know it is not linked list here.. )
        j = i - 1
        while j > 0 and lst[j] > temp:
            j -= 1
        lst.insert(j,temp) # insert a node for the current element to the right place.
 
def gnome_sort(lst):
    """
    Aka stupid sort, very similar to insertion sort. 
    Gnome sort is based on swapping not insertion.
    Time complexity O(n^2). Space complexity O(1).Stable.
    Comparing with insertion sort, after poping up an element, 
    it requires several comparisons to move back to the cursor location.
    """
    pos = 0
    while pos < len(lst):
        if pos == 0 or lst[pos] >= lst[pos-1]:
            pos += 1
        else:
            lst[pos], lst[pos-1] = lst[pos-1], lst[pos]
            pos -= 1
 
def shell_sort(lst):
    """
    Insertion sort needs to move the ith element by comparing its neighbourhoods and exchanging the place.
    It may take a long time to insert an element at the far end to its right place.
    Shell sort uses a gap to exchange two elements distant from each other.
    How to choose the gap is still an open question (feels like sample rate?)
    Average time complexity depends on the chosen gap. Worst time complexity O(nLog^2n). Space O(1). Unstable.
    """
    N = len(lst)
    gaps = [int(N/(2**x)) for x in range(1,N) if 2**x < N]   # x < Log2(N)
    for gap in gaps:
        for i in range(gap, N):
            temp = lst[i]
            j = i
            while j >= gap and lst[j - gap] > temp:
                lst[j] = lst[j - gap]
                j -= gap
            lst[j] = temp

def comb_sort(lst):
    """
    Similar to shellsort, combsort can be used as bubble sort with gaps.
    Time complexity O(n^2). Space complexity O(1). Unstable.
    """ 
    gap = len(lst)
    shrink_factor = 0.8
    swapped = True
    while gap > 1 or swapped:
        if gap > 1:
            gap *= shrink_factor
            gap = int(gap)
        i = 0
        swapped = False
        while i + gap < len(lst):
            if lst[i] > lst[i+gap]:
                lst[i], lst[i+gap] = lst[i+gap], lst[i]
                swapped = True
            i += 1

def stooge_sort(lst, start, length):
    i = start
    j = length
    """
    An inefficient recursive sorting algorithm.
    Sorting first 2/3, then last 2/3, then first 2/3.....
    Time: O(n^(Log3/Log1.5)). Space: O(n). Unstable.
    """
    if lst[j] < lst[i]:
        lst[i], lst[j] = lst[j], lst[i]
    if j - i + 1 > 2:
        t = int((j - i + 1) / 3)
        stooge_sort(lst, i, j-t)
        stooge_sort(lst, i+t, j)
        stooge_sort(lst, i, j-t)
    #return lst

def cycle_sort(lst):
    """
    In-place sorting algorithm. Take an item and serach the whole list to find out
    how many elements are smaller than the current item. Then put the current item to
    the right place and take the replaced item as current.
    Time O(n^2). Space O(1). Unstable
    """

    for start in range(len(lst)-1):
        item = lst[start]
        pos = start
        for i in range(start+1, len(lst)):
            if lst[i] < item:
                pos += 1
        if pos == start:
            continue
        while item == lst[pos]:
            pos += 1
        lst[pos], item = item, lst[pos]

        while pos != start:
            pos = start
            for i in range(start+1, len(lst)):
                if lst[i] < item:
                    pos += 1
            while item == lst[pos]:
                pos += 1
            lst[pos], item = item, lst[pos]


def merge_sort_recursive(lst):
    """
    A divide and conquer sorting algorithm. Recursively divide a whole list into sublists until the
    total lenght is 1. Then merge two sublists into one sorted based on comparison.
    Time complexity O(nLog(n)). Space complexity O(n). Stable.
    This recursive implementation is based on the Wiki examplar code without using deque.
    """
    if len(lst) <= 1:
        return lst

    def merge(left, right):
        merged = []
        while left and right:
            merged.append(left.pop(0) if left[0] < right[0] else right.pop(0))
        merged.extend(left if left else right)
        return merged

    middle = int(len(lst) // 2)
    left = merge_sort_recursive(lst[:middle])
    right = merge_sort_recursive(lst[middle:])
    return merge(left, right)


def main():
    input_lst = list(range(50, 0, -1))
    #lst = merge_sort_recursive(input_lst)
    quick_sort_inplace_recursive(input_lst, 0, len(input_lst)-1)
    print (input_lst)

if __name__ == "__main__": main()
