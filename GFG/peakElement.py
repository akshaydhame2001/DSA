class Solution:   
    def peakElement(self,arr):
        n = len(arr)
        # Code here
        if n == 0:
            return -1  # No peak in an empty array
        
        # Check the first element
        if n == 1 or arr[0] > arr[1]:
            return 0
        
        # Check the last element
        if arr[n-1] > arr[n-2]:
            return n-1
        
        for i in range(1, n-1):
            if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                return i
        return -1