from typing import List

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        dic = {0:0, 1:0, 2:0} # {0:2, 1:2, 2:2}
        for i in nums:
            if i == 0:
                dic[0] += 1
            elif i == 1:
                dic[1] += 1
            elif i == 2:
                dic[2] += 1
        for i in range(len(nums)): # inplace
            if i < dic[0]:
                nums[i] = 0
            elif i < dic[1]+dic[0]:
                nums[i] = 1
            elif i < dic[2]+dic[1]+dic[0]:
                nums[i] = 2

def main():
    # Initialize the input list nums
    nums = [2, 0, 2, 1, 1, 0]  # Example input
    print("Original list:", nums)

    # Create an instance of the Solution class
    solution = Solution()

    # Call the sortColors method to sort the list in-place
    solution.sortColors(nums)

    # Print the sorted list
    print("Sorted list:", nums)

if __name__ == "__main__":
    main()
