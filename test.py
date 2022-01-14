def binary_search(nums, query):
    lo = 0
    hi = len(nums) -1 
    while lo<=hi:
        mid = (lo+hi)//2
        
        if mid>0 and nums[mid -1]> nums[mid]:
            if nums[mid] == query:
                return mid
            elif nums[len(nums)-1] > query:
                lo = mid; hi = len(nums) -1
                while lo<=hi:
                    mid1 = (lo+hi)//2
                    if nums[mid1] == query:
                        return mid1
                    elif nums[mid1] < query:
                        lo = mid1+1
                    else:
                        hi = mid1-1
            else:
                lo = 0; hi = mid-1
                while lo<=hi:
                    mid1 = (lo+hi)//2
                    if nums[mid1] == query:
                        return mid1
                    elif nums[mid1] < query:
                        lo = mid1+1
                    else:
                        hi = mid1-1
                
        elif nums[mid] > nums[hi]:
            lo = mid+1
        else:
            hi = mid-1
            
nums = [4,5,6,7,0,1,2]; query = 2

print(binary_search(nums, query))