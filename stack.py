class stack_list():
    def __init__(self, nums=[]):
        self._array = nums

    def is_Empty(self):
        if  not self._array:
            return True
        else:
            return False

    def length(self):
        return len(self._array)

    def top(self):
        return self._array[-1]

    def push(self, num):
        self._array.append(num)

    def pop(self):
        return self._array.pop()

    def travel(self):
        print(self._array)

