"""
    reference: https://www.analyticsvidhya.com/blog/2021/08/explore-the-magic-methods-in-python/#:~:text=__getitem__%20method%20is,like%20list%2C%20tuple%2C%20etc.&text=__setitem__%20method%20is,is%20also%20used%20with%20containers.
    python implicitly invokes the magic methods to provide additional functionality to it.
"""


# __getitem__
class A:
    def __init__(self, item):
        self.item = item

    def __getitem__(self, index):
        return self.item[index]


a = A([1, 2, 3])
print(f"First item: {a[0]}")
print(f"Second item: {a[1]}")


# __len__
class LenExample:
    def __init__(self, item):
        self.item = item

    def __len__(self):
        return len(self.item)


# When we invoke the len() method the length of the list named item is returned that is defined inside the __len__ method.
len_instance = LenExample([1, 2, 3])
print(len(len_instance))


# Output: 3

# __call__
# __call__ magic method is invoked when the instance of a class is invoked. Instead of writing another method to
# perform certain operations, we can use the __call__ method to directly call from the instance name.
class CallExample:
    def __init__(self, val):
        self.val = val

    def __call__(self, b):
        return self.val * b


call_example = CallExample(5)
print(call_example(6))


# __iter__
# __iter__ method is used to provide a generator object for the provided instance.
# We can make use of iter() and next() method to leverage __iter__ method.
class Squares:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __iter__(self):
        for value in range(self.start, self.stop + 1):
            yield value ** 2


i = iter(Squares(1, 3))
print(next(i))
print(next(i))
print(next(i))
