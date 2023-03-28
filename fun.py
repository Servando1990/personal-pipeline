from click import option
from numpy import sort

from typing import List

lst = [1, 2, 3, 4, 5]
lst_sq = [0, 2, 4, 6, 8]
nested_lst = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
Input = [1, 0, 1, 0, 1, 0, 0, 1]


def sum_odd(lst):
    odd_sum = 0

    for num in lst:
        if num % 2 != 0:
            odd_sum += num
    return print(odd_sum)


def sum_odd_lambda(lst):
    return print(sum(filter(lambda x: x % 2 != 0, lst)))


def reversed_sorted_list(lst: List):
    lst.reverse()
    lst_1 = lst[::-1]
    return print("first_option", lst_1, "second option", lst)


def flatten_list(lst):
    return print([element for sublist in lst for element in sublist])


def sun_largest_squares(lst, n):
    sorted_list = sorted(lst, reverse=True)
    sorted_list_squares = sorted(lst, key=lambda x: x**2, reverse=True)

    return print(sum(sorted_list_squares[:n]))


# Define a simple decorator
def my_decorator(func):
    def wrapper():
        print("Before the function is called.")
        func()
        print("After the function is called.")

    return wrapper


# Define a function to be decorated
def my_function():
    print("This is my function.")


# Define a list of numbers
numbers = [1, 2, 3, 4, 5]

# Use filter() with a lambda function to keep only the even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))  # Returns [2, 4]

# Define a list of numbers
numbers = [1, 2, 3, 4, 5]

# Use map() with a lambda function to square each number
squares = list(map(lambda x: x**2, numbers))  # Returns [1, 4, 9, 16, 25]
my_function = my_decorator(my_function)


def main():
    sum_odd(lst)
    sum_odd_lambda(lst)
    reversed_sorted_list(lst)
    flatten_list(nested_lst)
    sun_largest_squares(lst_sq, 2)
    # Decorate the function with the decorator

    # Call the decorated function
    my_function()


if __name__ == "__main__":
    main()
