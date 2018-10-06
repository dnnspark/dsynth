import multiprocessing as mp
import functools
import sys

def func1(x):
    return x**2

func3 = lambda x: x**2

def func4(x,y):
    return x + y

def gen1(c):
    for x in range(1000):
        yield x+c

def test_map_simple():

    pool = mp.Pool(4)
    x = [1,2,3,4]
    y = pool.map(func1, x)
    print(y)

# AttributeError: Can't pickle local object 'test_map_with_nested_function.<locals>.func2'
# def test_map_with_nested_function():

#     def func2(x):
#         return x**2

#     pool = mp.Pool(4)
#     x = [1,2,3,4]
#     y = pool.map(func2, x)
#     print(y)

# _pickle.PicklingError: Can't pickle <function <lambda> at 0x10bd67e18>: attribute lookup <lambda> on __main__ failed
# def test_map_with_lambda():

#     pool = mp.Pool(4)
#     x = [1,2,3,4]
#     y = pool.map(func3, x)
#     print(y)

def test_map_with_partial():

    foo = functools.partial(func4, y=10)
    pool = mp.Pool(4)
    x = [1,2,3,4]
    y = pool.map(foo, x)
    print(y)


def test_map_with_generator():

    g = gen1(1)

    pool = mp.Pool(4)
    y = pool.map(func1, g, chunksize=2)
    print(y)

def test_starmap_with_zipped_generators():

    # pool.starmap is available since python 3.3
    if not sys.version_info >= (3,3):
        return

    g1 = gen1(1)
    g2 = gen1(2)

    pool = mp.Pool(4)
    # y = pool.map(func4, zip(g1,g2))
    y = pool.starmap(func4, zip(g1,g2), chunksize=250) # python >= 3.3
    print(y)


# if __name__ == '__main__':
#     test_map_simple()
#     # test_map_with_nested_function()
#     # test_map_with_lambda()
#     test_map_with_partial()
#     test_map_with_generator()
#     test_starmap_with_zipped_generators()
