import numpy as np
import fastms


def simple():
    arr = np.zeros((100, 100, 3), dtype=np.float64)
    s = fastms.FMSSolver()
    res = s.run(arr)
    print('res: ', res)


def params():
    p = fastms.Parameters()
    print('p: ', p)


if __name__ == "__main__":
    simple()
    # test_errors()
