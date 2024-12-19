from inspect import getmembers, isfunction

import test_seq_dist, test_topk, test_inference


def discover_tests(module):
    """
    Basic test discovery function
    """
    fns = getmembers(module, isfunction)
    return [fn for fn in fns if fn[0].startswith('test')]


def run_test_fn(fn):
    passed = 0
    try:
        fn[1]()
        print("- Pass: ", fn[0])
        passed += 1
    except:
        print("x FAIL: ", fn[0])
    return passed


def run_tests(fn_set):
    """
    Execute a list of test functions
    """
    passed = 0
    for fn in fn_set:
        passed += run_test_fn(fn)
    print(f"{passed}/{len(fn_set)} tests passed")


if __name__ == '__main__':
    test_sets = [test_seq_dist, test_topk, test_inference]

    for test_set in test_sets:
        print("="*50)
        print(f"Running tests in {test_set.__name__}")
        print("="*50)

        run_tests(discover_tests(test_set))
