from functools import partial
from multiprocessing import Pool
from subprocess import run
from tests.utils import config_file, decode

def test_main():
    expected_usage = 'Usage: python3 -m text_categorizer --trainer <configuration file>\n       python3 -m text_categorizer --prediction_server <configuration file> <port>\n'
    args = ['python', '-m', 'text_categorizer']
    all_args = []
    for i in range(1, 6):
        all_args.append(args)
        args.append(str(i))
    all_args.append(['python', '-m', 'text_categorizer', '--trainer', config_file])
    with Pool() as pool:
        cps = pool.map(func=partial(run, capture_output=True), iterable=all_args)
        assert all([cp.returncode == 0 for cp in cps])
        cps = cps[0:-1]
        assert all([decode(cp.stdout) == expected_usage for cp in cps])
        assert all([decode(cp.stderr) == '' for cp in cps])
