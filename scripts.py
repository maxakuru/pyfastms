import subprocess


def test():
    subprocess.run(
        ['python', '-m', 'tests.test_basic']
    )
