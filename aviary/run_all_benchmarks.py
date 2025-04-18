import subprocess

status = subprocess.run(['testflo', '--testmatch=bench_test*'])
if status.returncode == 1:
    # disable mpi and try again - can cause PermissionError for "forbidden sockets"
    print('Re-running all benchmark tests with MPI disabled')
    subprocess.run(['testflo', '--testmatch=bench_test*', '--nompi'])
