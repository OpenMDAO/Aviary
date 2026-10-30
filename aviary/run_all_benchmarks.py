import subprocess

process = subprocess.Popen(
    ['testflo', '--nocapture', '--testmatch=bench_test*'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

lines = []
for line in process.stdout:
    print(line, end='', flush=True)
    lines.append(line)

# Wait for the subprocess to finish and get the exit code
return_code = process.wait()

print('\n\n')
print('Benchmark Results')
print('\n')

j = 0
results = {}
for j, line in enumerate(lines):
    if 'BENCH:' in line:
        line = line.partition('BENCH: ')[-1]
        if line in results:
            raise RuntimeError(f'Use a unique name for test {line}!')
        results[line] = lines[j + 1]

for name, bench_data in sorted(results.items()):
    print(name)
    print(bench_data)
