import subprocess

returns = subprocess.run(
    ['testflo', '--nocapture', '--testmatch=bench_test*'],
    capture_output=subprocess.PIPE,
    text=True,
)

DEBUG = False
if DEBUG:
    print(returns.stdout)

lines = returns.stdout.split('\n')

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
    print('\n')

# Summary
print('\n')
print('Testflo Summary')
print('\n')

for line in lines[-11:]:
    print(line)
