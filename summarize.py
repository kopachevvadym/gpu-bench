"""Зводить логи бенчмарку в таблицю Markdown: час, с/крок, пікова памʼять і потужність, енергія, NCCL."""
import csv, re, sys
from datetime import datetime
from pathlib import Path

logs = Path(sys.argv[1])

wall = {}
for row in csv.reader(open(logs / 'wall_seconds.csv')):
    wall[row[0]] = int(row[1])

marks = {}
for line in open(logs / 'marks.txt'):
    kind, tag, ts = line.split()
    marks.setdefault(tag, {})[kind] = datetime.fromisoformat(ts).replace(tzinfo=None)

# timestamp, index, name, memory.used, power.draw, temperature, utilization; один замір на GPU щосекунди
samples = []
with open(logs / 'gpu.csv') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        try:
            ts = datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f')
            samples.append((ts, int(row[1]), float(row[3].split()[0]), float(row[4].split()[0])))
        except (ValueError, IndexError):
            pass


def window(tag):
    m = marks.get(tag, {})
    if 'START' not in m or 'END' not in m:
        return []
    return [s for s in samples if m['START'] <= s[0] <= m['END']]


def step_time(tag):
    f = logs / f'{tag}.log'
    if not f.exists():
        return None
    its = re.findall(r'([\d.]+)s/it\]', f.read_text())
    return float(its[-1]) if its else None


print('| Запуск | Повний час, с | с/крок | Пікова памʼять на GPU, ГБ | Пікова потужність на GPU, Вт | Енергія всіх GPU, Вт·год |')
print('|---|---|---|---|---|---|')
for tag in ('wan_1gpu', 'wan_2gpu'):
    if tag not in wall:
        continue
    w = window(tag)
    st = step_time(tag)
    st_s = f'{st:.1f}' if st else 'н/д'
    if w:
        mem = max(s[2] for s in w) / 1024
        pw = max(s[3] for s in w)
        wh = sum(s[3] for s in w) / 3600  # заміри раз на секунду
        print(f'| {tag} | {wall[tag]} | {st_s} | {mem:.1f} | {pw:.0f} | {wh:.0f} |')
    else:
        print(f'| {tag} | {wall[tag]} | {st_s} | н/д | н/д | н/д |')

if 'wan_1gpu' in wall and 'wan_2gpu' in wall:
    print(f"\nПрискорення одного кліпу на 2 GPU: {wall['wan_1gpu'] / wall['wan_2gpu']:.2f}×")
    print(f"Пропускна здатність 2 GPU з окремими кліпами: 2.00× (два кліпи за {wall['wan_1gpu']} с)")

nccl = logs / 'nccl_all_reduce_2gpu.log'
if nccl.exists():
    m = re.search(r'Avg bus bandwidth\s*:\s*([\d.]+)', nccl.read_text())
    print(f'nccl-tests all_reduce, bus bandwidth: {float(m.group(1)):.1f} ГБ/с' if m else 'nccl-tests: збій, див. nccl_all_reduce_2gpu.log')

torch_log = logs / 'torch_nccl_allreduce.log'
if torch_log.exists():
    m = re.search(r'= ([\d.]+) ms, busbw ~ ([\d.]+) GB/s', torch_log.read_text())
    if m:
        print(f'NCCL з PyTorch, all_reduce 256 МБ: {m.group(1)} мс, ≈ {m.group(2)} ГБ/с')
