"""Зводить логи бенчмарку в таблицю Markdown: час, пікова памʼять, пікова потужність, NCCL busbw."""
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
    marks.setdefault(tag, {})[kind] = datetime.fromisoformat(ts)

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

def peaks(tag):
    m = marks.get(tag, {})
    if 'START' not in m or 'END' not in m:
        return None, None
    start, end = m['START'].replace(tzinfo=None), m['END'].replace(tzinfo=None)
    win = [s for s in samples if start <= s[0] <= end]
    if not win:
        return None, None
    return max(s[2] for s in win) / 1024, max(s[3] for s in win)

busbw = None
nccl = logs / 'nccl_all_reduce_2gpu.log'
if nccl.exists():
    m = re.search(r'Avg bus bandwidth\s*:\s*([\d.]+)', nccl.read_text())
    busbw = float(m.group(1)) if m else None

print('| Запуск | Час, с | Пікова памʼять на GPU, ГБ | Пікова потужність на GPU, Вт |')
print('|---|---|---|---|')
for tag in ('wan_1gpu', 'wan_2gpu'):
    if tag in wall:
        mem, pw = peaks(tag)
        print(f"| {tag} | {wall[tag]} | {mem:.1f} | {pw:.0f} |" if mem else f"| {tag} | {wall[tag]} | н/д | н/д |")
if 'wan_1gpu' in wall and 'wan_2gpu' in wall:
    print(f"\nПрискорення на 2 GPU: {wall['wan_1gpu'] / wall['wan_2gpu']:.2f}×")
if busbw is not None:
    print(f"NCCL all_reduce, середня bus bandwidth: {busbw:.1f} ГБ/с")
