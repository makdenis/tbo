import random
import string
import multiprocessing as mp, os
import time
from collections import Counter
from functools import reduce
from datetime import datetime


def gen_chunk(filename):
    tmp =[]
    for i in range(1024):
        tmp.append(
            f"{''.join(random.choice(string.ascii_letters) for _ in range(random.randint(30, 64)))}\n")

    to_write = ''.join(tmp)
    with open(filename, 'a+', encoding='utf-8') as file:
        file.write(to_write)


def gen_file(filename, size):
    random.seed(datetime.now())
    if not os.path.exists(filename):
        open(filename, 'w').close()
    while os.path.getsize(filename) / (1024 * 1024 * size) < 1:
        gen_chunk(filename)


def process_wrapper(chunkStart, chunkSize, filename):
    with open(filename) as f:
        f.seek(chunkStart)
        lines = f.read(chunkSize).splitlines()
        c = Counter(lines)
        return c


def reducer(cnt1, cnt2):
    cnt1.update(cnt2)
    return cnt1


def sort(item):
    return dict(sorted(item.items(), key=lambda item: item[1]))


def chunkify(filename, size=1024 * 1024):
    fileEnd = os.path.getsize(filename)
    with open(filename, 'rb') as f:
        chunkEnd = f.tell()
        while True:
            chunkStart = chunkEnd
            f.seek(size, 1)
            f.readline()
            chunkEnd = f.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd > fileEnd:
                break


if __name__ == '__main__':
    filename = "lab.txt"
    # gen_file(filename, 1024*3)
    t0 = time.time()
    pool = mp.Pool(4)
    jobs = []

    for chunkStart, chunkSize in chunkify(filename):
        jobs.append(pool.apply_async(process_wrapper, (chunkStart, chunkSize, filename)))
    result = []
    for job in jobs:
        result.append(job.get())
    pool.close()
    reduced = reduce(reducer, result)
    result = sort(reduced)
    t1 = time.time()
    total = t1 - t0
    print(list(result.items())[:100])
    print(total)
