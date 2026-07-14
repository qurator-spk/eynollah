import os
import sys
import more_itertools
import numpy as np
import argparse

parser = argparse.ArgumentParser(
                    prog='plot',
                    description='Print metrics',
                    epilog='...')
parser.add_argument('stem1', default='nohup.out.eynollah-206-noautosize-keepseps-refactoring9-crop-after2')
parser.add_argument('stem2', default='eval-gt2-onlyfg-pixel-f1-v0.5-206-noautosize-keepseps-refactoring9-crop-after2')

def main():
    args = parser.parse_args()

    prefix = os.path.dirname(sys.argv[0])
    files = [path for path in os.listdir(prefix)
             if path.startswith(args.stem1)
             and path.endswith('.txt')]

    for path in files:
        name = path[len(args.stem1):]
        metric = name.split('.')[-2]
        series = name[:-(len(metric) + 5)]
        path = os.path.join(prefix, path)

        if metric == 'peak-vram':
            cat, peak = np.loadtxt(path, unpack=True, dtype=np.dtype([("category", "U15"), ("peak", float)]))
            peak //= 1024
            peak //= 1024
            print(metric, f"{series: <52}", dict(zip(cat.tolist(), peak.tolist())))
        else:
            val = np.loadtxt(path)
            print(metric, f"{series: <52}", {'µ': float(np.round(np.mean(val), 2)),
                                             'M': float(np.round(np.median(val), 2)),
                                             'min': float(np.min(val)),
                                             'max': float(np.max(val))})


    files = [path for path in os.listdir(prefix)
             if path.startswith(args.stem2)
             and path.endswith('.log')]


    for path in files:
        name = path[len(args.stem2):]
        series = name[:-4]
        path = os.path.join(prefix, path)
        with open(path, 'r') as fd:
            lines = fd.readlines()
        result_v0_5 = {}
        result_v206 = {}
        for metric, v0_5, v206 in more_itertools.chunked(lines, n=3, strict=True):
            metric = metric[len(prefix) + 1:-1]
            result_v0_5[metric] = float(np.round(float(v0_5.strip()), 3))
            result_v206[metric] = float(np.round(float(v206.strip()), 3))
        print("pixel-f1", f"{series: <52}", result_v206)

    print("pixel-f1", "v0.5                                       ", result_v0_5)


