import sys
import json
import tqdm
import multiprocessing
import os

class RunInfo:
    minLen = 500
    period = 100
    @staticmethod
    def check(tokens):
        size = len(tokens)
        return  size>= RunInfo.minLen and size < RunInfo.minLen + RunInfo.period

workerNum = 40

def processOne(task):
    infile, outfile = task
    print(outfile)
    with open(infile) as f, open(outfile, 'w') as fw:
        for line in tqdm.tqdm(f):
            x = json.loads(line)
            tokens = x['input'] + x['output']
            if RunInfo.check(tokens):
                continue
            fw.write(line)


infolder, outfolder = sys.argv[1:3]

for sizeMin in range(100, 600, 100):
    RunInfo.minLen= sizeMin
    tasks = []
    destFolder = os.path.join(outfolder, f"{sizeMin}")
    if os.path.exists(destFolder):
        import shutil
        shutil.rmtree(destFolder)
    os.makedirs(destFolder)
    print("group:", sizeMin, destFolder)
    for fn in os.listdir(infolder):
        tasks.append((
            os.path.join(infolder, fn),
            os.path.join(destFolder, fn)
        ))
    workers = multiprocessing.Pool(workerNum)
    workers.map(processOne, tasks)
    workers.close()
    workers.join()
