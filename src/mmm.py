import multiprocessing
from time import sleep

def worker(procnum, return_dict, semaphore):
    """worker function"""
    with semaphore:
        print(str(procnum) + " represent!")
        sleep(3)
        return_dict[procnum] = procnum


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    semaphore = multiprocessing.Semaphore(2)
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i, return_dict, semaphore))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(return_dict.values())