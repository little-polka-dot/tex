import concurrent.futures
import types
import time
import random
import asyncio
import tornado
import os


def thread_method():
    if '__sleep__' not in globals():
        globals()['__sleep__'] = random.randint(1, 5)
        print(f'init({globals()["__sleep__"]})', os.getpid(), os.getppid())
    time.sleep(globals()['__sleep__'])
    print(__name__, globals()['__sleep__'], os.getpid(), os.getppid())
    return f'sleep({globals()["__sleep__"]})'


async def main():
    lp = asyncio.get_event_loop()
    print(id(lp))
    return await lp.run_in_executor(pool, thread_method)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')  # cuda要求的进程启动方式
    print(os.cpu_count(), os.getpid())
    pool = concurrent.futures.ProcessPoolExecutor(2)

    work = asyncio.wait([main(), main(), main(), main()])

    lp = asyncio.get_event_loop()
    print(id(lp))
    print(lp.run_until_complete(work))



