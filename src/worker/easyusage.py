
import sys
import select
import multiprocessing as mp

def stdin():
    """
    Reads lines from stdin pipe and yields results one-by-one.

    Yields:
        line: The next line from stdin

    Example:
        will yield a line for each txt file found in this folder
        find . -name "*.txt" | python file.py

        you can also read a file line by line
        cat filename.txt | python file.py

    """

    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:

        line = sys.stdin.readline()

        if not line:
            yield from []
            break

        line = line.strip()
        yield line


def parallel(lines, func, args, kwargs, procs=1):
    """

    Takes iterator (list or generator) `lines` and spawns # `procs` processes, calling
    `func` with prefined arguments `args` and `kwargs`.

    Using a queue and multiprocessing to call `func` with the format

    func(line, *args, **kwargs)

    Args:
        lines: iterator to be parallelised.
        func: function to call every line.
        args: Variable length argument list for `func`.
        kwargs: Arbitrary keyword arguments for `func`.
        procs: how many processes to start.

    Returns:
        results: List of all results from the parallel call (random order).

    """

    # Start a queue with the size of processes for jobs and a result queue to
    # collect results
    q_res = mp.Queue()
    q_job = mp.Queue(maxsize=procs)

    # print lock
    iolock = mp.Lock()

    # Start the pool and await queue data
    pool = mp.Pool(procs,
            initializer=process,
            initargs=(q_job, q_res, iolock, func, args, kwargs))

    # stream the data to queue
    for line in lines:

        # halts if queue is full
        q_job.put(line)

    # stop the process and pool
    for _ in range(procs):
        q_job.put(None)

    pool.close()
    pool.join()

    # Collect all results
    results = []
    while not q_res.empty():
        results.append(q_res.get(block=False))

    return results


def process(q, results, iolock, func, args, kwargs):
    """

    multiprocessing interface for calling

    func(x, *args, **kwargs) with `x` coming from q

    args
        q: multiprocessing queue for distributing workload.
        results: multiprocessing queue for collecting results.
        iolock: print lock.
        func: function to be called with `q` output.
        kwargs: Arbitrary keyword arguments for `func`.
        procs: how many processes to start.

    """

    kwargs["iolock"] = iolock

    while True:

        line = q.get()

        if line is None:
            break

        result = func(line, *args, **kwargs)
        results.put(result)

    return

