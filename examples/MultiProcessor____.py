import multiprocessing as mp


class MultiProcessor(object):
    def __init__(self, target_func, queue=None, num_proc=mp.cpu_count()):

        self.queue = mp.Queue() if queue is None else queue
        self.target_func = target_func
        self.pool = mp.Pool(processes=num_proc)
        self.output = mp.Queue()
        self.callback = lambda x: self.output.put(x)

    def run(self):
        if not self.queue.empty():
            self.pool.starmap_async(
                self.target_func, (self.queue.get(),), callback=self.callback
            )
