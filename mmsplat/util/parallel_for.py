from threading import Thread
from threading import Lock
import itertools

class _ParallelFor:
    def __init__(self):
        self.lock = Lock()
        self.counter = 0

    # run len(list) many threads, list must contain parameter tuples
    @staticmethod
    def simple_run_parallel(func, list):
        threads = []
        for param in list:
            th = Thread(target=func, args=param)
            th.start()
            threads.append(th)

        for th in threads:
            th.join()

    
    def _parallel_for_thread(self, list, func, counter_function, counter_total):
        for el in list:
            func(el)
            with self.lock:
                if self.counter < 0:  # kill request
                    return

                self.counter += 1
                if counter_function is not None:
                    counter_function(self.counter, counter_total)


    def run(self, func, list, num_threads=8, counter_function=None):
        self.counter = 0

        batches = [itertools.islice(list, t_idx, None, num_threads) for t_idx in range(num_threads)]
        thread_params = [(batch, func, counter_function, len(list)) for batch in batches]
            
        try:
            _ParallelFor.simple_run_parallel(self._parallel_for_thread, thread_params)
        except KeyboardInterrupt:
            with self.lock:
                self.counter=-1

        


def parallel_for(func, list, num_threads=8, counter_function=None):
    _ParallelFor().run(func, list, num_threads, counter_function)
    
def run_parallel(func, param_list):
    _ParallelFor.simple_run_parallel(func, param_list)