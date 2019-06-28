import os
import sys
import threading
try:
    import queue
except:
    import Queue as queue
import logging
logging.basicConfig(level=logging.INFO)


def add_path(path):
    path = os.path.abspath(path)
    if path not in sys.path:
        logging.info("load path %s" % path)
        sys.path.insert(0, path)

pwd = os.path.dirname(os.path.realpath(__file__))
print(pwd)

add_path(os.path.join(pwd, '..', 'api'))
add_path(os.path.join(pwd, '..'))

from api.application import Bucket

class Worker(threading.Thread):
    def __init__(self, tasks):
        threading.Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            func, args, kw = self.tasks.get()
            try:
                func(*args, **kw)
            except RuntimeErr as e:
                print(e)
            finally:
                self.tasks.task_done()

class ThreadingPool:
    def __init__(self, max_workers=5):
        self.tasks = queue.Queue(max_workers)
        for _ in range(max_workers):
            Worker(self.tasks)
    
    def add_task(self, func, *args, **kw):
        self.tasks.put((func, args, kw))

    def wait_completion(self):
        self.tasks.join()

    def run(self):
        raise Exception("Not Implemented!")

class DownloaderPool(ThreadingPool):
    def __init__(self, max_workers=5):
        ThreadingPool.__init__(self, max_workers)
        self._bucket = Bucket(obs_name="modelarts-explore", bucket_name="ai-course-001")

    def run(self, files):
        target_path_head = os.path.join(
                self._bucket._obs_name,
                self._bucket._bucket_name
                )
        target_path = "{}/dog_and_cat_recognition/output/services/V0009/model/".format(target_path_head)
        def upload(f):
            self._bucket.upload(f, target_path)

        for f in files:
            self.add_task(upload, f)


def Program(raw_args):

    target_bucket_path = "modelarts-explore/ai-course-001/dog_and_cat_recognition/output/services/V0009/model/"

    packed = (
        os.path.join(pwd, "..", "dataset.py"),
        os.path.join(pwd, "customize_service.py"),
        os.path.join(pwd, "config.json"),
        os.path.join(pwd, "..", "..", "config/config.py"),
        os.path.join(pwd, "..", "..", "config/global_settings.py"),
        os.path.join(pwd, "..", "settings.py"),
    )

    packed = [os.path.abspath(p) for p in packed]
    
    # deploy files to remote S3
    pool = DownloaderPool(len(packed))
    pool.run(packed)
    pool.wait_completion()

if __name__ == "__main__":
    sys.exit(Program(sys.argv[1:]))
