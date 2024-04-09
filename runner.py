from multiprocessing import Process, Queue
from mmpose.apis import MMPoseInferencer
import pickle
import cv2
import os


class QueryWorker(Process):

    def __init__(self, queue, resultsFromChildren):
        super().__init__()

        self.screenshotPath = "extracted-screenshots"
        self.filePath = "extracted-files"
        self.queue = queue
        self.results = resultsFromChildren
        self.inferencer = MMPoseInferencer(
           "wholebody"
        )

    def run(self):
        for data in iter(self.queue.get, None):
            self.processWork(data)
        print("DONE")

    def processWork(self, entity):
        screenshot_path = os.path.join(self.screenshotPath, "{:010d}.jpg".format(entity[1]))
        result_path = self.filePath

        cv2.imwrite(screenshot_path, entity[0])

        result_generator = self.inferencer(screenshot_path, show=False, out_dir=result_path)
        result = next(result_generator)

        self.results.put([entity[1], result])


class WorkScheduler():

    def __init__(self, config):
        self.numWorkers = config["numWorkers"]
        self.workerPool = []

        self.resultsFromChildren = Queue()
        self.workToDo = Queue()

        print("Setting up workers...")
        for worker in range(self.numWorkers):
            self.workerPool.append(QueryWorker(self.workToDo, self.resultsFromChildren))

        for worker in range(self.numWorkers):
            self.workerPool[worker].start()

    def add_work(self, work):
        self.workToDo.put(work)

    def get_results(self):
        for worker in range(self.numWorkers):
            self.workToDo.put(None)

        for worker in range(self.numWorkers):
            self.workerPool[worker].join()

        combined_results = {}

        return combined_results


if __name__ == '__main__':

    config = {"numWorkers": 1}

    workerScheduler = WorkScheduler(config=config)

    capture = cv2.VideoCapture(0)
    end = 200

    capture.set(1, 0)

    for idx in range(0, end):
        ret, frame = capture.read()

        if ret:
            workerScheduler.add_work([frame, idx])
        else:
            capture.release()
            break

    capture.release()
    results = workerScheduler.get_results()

    exit(0)