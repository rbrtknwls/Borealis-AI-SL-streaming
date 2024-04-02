from multiprocessing import Process, Queue
import random
import cv2
import os

class QueryWorker(Process):

    def __init__(self, queue, resultsFromChildren, savePath):
        super().__init__()

        self.savePath = savePath
        self.queue = queue
        self.results = resultsFromChildren

    def run(self):
        for data in iter(self.queue.get, None):
            self.processWork(data)

    def processWork(self, entity):
        save_path = os.path.join(self.savePath, "{:010d}.jpg".format(entity[1]))
        print(save_path)
        cv2.imwrite(save_path, entity[0])
        result = random.randint(1, 2)
        self.results.put([entity[1], result])


class WorkScheduler():

    def __init__(self, config):
        self.numWorkers = config["numWorkers"]
        self.workerPool = []

        self.resultsFromChildren = Queue()
        self.workToDo = Queue()

        print("Setting up workers...")
        for worker in range(self.numWorkers):
            self.workerPool.append(QueryWorker(self.workToDo, self.resultsFromChildren, "extracted-files"))

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

        while (not self.resultsFromChildren.empty()):
            result = self.resultsFromChildren.get()
            combined_results[result[0]] = result[1]

        return combined_results


if __name__ == '__main__':


    config = {"numWorkers": 2}

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
    print(results)