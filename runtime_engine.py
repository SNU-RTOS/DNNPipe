"""
@author: Woobean Seo
@affiliation: Real-Time Operating System Laboratory, Seoul National University
@contact: wbseo@redwood.snu.ac.kr
@date: 2024-12-10
"""
import socket
import threading
import queue
import numpy as np
import struct
import time
import tensorflow.lite as tflite
import argparse
import logging
from loader import PreprocessedDataLoader

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NUM_INPUTS = 100
INPUT_SHAPE = (1, 224, 224, 3)

def send_tensor(sock, tensor):
    data = tensor.tobytes()
    header = struct.pack('>I', len(data))
    sock.sendall(header + data)

def receive_tensor(sock, shape, dtype=np.float32):
    header = sock.recv(4)
    data_length = struct.unpack('>I', header)[0]
    data = sock.recv(data_length, socket.MSG_WAITALL)
    return np.frombuffer(data, dtype=dtype).reshape(shape)

class PipelineStage:
    def __init__(self, stage_num, model_path, prev_host, prev_port, next_host, next_port, data_path):
        self.stage_num = stage_num
        self.model_path = model_path
        
        logger.info(f"Loading model from {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']
        
        self.dataset = h5py.File(data_path, 'r')['images']
        self.current_idx = 0
        
        self.prev_host = prev_host
        self.prev_port = prev_port
        self.next_host = next_host
        self.next_port = next_port
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.results = []
        self.lock = threading.Lock()

    def receiver(self):
        if self.stage_num == 'f':
            logger.info("First stage: Loading preprocessed data")
            for _ in range(NUM_INPUTS):
                data = self.dataset[self.current_idx]
                self.current_idx += 1
                input_data = np.expand_dims(data, axis=0)  
                self.input_queue.put(input_data)
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((self.prev_host, self.prev_port))
                logger.info(f"Waiting for connection on {self.prev_host}:{self.prev_port}")
                sock.listen(1)
                conn, addr = sock.accept()
                logger.info(f"Connected from previous stage: {addr}")
                with conn:
                    for _ in range(NUM_INPUTS):
                        tensor = receive_tensor(conn, self.input_shape)
                        self.input_queue.put(tensor)

    def executer(self):
        logger.info("Starting model inference")
        for i in range(NUM_INPUTS):
            tensor = self.input_queue.get()
            if i == 0:
                self.start_time = time.time()
            infer_time_start = time.time()
            self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
            self.interpreter.invoke()
            result = self.interpreter.get_tensor(self.output_details[0]['index'])
            self.output_queue.put(result)
            logger.debug(f"Inference time for input {i+1}: {time.time()-infer_time_start:.4f}s")

    def sender(self):
        if self.stage_num == 'l':
            logger.info("Last stage: Collecting results")
            for i in range(NUM_INPUTS):
                tensor = self.output_queue.get()
                self.results.append(tensor)
                if i == NUM_INPUTS - 1:
                    elapsed_time = time.time() - self.start_time
                    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                logger.info(f"Connecting to next stage at {self.next_host}:{self.next_port}")
                while True:
                    try:
                        sock.connect((self.next_host, self.next_port))
                        logger.info(f"Connected to next stage: {self.next_host}:{self.next_port}")
                        break
                    except ConnectionRefusedError:
                        logger.warning("Connection refused. Retrying in 5 seconds...")
                        time.sleep(5)
                
                for _ in range(NUM_INPUTS):
                    tensor = self.output_queue.get()
                    send_tensor(sock, tensor)

    def run(self):
        threads = [
            threading.Thread(target=self.receiver),
            threading.Thread(target=self.executer),
            threading.Thread(target=self.sender),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pipeline Stage')
    parser.add_argument('--stage', type=str, required=True, help='first: f, intermidiate: i, last: l')
    parser.add_argument('--model', type=str, required=True, help='Path to TFLite model')
    parser.add_argument('--prev-host', type=str, default='0.0.0.0', help='Previous stage host')
    parser.add_argument('--prev-port', type=int, default=None, help='Previous stage port')
    parser.add_argument('--next-host', type=str, default=None, help='Next stage host')
    parser.add_argument('--next-port', type=int, default=None, help='Next stage port')
    parser.add_argument('--data-path', type=int, default=None, help='Next stage port')
    return parser.parse_args()

def main():
    args = parse_arguments()
    logger.info(f"Starting pipeline stage {args.stage}")
    logger.info(f"Model path: {args.model}")
    logger.info(f"Previous stage: {args.prev_host}:{args.prev_port}")
    if args.next_host and args.next_port:
        logger.info(f"Next stage: {args.next_host}:{args.next_port}")

    model_path = f'partitioned_models/sub_model_{args.model}.tflite'
    stage = PipelineStage(
        stage_num=args.stage,
        model_path=model_path,
        prev_host=args.prev_host,
        prev_port=args.prev_port,
        next_host=args.next_host,
        next_port=args.next_port,
        data_path=args.data_path,
    )
    stage.run()

if __name__ == "__main__":
    main()