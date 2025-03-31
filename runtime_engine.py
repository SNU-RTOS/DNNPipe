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
import h5py
import os
import csv

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
    def __init__(self, stage_num, model_path, bind_addr, bind_port, next_host, next_port, data_path, model_name, fps):
        self.stage_num = stage_num
        self.model_path = model_path
        self.fps = fps
        self.frame_time = 1 / self.fps
        self.model_name = model_name
        
        logger.info(f"Loading model from {model_path}")
        self.interpreter = tflite.Interpreter(model_path=f'{model_path}/{model_name}/sub_model_{self.stage_num}.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']
        if data_path:
            self.dataset = h5py.File(data_path, 'r')['images']
        self.current_idx = 0
        
        self.start = []
        self.end = []
        self.queue_wait_times = []
        self.inference_times = []
        
        self.bind_addr = bind_addr
        self.bind_port = bind_port
        self.next_host = next_host
        self.next_port = next_port
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.results = []
        self.lock = threading.Lock()

    def connect(self):
        """ 모든 장치가 연결될 때까지 대기 후, Y를 입력하면 시작 """
        if self.bind_addr is not None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((self.bind_addr, self.bind_port))
                logger.info(f"Waiting for connection on {self.bind_addr}:{self.bind_port}")
                sock.listen(1)
                self.conn, addr = sock.accept()
                logger.info(f"Connected from previous stage: {addr}")

        if self.stage_num != 4:
            while True:
                try:
                    self.next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    logger.info(f"Connecting to next stage at {self.next_host}:{self.next_port}")
                    self.next_sock.connect((self.next_host, self.next_port))
                    logger.info(f"Connected to next stage: {self.next_host}:{self.next_port}")
                    break
                except ConnectionRefusedError:
                    logger.warning("Connection refused. Retrying in 5 seconds...")
                    time.sleep(5)
        
        # 모든 장치 연결 완료 후, 사용자 입력 대기
        if self.stage_num == 1:
            input("\n✅ All connections established! Press 'Y' to start: ")
            logger.info("Starting pipeline execution...")
        
    def receiver(self):
        """ 일정한 FPS 간격으로 데이터를 큐에 입력 """
        start_time_global = time.time()
        if self.bind_addr == None:
            logger.info("First stage: Loading preprocessed data")
            for i in range(NUM_INPUTS):
                data = self.dataset[self.current_idx]
                self.current_idx += 1
                input_data = np.expand_dims(data, axis=0)
                enqueue_time = time.time()
                self.input_queue.put((input_data, enqueue_time))
                self.start.append(time.time())
                # 🎯 FPS 유지 (다음 입력까지 대기)
                
                next_frame_time = start_time_global + ((i + 1) * self.frame_time)
                sleep_time = max(0, next_frame_time - time.time())
                # logger.info(f"Start time of {i}th input: {time.time()}")
                time.sleep(sleep_time)
        else:
            for _ in range(NUM_INPUTS):
                tensor = receive_tensor(self.conn, self.input_shape)
                self.start.append(time.time())
                enqueue_time = time.time()
                self.input_queue.put((tensor, enqueue_time))

    def executer(self):
        """ 추론을 수행하는 스레드 """
        logger.info("Starting worker thread (inference)")
        
        for i in range(NUM_INPUTS):
            input_data, enqueue_time = self.input_queue.get()  # 큐에서 데이터 가져오기

            # ⏳ 추론 시작 시간
            start_time = time.time()
            queue_wait_time = (start_time - enqueue_time) * 1000  # ms 변환

            # ⚡ 모델 추론 실행
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            result = self.interpreter.get_tensor(self.output_details[0]['index'])
            self.output_queue.put(result)

            # ⏳ 추론 종료 시간
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # ms 변환

            # 🟢 리스트에 시간 데이터 추가
            self.queue_wait_times.append(queue_wait_time)
            self.inference_times.append(inference_time)
            #logger.debug(f"Inference {i+1}: Queue Wait {queue_wait_time:.2f} ms, Inference {inference_time:.2f} ms, Total {total_latency:.2f} ms")

    def sender(self):
        if self.stage_num == 4:
            logger.info("Last stage: Collecting results")
            for i in range(NUM_INPUTS):
                tensor = self.output_queue.get()
                self.end.append(time.time())
                self.results.append(tensor)
        else:
            for i in range(NUM_INPUTS):
                tensor = self.output_queue.get()
                send_tensor(self.next_sock, tensor)
                self.end.append(time.time())

    def save_logs(self):
        """ 📝 Queue Wait Time & Inference Time 데이터를 MATLAB 친화적인 CSV 형식으로 저장 """
        log_dir = f"./logs/{self.model_name}"
        os.makedirs(log_dir, exist_ok=True)

        # 파일 저장 경로
        file_path = f"{log_dir}/stage{self.stage_num}_{self.fps}fps.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # MATLAB에서 쉽게 읽을 수 있도록 데이터 저장
            writer.writerow(["Frame Index", "Queue Wait Time (ms)", "Inference Time (ms)", "Stage Start Time (ms)", "Stage End Time (ms)"])
            
            for idx, (queue, inference, start, end) in enumerate(zip(self.queue_wait_times, self.inference_times, self.start, self.end)):
                writer.writerow([idx, f"{queue:.2f}", f"{inference:.2f}", f"{start:.2f}", f"{end:.2f}"])
        
        logger.info(f"✅ Log saved to {file_path}")
        
    def run(self):
        self.connect() 
        
        threads = [
            threading.Thread(target=self.receiver),
            threading.Thread(target=self.executer),
            threading.Thread(target=self.sender),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.save_logs()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pipeline Stage')
    parser.add_argument('--stage', type=int, required=True, help='Stage number')
    parser.add_argument('--model-path', type=str, required=True, help='Path of the partitioned model')
    parser.add_argument('--bind-addr', type=str, default=None, help='Binding address for connection')
    parser.add_argument('--bind-port', type=int, default=None, help='Binding port for connection')
    parser.add_argument('--next-host', type=str, default=None, help='Next stage host')
    parser.add_argument('--next-port', type=int, default=None, help='Next stage port')
    parser.add_argument('--data-path', type=str, default=None, help='Input dataset path')
    parser.add_argument('--model-name', type=str, default=None, help='Name of model')
    parser.add_argument('--fps', type=int, default=None, help='Input FPS')
    return parser.parse_args()

def main():
    args = parse_arguments()
    logger.info(f"Starting pipeline stage {args.stage}")
    stage = PipelineStage(
        stage_num=args.stage,
        model_path=args.model_path,
        bind_addr=args.bind_addr,
        bind_port=args.bind_port,
        next_host=args.next_host,
        next_port=args.next_port,
        data_path=args.data_path,
        model_name=args.model_name,
        fps = args.fps
    )
    stage.run()

if __name__ == "__main__":
    main()