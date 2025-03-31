# DNNPipe: Dynamic Programming-based Optimal DNN Partitioning for Pipelined Inference on IoT Networks

DNNPipe: Dynamic Programming-based Optimal DNN Partitioning for Pipelined Inference on IoT Networks
This repository contains the implementation of our paper "DNNPipe: Dynamic Programming-based Optimal DNN Partitioning for Pipelined Inference on IoT Networks".

## Overview
DNNPipe is an optimal DNN partitioning algorithm that enables efficient execution of DNN inference on resource-constrained IoT devices through pipeline parallelization. Using dynamic programming with two novel pruning techniques (UBP and USP), DNNPipe maximizes throughput while minimizing runtime overhead for DNN inference.

## System Requirements

- NVIDIA Jetson AGX Xavier (Ubuntu 18.04.6 LTS, JetPack 4.6)
- NVIDIA Jetson Nano 2GB (Ubuntu 18.04.6 LTS, JetPack 4.6)
- Google Coral Dev Board (Mendel Linux 5.0)
- Python 3.11
- TensorFlow 2.5

## Usage
The system consists of three main components:

1. The model partitioner (`model_partitioner.py`) : Generates optimal pipeline plan and partitions DNN model
  - Input: DNN model in H5 format, configuration file in JSON
  - Output: Pipeline plan, Partitioned sub-models in TFLite format
  ```bash
  python model_partitioner.py --model-path ./original_models --config ./configurations --output-dir ./partitioned_submodels --model-name model
  ```
2. The deployer (`deployer.py`): Distributes partitioned models across IoT nodes
  - Input: Device registry in JSON, Pipeline plan in JSON, partitiond sub-models
  - Output: None
  ```bash
  python deployer.py --device-registry ./configurations/device_registry.json --model-path ./partitioned_submodels/ --model-name model
  ```
3. The runtime engine (`runtime_engine.py`): Executes pipelined inference
  - Input: stage number, binding address for connection, binding port for connection, ip address of next stage, port address of next stage, dataset path
  - Output: None
  ```bash
  python runtime_engine.py --stage 2 --model-path partitioned_submodels --bind-addr 0.0.0.0 --bind-port 50000 --next-host 192.168.100.61 --next-port 50001 --data-path dataset.h5 --model-name model
  ```
## Device and Model Configuration Guide
	-"device_config": Device configuration parameters 
	   -"N": Number of available devices N
	   -"C": Ordered list representing the sequence of performance scaling factors per unit relative to device 1
	   -"M": Ordered list representing the sequence of available memory in megabytes per device
	-"model_config": DNN model configuration parameters 
	   -"U": Number of units U in the DNN model
	   -"E": Ordered list representing the sequence of execution times in milliseconds per unit when running on device 1
	   -"R": Ordered list representing the sequence of required runtime memory in megabytes per unit
For detailed parameter descriptions, please refer to Section 3.1 and 5.3.1 of the paper.
