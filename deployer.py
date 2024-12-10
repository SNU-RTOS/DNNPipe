"""
@author: Woobean Seo
@affiliation: Real-Time Operating System Laboratory, Seoul National University
@contact: wbseo@redwood.snu.ac.kr
@date: 2024-12-10
"""
import paramiko
from scp import SCPClient
import os
import logging
import json
import argparse

class DeviceRegistry:
    def __init__(self, registry_file="device_registry.json"):
        self.registry_file = registry_file
        self.devices = self._load_registry()
    
    def _load_registry(self):
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                return data['devices']
        except FileNotFoundError:
            return []
    
    def get_device_by_id(self, device_id):
        for device in self.devices:
            if device['device_id'] == device_id:
                return device
        return None

class Deployer:
    def __init__(self, registry, plan, model_path):
        self.registry = registry
        self.logger = logging.getLogger(__name__)
        self.partition_mapping = self._load_partition_mapping(plan)
        self.model_path = model_path
        self.model_device_mapping = {
            idx+1: device_id 
            for idx, (_, _, device_id) in enumerate(self.partition_mapping)
        }
    
    def _load_partition_mapping(self, plan):
        with open(plan, 'r') as f:
            data = json.load(f)
            return data['pipeline_plan']

    def deploy_models(self):
        for model_num, device_id in self.model_device_mapping.items():
            device = self.registry.get_device_by_id(device_id)
            if device:
                self.deploy_to_device(device, model_num)
            else:
                self.logger.error(f"Device with ID {device_id} not found in registry")

    def deploy_to_device(self, device, model_num):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            self.logger.info(f"Connecting to {device['host']}...")
            ssh.connect(
                device['host'],
                username=device['username'],
                password=device['password']
            )

            stdin, stdout, stderr = ssh.exec_command('mkdir -p DNNPipe/partitioned_models')
            
            with SCPClient(ssh.get_transport()) as scp:
                local_path = f"{self.model_path}/sub_model_{model_num}.tflite"
                remote_path = f"DNNPipe/{self.model_path}/sub_model_{model_num}.tflite"
                
                self.logger.info(f"Transferring {local_path} to {device['host']}:{remote_path}")
                scp.put(local_path, remote_path)
                self.logger.info(f"Successfully deployed model {model_num} to {device['host']}")

        except Exception as e:
            self.logger.error(f"Failed to deploy model to {device['host']}: {str(e)}")
            
        finally:
            ssh.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pipeline Stage')
    parser.add_argument('--device_registry', type=str, required=True, help='Device registry file')
    parser.add_argument('--plan', type=str, required=True, help='Path to original h5 model')
    parser.add_argument('--model_path', type=str, default='0.0.0.0', help='Path of the partitioned sub-models')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    registry = DeviceRegistry(args.device_registry)
    deployer = Deployer(registry, args.plan, args.model_path)
    deployer.deploy_models()

if __name__ == "__main__":
    main()