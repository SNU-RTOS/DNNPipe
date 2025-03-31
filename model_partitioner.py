"""
@author: Woobean Seo
@affiliation: Real-Time Operating System Laboratory, Seoul National University
@contact: wbseo@redwood.snu.ac.kr
@date: 2024-12-10
"""
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, EfficientNetV2B0, DenseNet169
from vit_keras import vit
from DNNPipe import DNNPipe
import json
import argparse

def load_config(file_path='config.json'):
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    N = config['device_config']['N']
    C = config['device_config']['C']
    M = config['device_config']['M']
    U = config['model_config']['U']
    E = config['model_config']['E']
    R = config['model_config']['R']
    
    return N, C, M, U, E, R

def DNNPartitioning(model, start, end, prev_outputs):
    stage_inputs = {}
    intra_stage_skips = {}
    inter_stage_skips = {}

    for inp in prev_outputs.keys():
        input_shape =  prev_outputs[inp].shape[1:]
        stage_inputs[inp] = tf.keras.layers.Input(shape=input_shape, name=inp)

    for stage_input in stage_inputs.keys():
        intra_stage_skips[stage_input] = stage_inputs[stage_input]
    if isinstance(model.layers[start].input, list):
        temp = []
        for stage_input in model.layers[start].input:
            temp.append(stage_inputs[stage_input.name.split('/')[0]])
            inter_stage_skips[stage_input.name.split('/')[0]] = stage_inputs[stage_input.name.split('/')[0]]
        x = temp
    else:
        if len(stage_inputs) == 1:
            x = next(iter(stage_inputs.values())) 
        else:
            x = stage_inputs[model.layers[start].input.name.split('/')[0]]
    
    for i in range(start, end):
        layer = model.layers[i]
        inbound_node = layer._inbound_nodes[0]
        inbound_layers = inbound_node.inbound_layers
        # print(i, layer.name)
        if isinstance(inbound_layers, list) and len(inbound_layers) > 1:
            for inbound_layer in layer._inbound_nodes[0].inbound_layers:
                origin_layer = model.get_layer(inbound_layer.name)
                origin_idx = model.layers.index(origin_layer)
                if origin_idx != i-1:
                    if origin_layer.name in intra_stage_skips.keys():
                        try:
                            x = layer(x)
                        except (ValueError, TypeError):
                            try:
                                x = layer([x, intra_stage_skips[origin_layer.name]])
                            except:
                                x = layer(x, intra_stage_skips[origin_layer.name])
                    elif origin_layer.name in inter_stage_skips.keys():
                        x = layer([x, inter_stage_skips[origin_layer.name]])
                    else:
                        x = layer(x)
        else:
            try:
                origin_layer = model.get_layer(layer._inbound_nodes[0].inbound_layers.name)
                x = layer(intra_stage_skips[origin_layer.name])
            except (TypeError, KeyError):
                try:
                    x = layer(x)
                except (TypeError, ValueError):
                    x = layer(x[0])

        if len(layer._outbound_nodes)>1:
            dest_idx = model.layers.index(layer._outbound_nodes[1].outbound_layer)
            if dest_idx < end:
                intra_stage_skips[layer.name] = x
            else:
                inter_stage_skips[layer.name] = x
        else:
            if i != len(model.layers)-1:
                dest_idx = model.layers.index(layer._outbound_nodes[0].outbound_layer)
                if dest_idx != i+1:
                    if dest_idx < end:
                        intra_stage_skips[layer.name] = x
                    else:
                        inter_stage_skips[layer.name] = x

    #generate submodel
    if inter_stage_skips:
        x=[x]+list(inter_stage_skips.values())
    else:
        try:
            x = list(x)[0]
        except TypeError:
            pass
    submodel = tf.keras.models.Model(inputs=list(stage_inputs.values()), outputs=x)
    return submodel

def create_sample_input(shape=(1, 224, 224, 3)):
    return np.random.rand(*shape)

def prepare_next_stage_inputs(sub_model, stage_outputs):
    next_inputs = {}
    for model_output, actual_output in zip(sub_model.outputs, stage_outputs):
        layer_name = model_output._keras_history[0].name
        next_inputs[layer_name] = actual_output
    return next_inputs

def save_models(output_dir, model_name, sub_model, stage_num, coral=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(sub_model)
    tflite_model = converter.convert()
    tflite_path = os.path.join(f"{output_dir}/{model_name}/", f"sub_model_{stage_num+1}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved tflite model to: {tflite_path}")
    return tflite_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pipeline Stage')
    parser.add_argument('--config', type=str, required=True, help='Device and model configuration file path')
    parser.add_argument('--model-path', type=str, required=True, help='Path to original h5 model')
    parser.add_argument('--output-dir', type=str, default='0.0.0.0', help='Directory to save the partitioned sub-models')
    parser.add_argument('--model-name', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_name = args.model_name
    N, C, M, U, E, R = load_config(f"{args.config}/{model_name}/config.json")
    P = DNNPipe(N, C, M, U, E, R)
    with open(f'./configurations/{model_name}/pipeline_plan.json', 'w') as f:
        json.dump({'pipeline_plan': P}, f)
    partitioning_points = [1] 
    for t in P:
        partitioning_points.append(t[1]+1)
    
    model = load_model(f"{args.model_path}/{model_name}.h5")
    sample_input = create_sample_input()
    num_stages = len(partitioning_points) - 1
    sub_models = []
    tflite_models = []
    for i in range(num_stages):
        if i == 0:
            stage_inputs = {model.layers[0].name: sample_input}
        else:
            stage_inputs = prepare_next_stage_inputs(sub_models[i-1], sub_models[i-1].outputs)
        sub_model = DNNPartitioning(
            model,
            partitioning_points[i],
            partitioning_points[i+1],
            stage_inputs
        )
        sub_models.append(sub_model)
        tflite_models.append(save_models(args.output_dir, model_name, sub_model, i))

if __name__ == "__main__":
    main()