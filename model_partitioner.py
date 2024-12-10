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
        stage_inputs[inp] = tf.keras.layers.Input(shape=input_shape, batch_size = 1, name=inp)

    for stage_input in stage_inputs.keys():
        intra_stage_skips[stage_input] = stage_inputs[stage_input]

    if isinstance(model.layers[start].input, list):
        temp = []
        for stage_input in model.layers[start].input:
            temp.append(stage_inputs[stage_input.name.split('/')[0]])
            inter_stage_skips[stage_input.name.split('/')[0]] = stage_inputs[stage_input.name.split('/')[0]]
        x = temp
    else:
        x = stage_inputs[model.layers[start].input.name.split('/')[0]]
    
    for i in range(start, end):
        layer = model.layers[i]
        print(i, layer.name)
        if isinstance(layer.input, list):
            for inbound_layer in layer._inbound_nodes[0].inbound_layers:
                origin_layer = model.get_layer(inbound_layer.name)
                origin_idx = model.layers.index(origin_layer)
                if origin_idx != i-1:
                    if origin_layer.name in intra_stage_skips.keys():
                        try:
                            x = layer([x, intra_stage_skips[origin_layer.name]])
                        except ValueError:
                            x = layer(x)
                    elif origin_layer.name in inter_stage_skips.keys():
                        x = layer([x, inter_stage_skips[origin_layer.name]])
                    else:
                        x = layer(x)
        else:
            try:
                x = layer(x)
            except ValueError:
                if i in [192]:
                    x = layer(x)
                else:
                    x = tf.squeeze(x, axis=0)
                    if i in [150]:
                        x = tf.squeeze(x, axis=0)
                    x = layer(x)
        if len(layer._outbound_nodes)>1:
            dest_idx = model.layers.index(layer._outbound_nodes[1].outbound_layer)
            if dest_idx < end:
                intra_stage_skips[layer.name] = x
            else:
                inter_stage_skips[layer.name] = x

    #generate submodel
    if inter_stage_skips:
        x=[x]+list(inter_stage_skips.values())
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

def save_models(output_dir, sub_model, stage_num, coral=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(sub_model)
    if coral:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  
        ]
    tflite_model = converter.convert()
    tflite_path = os.path.join(output_dir, f"sub_model_{stage_num+1}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved tflite model to: {tflite_path}")
    return tflite_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pipeline Stage')
    parser.add_argument('--config', type=str, required=True, help='Device and model configuration file path')
    parser.add_argument('--model', type=str, required=True, help='Path to original h5 model')
    parser.add_argument('--output_dir', type=str, default='0.0.0.0', help='Directory to save the partitioned sub-models')
    return parser.parse_args()

def main():
    args = parse_arguments()
    N, C, M, U, E, R = load_config(args.config)
    P = DNNPipe(N, C, M, U, E, R)
    partitioning_points = [P[0][0]] 
    for t in P:
        partitioning_points.append(t[1])
    model = load_model(args.model)
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
        use_coral = (P[i][2] == 3 or 4)
        tflite_models.append(save_models(args.output_dir, sub_model, i, coral=use_coral))

    with open('pipeline_plan.json', 'w') as f:
        json.dump({'pipeline_plan': P}, f)

if __name__ == "__main__":
    main()