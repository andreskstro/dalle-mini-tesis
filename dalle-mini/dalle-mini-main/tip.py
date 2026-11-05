import numpy as np
import json


with open("eb_neuron_influences.json") as f:
    data = json.load(f)

vector = data[0]["layer_influences"]["model.decoder.layers.FlaxBartDecoderLayers.GLU_0.Dense_0.kernel"]

arr = np.array(vector)
print("Max:", arr.max())
print("Mean:", arr.mean())
print("Std:", arr.std())
print("Min:", arr.min())