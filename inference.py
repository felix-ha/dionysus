import requests
import timeit

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import onnxruntime as ort


def run_inference_onnx_local(x):
    session = ort.InferenceSession("models/model.onnx")
    input_name = session.get_inputs()[0].name

    input_data = np.array([[1,1.2,1,1]])  # Provide your input data here
    input_data = input_data.astype(np.float32) 

    logits = session.run(None, {input_name: input_data})
    return F.softmax(torch.tensor(logits), dim=0).numpy()


def run_inference_onnx_webservice(x):

    x = x[0].numpy().tolist()
    
    url = "http://localhost:80/inference"
    json_content = {
        "model": "model.onnx",
        "x": x
    }
    response = requests.post(url, json=json_content)
    results_json = response.json()
    return np.array(results_json["Logits"])

def run_inference_pytorch_local(x):
    n_features = 4
    n_classes = 5

    model = nn.Sequential(nn.Linear(n_features, 15), 
                          nn.Tanh(), nn.Dropout(0.5),
                          nn.Linear(15, 15), 
                         nn.Tanh(), nn.Dropout(0.5),
                            nn.Linear(15, n_classes)) 
    checkpoint = torch.load('models/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        logits = model(x)

    return F.softmax(logits, dim=0).numpy()

def timing(n, x):
    execution_time_onnx = timeit.timeit(lambda: run_inference_onnx_webservice(x), number=n)
    execution_time_torch = timeit.timeit(lambda: run_inference_pytorch_local(x), number=n)
    print("Execution time onnx :", execution_time_onnx/n, "seconds")
    print("Execution time torch:", execution_time_torch/n, "seconds")

x = torch.rand(4)

logits_onnx = run_inference_onnx_webservice(x)
#logits_onnx_local = run_inference_onnx_local(x)
logits_torch = run_inference_pytorch_local(x)

print("onnx go", logits_onnx, np.argmax(logits_onnx))
#print("onnx local", logits_onnx_local, np.argmax(logits_onnx_local))
print("torch", logits_torch, np.argmax(logits_torch))

N = 1000
for _ in range(N):
    correct = 0
    x = torch.rand(4)

    logits_onnx = run_inference_onnx_webservice(x)
    logits_torch = run_inference_pytorch_local(x)
    if np.argmax(logits_onnx) == np.argmax(logits_torch):
        correct += 1
    
print(correct/N)


# timing(1000, x)
