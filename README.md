# ${Opara}$

${Opara}$ is a lightweight and resource-aware DNN Operator parallel scheduling framework to accelerate the execution of DNN inference on GPUs. Specifically, ${Opara}$ first employs CUDA Graph and CUDA Streams to automatically parallelize the execution of multiple DNN operators. It further leverages the resource requirements of DNN operators to judiciously adjust the operator launch order on GPUs to expedite DNN inference.

# System overview of ${Opara}$

${Opara}$ comprises four components including Model Profiler, Operator Launcher, Stream Allocator, and Graph Capturer. As illustrated in  the subsequent figure, ${Opara}$ takes DNN models and input tensors (i.e., inference data) from users. According to the operator dependencies in the DAG of DNN models, the Stream Allocator first employs a stream allocation algorithm to determine which stream the operators should be allocated to. The Model Profiler then gathers the resource requirements of each operator during the model profiling process. With such resource requirements of operators, the Operator Launcher further employs a resource-aware operator launch algorithm to optimize the operator launch order on GPUs. Finally, the Graph Capturer generates a parallelized CUDA Graph by combing the stream allocation plan and operator launch order, thereby enabling efficient DNN inference on GPUs.
![overview](/figures/overview.png)

# Installation

```shell
git clone https://github.com/OparaSys/Opara.git
cd Opara
pip install -r requirements.txt
```

# Usage

```shell
import torch
import torchvision
from Opara import GraphCapturer

model = torchvision.models.googlenet().eval()
model = model.to(device="cuda:0")
x = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32).to(device="cuda:0")
inputs = (x,)
# Submit DNN model and input tensors as two parameters to instantiate a model execution with parallel operator execution.
Opara = GraphCapturer.capturer(inputs, model)
output = Opara(*inputs)
```

# Example

```shell
python examples/googlenet_example.py
```
