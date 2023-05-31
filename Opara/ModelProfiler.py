import torch
from torch.fx import symbolic_trace
import torch.profiler
from torch.fx import Interpreter
import torch._dynamo.eval_frame
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")
    

def profile(model, inputs):
    model = model.cuda()
    symbolic_traced = symbolic_trace(model)
    # dynamo.reset()
    # explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose = dynamo.explain(model, *inputs)
    # print("len(graphs):", len(graphs))
    # symbolic_traced = graphs[0]
    # symbolic_traced = torch._dynamo.export(model, *inputs)[0]
    interpreter = Interpreter(symbolic_traced)
    
    path = os.path.abspath(os.path.dirname(__file__)) + "/profile_result/"
    
    inputs_name = symbolic_traced.__class__.__name__
    for i in inputs:
        inputs_name += str(i.shape) + "_"

    def trace_handler(p):
        p.export_chrome_trace(path + inputs_name + ".pt.trace.json")
    
    
    with torch.profiler.profile(
        on_trace_ready=trace_handler,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
        # with_flops=True
    ) as p:
        for i in range(1):
            out_torch = interpreter.run(*inputs)
            # out_torch = model(*inputs)
            p.step()
    return 

    
