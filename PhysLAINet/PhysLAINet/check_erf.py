import onnx

model = onnx.load("checkpoints/physlainet.onnx")
for node in model.graph.node:
    if "Erf" in node.op_type:
        print("Found Erf!")
for i, node in enumerate(model.graph.node):
    if "Erf" in node.op_type:
        print(f"Found Erf at node {i}: {node.name}")
        print("Inputs:", node.input)
        print("Outputs:", node.output)
        print("-" * 50)
output_to_node = {}
for node in model.graph.node:
    for out in node.output:
        output_to_node[out] = node

def trace_back(tensor_name, depth=3):
    for _ in range(depth):
        node = output_to_node.get(tensor_name)
        if node is None:
            break
        print("←", node.op_type, node.name)
        if len(node.input) > 0:
            tensor_name = node.input[0]

trace_back("/Div_output_0")