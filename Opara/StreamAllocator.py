from torch.cuda.streams import Stream, Event

def assign_stream(graph):
    # print(len(graph.nodes))
    for node in graph.nodes:
        setattr(node, 'stream', None)
        setattr(node, 'event', None)
        setattr(node, 'event_to_wait', [])
    streams, events = [], []
    for node in graph.nodes:
        node.event = Event()
        events.append(node.event)
        for input_node in node.all_input_nodes:
            if node == list(input_node.users.keys())[0]:
                node.stream = input_node.stream
                break
        if node.stream is None:
            node.stream = Stream()
            streams.append(node.stream)
    for node in graph.nodes:
        for input_node in node.all_input_nodes:
            if node.stream != input_node.stream:
                if input_node.event not in node.event_to_wait:
                    node.event_to_wait.append(input_node.event)
    return streams, events