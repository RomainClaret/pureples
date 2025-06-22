"""
Varying visualisation tools.
"""

import pickle
import graphviz
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import os
from graphviz import Digraph

# Try to import pygraphviz, but make it optional
try:
    import pygraphviz as pgv
    HAS_PYGRAPHVIZ = True
except ImportError:
    pgv = None
    HAS_PYGRAPHVIZ = False


def draw_net(net, filename=None, node_names={}, node_colors={}):
    """
    Draw neural network with arbitrary topology.
    """
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph('svg', node_attr=node_attrs)

    inputs = set()
    for k in net.input_nodes:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box',
                       'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in net.output_nodes:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled',
                      'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, _attributes=node_attrs)

    for node, _, _, _, _, links in net.node_evals:
        for i, w in links:
            node_input, output = node, i
            a = node_names.get(output, str(output))
            b = node_names.get(node_input, str(node_input))
            style = 'solid'
            color = 'green' if w > 0.0 else 'red'
            width = str(0.1 + abs(w / 5.0))
            dot.edge(a, b, _attributes={
                     'style': style, 'color': color, 'penwidth': width})

    dot.render(filename)

    return dot


def onclick(event):
    """
    Click handler for weight gradient created by a CPPN. Will re-query with the clicked coordinate.
    """
    plt.close()
    x = event.xdata
    y = event.ydata

    path_to_cppn = "es_hyperneat_xor_small_cppn.pkl"
    # For now, path_to_cppn should match path in test_cppn.py, sorry.
    with open(path_to_cppn, 'rb') as cppn_input:
        cppn = pickle.load(cppn_input)
        from pureples.es_hyperneat.es_hyperneat import find_pattern
        pattern = find_pattern(cppn, (x, y))
        draw_pattern(pattern)


def draw_pattern(im, res=60):
    """
    Draws the pattern/weight gradient queried by a CPPN.
    """
    fig = plt.figure()
    plt.axis([-1, 1, -1, 1])
    fig.add_subplot(111)

    a = range(res)
    b = range(res)

    for x in a:
        for y in b:
            px = -1.0 + (x/float(res))*2.0+1.0/float(res)
            py = -1.0 + (y/float(res))*2.0+1.0/float(res)
            c = str(0.5-im[x][y]/float(res))
            plt.plot(px, py, marker='s', color=c)

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.grid()
    plt.show()


def draw_es(id_to_coords, connections, filename):
    """
    Draw the net created by ES-HyperNEAT
    """
    fig = plt.figure()
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    fig.add_subplot(111)

    for c in connections:
        color = 'red'
        if c.weight > 0.0:
            color = 'black'
        plt.arrow(c.x1, c.y1, c.x2-c.x1, c.y2-c.y1, head_width=0.00, head_length=0.0,
                  fc=color, ec=color, length_includes_head=True)

    for (coord, _) in id_to_coords.items():
        plt.plot(coord[0], coord[1], marker='o', markersize=8.0, color='grey')

    plt.grid()
    fig.savefig(filename)


def draw_net2(net, filename=None, node_names={}, node_colors={}, fast_compile=False):
    """
    Draw neural network with arbitrary topology using pygraphviz.
    """
    if not HAS_PYGRAPHVIZ:
        raise ImportError("pygraphviz is required for draw_net2. Install it with: pip install pygraphviz")
    
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = pgv.AGraph(directed=True, strict=True, node_attr=node_attrs)

    inputs = set()
    for k in net.input_nodes:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box',
                       'fillcolor': node_colors.get(k, 'lightgray')}
        dot.add_node(name, **input_attrs)

    outputs = set()
    for k in net.output_nodes:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled',
                      'fillcolor': node_colors.get(k, 'lightblue')}
        dot.add_node(name, **node_attrs)

    for node, _, _, _, _, links in net.node_evals:
        for i, w in links:
            node_input, output = node, i
            a = node_names.get(output, str(output))
            b = node_names.get(node_input, str(node_input))
            style = 'solid'
            color = 'green' if w > 0.0 else 'red'
            width = str(0.1 + abs(w / 5.0))
            dot.add_edge(a, b, style=style, color=color, penwidth=width)

    if fast_compile:
        dot.layout(prog='neato')
    else:
        dot.layout(prog='dot')

    # Get the bounding box of the layout
    bb = dot.graph_attr["bb"]

    # Extract the bounding box dimensions
    bb_x1, bb_y1, bb_x2, bb_y2 = [float(val) for val in bb.split(",")]
    bb_width = bb_x2 - bb_x1
    bb_height = bb_y2 - bb_y1

    # Calculate the graph size (in inches) based on the bounding box dimensions
    graph_width = math.ceil(bb_width / 72)  # 72 points per inch
    graph_height = math.ceil(bb_height / 72)  # 72 points per inch

    # Set the size of the graph
    dot.graph_attr.update(size="{},{}".format(graph_width, graph_height))

    if filename:
        dot.draw(filename)

    return dot


def count_edges(model):
    """
    1) If needed, populate model.node_info, model.used_input_nodes,
       model.dependencies and model.dependents from model.node_evals.
    2) Count incoming and outgoing edges for each node in model.node_info.
    Returns (in_edges, out_edges) as defaultdict(int).
    """
    # --- 1) Bootstrap the network metadata if missing ---
    # We check for node_info being empty or absent
    if not hasattr(model, 'node_info') or not model.node_info:
        model.node_info        = {}
        model.used_input_nodes = set()
        model.dependencies     = defaultdict(set)
        model.dependents       = defaultdict(set)

        for node, activation_func, agg_func, bias, response, links in model.node_evals:
            # Store the raw info dict
            model.node_info[node] = {
                'links':           links,
                'bias':            bias,
                'response':        response,
                'activation_func': activation_func,
                'agg_func':        agg_func
            }
            # Track which inputs are actually used
            for inp, _ in links:
                if inp in model.input_nodes:
                    model.used_input_nodes.add(inp)
                else:
                    model.dependencies[node].add(inp)
                    model.dependents[inp].add(node)

    # --- 2) Now count edges ---
    in_edges  = defaultdict(int)
    out_edges = defaultdict(int)

    for node, info in model.node_info.items():
        for input_node, _ in info['links']:
            in_edges[node]      += 1
            out_edges[input_node] += 1

    return in_edges, out_edges


def format_node_label(prefix, node_id, in_count=None, out_count=None, funcs=None):
    """Format a node's label with id, edge counts, and functions."""
    lines = [f"{prefix}_{node_id}"]
    if in_count is not None:
        lines.append(f"in: {in_count}")
    if out_count is not None:
        lines.append(f"out: {out_count}")

    if funcs:
        # Activation
        act = funcs.get('activation')
        if act is not None:
            if isinstance(act, str):
                name = act.replace('_activation', '')
            elif callable(act):
                # use the function's name
                name = act.__name__.replace('_activation', '')
            else:
                name = str(act)
            lines.append(f"act: {name}")

        # Aggregation
        agg = funcs.get('agg')
        if agg is not None:
            if isinstance(agg, str):
                agg_name = agg
            elif callable(agg):
                agg_name = agg.__name__
            else:
                agg_name = str(agg)
            lines.append(f"agg: {agg_name}")

    return "\n".join(lines)


def draw_net3(net, filename=None, node_names={}, node_colors={}, fast_compile=False):
    """
    Draw neural network with arbitrary topology using pygraphviz, with structured layers.
    """
    if not HAS_PYGRAPHVIZ:
        raise ImportError("pygraphviz is required for draw_net3. Install it with: pip install pygraphviz")
    
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'
    }

    dot = pgv.AGraph(directed=True, strict=False, node_attr=node_attrs, rankdir='LR')

    # Determine used input nodes by checking connections
    used_inputs = set()
    for _, _, _, _, _, links in net.node_evals:
        for i, _ in links:
            if i in net.input_nodes:
                used_inputs.add(i)

    # Create input nodes, only if they are used
    for k in net.input_nodes:
        if k in used_inputs:
            name = node_names.get(k, str(k))
            input_attrs = {
                'style': 'filled',
                'shape': 'box',
                'fillcolor': node_colors.get(k, 'lightgray')
            }
            dot.add_node(name, **input_attrs)
            dot.add_subgraph([name], name='cluster_input', rank='same')

    # Create output nodes
    for k in net.output_nodes:
        name = node_names.get(k, str(k))
        output_attrs = {
            'style': 'filled',
            'fillcolor': node_colors.get(k, 'lightblue')
        }
        dot.add_node(name, **output_attrs)
        dot.add_subgraph([name], name='cluster_output', rank='same')

    # Create hidden nodes and connections
    hidden_nodes = set()
    for node, _, _, _, _, links in net.node_evals:
        if node not in net.input_nodes and node not in net.output_nodes:
            hidden_nodes.add(node)
            name = node_names.get(node, str(node))
            dot.add_node(name, **node_attrs)
        for i, w in links:
            if i in used_inputs or i in hidden_nodes or i in net.output_nodes:
                node_input, output = node, i
                a = node_names.get(output, str(output))
                b = node_names.get(node_input, str(node_input))
                style = 'solid'
                color = 'green' if w > 0.0 else 'red'
                width = str(0.1 + abs(w / 5.0))
                dot.add_edge(a, b, style=style, color=color, penwidth=width)

    # Group hidden nodes by layers if possible
    hidden_layers = {}
    for node, _, _, _, _, links in net.node_evals:
        if node not in net.input_nodes and node not in net.output_nodes:
            # Count the number of connections to determine the layer
            layer = 0
            for _, _, _, _, _, other_links in net.node_evals:
                if any(link[0] == node for link in other_links):
                    layer += 1
            if layer not in hidden_layers:
                hidden_layers[layer] = []
            hidden_layers[layer].append(str(node))

    for layer in hidden_layers:
        dot.add_subgraph(hidden_layers[layer], name='cluster_{}'.format(layer), rank='same')

    # Graph layout
    if fast_compile:
        dot.layout(prog='neato')
    else:
        dot.layout(prog='dot')

    # Set the graph size
    dot.graph_attr.update(size="{},{}".format(8, 8))

    if filename:
        dot.draw(filename)

    return dot


def draw_net4(net, filename=None, node_names={}, node_colors={}, fast_compile=False):
    """
    Draw neural network with arbitrary topology using pygraphviz, with structured layers.
    """
    if not HAS_PYGRAPHVIZ:
        raise ImportError("pygraphviz is required for draw_net4. Install it with: pip install pygraphviz")
    
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'
    }

    dot = pgv.AGraph(directed=True, strict=False, node_attr=node_attrs, rankdir='LR')

    # Determine used input nodes by checking connections
    used_inputs = set()
    for _, _, _, _, _, links in net.node_evals:
        for i, _ in links:
            if i in net.input_nodes:
                used_inputs.add(i)

    # Create input nodes, only if they are used
    for k in net.input_nodes:
        if k in used_inputs:
            name = node_names.get(k, str(k))
            input_attrs = {
                'style': 'filled',
                'shape': 'box',
                'fillcolor': node_colors.get(k, 'lightgray')
            }
            dot.add_node(name, **input_attrs)

    # Create output nodes
    for k in net.output_nodes:
        name = node_names.get(k, str(k))
        output_attrs = {
            'style': 'filled',
            'fillcolor': node_colors.get(k, 'lightblue')
        }
        dot.add_node(name, **output_attrs)

    # Create hidden nodes and connections
    hidden_nodes = set()
    for node, _, _, _, _, links in net.node_evals:
        if node not in net.input_nodes and node not in net.output_nodes:
            hidden_nodes.add(node)
            name = node_names.get(node, str(node))
            dot.add_node(name, **node_attrs)
        for i, w in links:
            if i in used_inputs or i in hidden_nodes or i in net.output_nodes:
                node_input, output = node, i
                a = node_names.get(node_input, str(node_input))
                b = node_names.get(output, str(output))
                style = 'solid'
                color = 'green' if w > 0.0 else 'red'
                width = str(0.1 + abs(w / 5.0))
                dot.add_edge(a, b, style=style, color=color, penwidth=width)

    # Group hidden nodes by layers if possible
    hidden_layers = {}
    for node in hidden_nodes:
        # Count the number of connections to determine the layer
        layer = 0
        for _, _, _, _, _, links in net.node_evals:
            if any(link[0] == node for link in links):
                layer += 1
        if layer not in hidden_layers:
            hidden_layers[layer] = []
        hidden_layers[layer].append(node)

    # Add nodes to layers
    for layer, nodes in hidden_layers.items():
        with dot.subgraph(name='cluster_' + str(layer)) as c:
            c.graph_attr['rank'] = 'same'
            for node in nodes:
                c.add_node(node_names.get(node, str(node)))

    # Graph layout
    if fast_compile:
        dot.layout(prog='neato')
    else:
        dot.layout(prog='dot')

    # Set the graph size
    dot.graph_attr.update(size="{},{}".format(8, 8))

    if filename:
        dot.draw(filename)

    return dot


def draw_net5(net, filename=None, node_names={}, node_colors={}, fast_compile=False):
    """
    Draw neural network with arbitrary topology using pygraphviz, with structured layers.
    """
    if not HAS_PYGRAPHVIZ:
        raise ImportError("pygraphviz is required for draw_net5. Install it with: pip install pygraphviz")
    
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'
    }

    dot = pgv.AGraph(directed=True, strict=False, node_attr=node_attrs, rankdir='LR')

    # Determine used input nodes by checking connections
    used_inputs = set()
    for _, _, _, _, _, links in net.node_evals:
        for i, _ in links:
            if i in net.input_nodes:
                used_inputs.add(i)

    # Create input nodes, only if they are used
    for k in net.input_nodes:
        if k in used_inputs:
            name = node_names.get(k, str(k))
            input_attrs = {
                'style': 'filled',
                'shape': 'box',
                'fillcolor': node_colors.get(k, 'lightgray')
            }
            dot.add_node(name, **input_attrs)

    # Create output nodes
    for k in net.output_nodes:
        name = node_names.get(k, str(k))
        output_attrs = {
            'style': 'filled',
            'fillcolor': node_colors.get(k, 'lightblue')
        }
        dot.add_node(name, **output_attrs)

    # Create hidden nodes and connections
    hidden_nodes = set()
    for node, _, _, _, _, links in net.node_evals:
        if node not in net.input_nodes and node not in net.output_nodes:
            hidden_nodes.add(node)
            name = node_names.get(node, str(node))
            dot.add_node(name, **node_attrs)
        for i, w in links:
            if i in used_inputs or i in hidden_nodes or i in net.output_nodes:
                node_input, output = node, i
                a = node_names.get(node_input, str(node_input))
                b = node_names.get(output, str(output))
                style = 'solid'
                color = 'green' if w > 0.0 else 'red'
                width = str(0.1 + abs(w / 5.0))
                dot.add_edge(a, b, style=style, color=color, penwidth=width)

    # Create a subgraph for input nodes to ensure they are on the left
    dot.add_subgraph([node_names.get(k, str(k)) for k in used_inputs], name='cluster_input', rank='min')

    # Create a subgraph for output nodes to ensure they are on the right
    dot.add_subgraph([node_names.get(k, str(k)) for k in net.output_nodes], name='cluster_output', rank='max')

    # Graph layout
    if fast_compile:
        dot.layout(prog='neato')
    else:
        dot.layout(prog='dot')

    # Set the graph size
    dot.graph_attr.update(size="{},{}".format(8, 8))

    if filename:
        dot.draw(filename)

    return dot


def draw_net6(net, filename=None, node_names={}, node_colors={}, fast_compile=False, ranksep='1.0'):
    """
    Draw neural network with arbitrary topology using pygraphviz, with structured layers.
    """
    if not HAS_PYGRAPHVIZ:
        raise ImportError("pygraphviz is required for draw_net6. Install it with: pip install pygraphviz")
    
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'
    }

    dot = pgv.AGraph(directed=True, strict=False, node_attr=node_attrs, rankdir='LR')

    # Determine used input nodes by checking connections
    used_inputs = set()
    for _, _, _, _, _, links in net.node_evals:
        for i, _ in links:
            if i in net.input_nodes:
                used_inputs.add(i)

    # Create input nodes, only if they are used
    for k in net.input_nodes:
        if k in used_inputs:
            name = node_names.get(k, str(k))
            input_attrs = {
                'style': 'filled',
                'shape': 'box',
                'fillcolor': node_colors.get(k, 'lightgray')
            }
            dot.add_node(name, **input_attrs)

    # Create output nodes
    for k in net.output_nodes:
        name = node_names.get(k, str(k))
        output_attrs = {
            'style': 'filled',
            'fillcolor': node_colors.get(k, 'lightblue')
        }
        dot.add_node(name, **output_attrs)

    # Create hidden nodes and connections
    hidden_nodes = set()
    for node, _, _, _, _, links in net.node_evals:
        if node not in net.input_nodes and node not in net.output_nodes:
            hidden_nodes.add(node)
            name = node_names.get(node, str(node))
            dot.add_node(name, **node_attrs)
        for i, w in links:
            if i in used_inputs or i in hidden_nodes or i in net.output_nodes:
                node_input, output = node, i
                a = node_names.get(node_input, str(node_input))
                b = node_names.get(output, str(output))
                style = 'solid'
                color = 'green' if w > 0.0 else 'red'
                width = str(0.1 + abs(w / 5.0))
                dot.add_edge(b, a, style=style, color=color, penwidth=width)  # Reversed the order of a and b

    # Create a subgraph for input nodes to ensure they are on the left
    dot.add_subgraph([node_names.get(k, str(k)) for k in used_inputs], name='cluster_input', rank='source')

    # Create a subgraph for output nodes to ensure they are on the right
    dot.add_subgraph([node_names.get(k, str(k)) for k in net.output_nodes], name='cluster_output', rank='sink')

    # Set space between layers
    dot.graph_attr.update(ranksep=ranksep)

    # Graph layout
    if fast_compile:
        dot.layout(prog='neato')
    else:
        dot.layout(prog='dot')

    # Set the graph size
    dot.graph_attr.update(size="{},{}".format(8, 8))

    if filename:
        dot.draw(filename)

    return dot


def draw_net7(net, filename=None, node_names={}, node_colors={}, fast_compile=False, ranksep='2.0'):
    """
    Build a Graphviz Digraph for `net`, optionally write it out, and return the Digraph object.
    """
    # prepare graph
    dot = Digraph(comment='Neural Network Topology')
    dot.attr(rankdir='LR')
    dot.attr(ranksep='2.0')
    
    # count in/out edges
    in_edges, out_edges = count_edges(net)

    # INPUT LAYER
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Layer')
        c.attr('node', style='filled', color='lightblue', shape='box')
        for idx in sorted(list(net.used_input_nodes)):
            label = format_node_label("In", idx, out_count=out_edges[idx])
            c.node(f'input_{idx}', label)
    
    # HIDDEN LAYER
    hidden = sorted([n for n in net.node_info.keys() if n not in net.output_nodes])
    with dot.subgraph(name='cluster_hidden') as c:
        c.attr(label='Hidden Layer')
        c.attr('node', style='filled', color='lightgreen', shape='box')
        for node in hidden:
            funcs = {
                'activation': net.node_info[node]['activation_func'],
                'agg': net.node_info[node]['agg_func']
            }
            label = format_node_label("H", node,
                                     in_count=in_edges[node],
                                     out_count=out_edges[node],
                                     funcs=funcs)
            c.node(f'hidden_{node}', label)
    
    # OUTPUT LAYER
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output Layer')
        c.attr('node', style='filled', color='lightpink', shape='box')
        for node in sorted(net.output_nodes):
            if node in net.node_info:
                funcs = {
                    'activation': net.node_info[node]['activation_func'],
                    'agg': net.node_info[node]['agg_func']
                }
                label = format_node_label("Out", node,
                                         in_count=in_edges[node],
                                         out_count=out_edges[node],
                                         funcs=funcs)
            else:
                label = format_node_label("Out", node) + "\n(no connections)"
            c.node(f'output_{node}', label)
    
    # Add edges between layers
    edge_count = defaultdict(int)
    max_edges_shown = None

    # Collect and sort all connections
    all_connections = []
    for node, info in net.node_info.items():
        for input_node, weight in info['links']:
            all_connections.append((input_node, node, weight))
    
    # Sort by weight magnitude
    sorted_connections = sorted(all_connections, key=lambda x: abs(x[2]), reverse=True)
    if max_edges_shown is not None:
        shown_connections = sorted_connections[:max_edges_shown]
    else:
        shown_connections = sorted_connections
    
    # Add edges with better spacing
    for input_node, target_node, weight in shown_connections:
        if input_node in net.input_nodes:
            src = f'input_{input_node}'
            if target_node in net.output_nodes:
                dst = f'output_{target_node}'
            else:
                dst = f'hidden_{target_node}'
        else:
            src = f'hidden_{input_node}'
            if target_node in net.output_nodes:
                dst = f'output_{target_node}'
            else:
                dst = f'hidden_{target_node}'
                
        if edge_count[src] < 5:  # Limit edges per node for clarity
            dot.edge(src, dst, label=f'{weight:.1f}')
            edge_count[src] += 1

    # choose layout engine
    dot.engine = 'neato' if fast_compile else 'dot'

    # set overall graph size
    dot.graph_attr.update(size="8,8")

    # optionally write out the .gv (and whatever default format your Graphviz backend uses)
    if filename:
        # split off the extension if present
        base, ext = os.path.splitext(filename)
        fmt = ext.lstrip('.').lower() or 'pdf'       # default to pdf
        dot.format = fmt
        os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
        dot.render(base, cleanup=True)

    # hand back the raw Digraph for further use
    return dot
