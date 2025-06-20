# utils/graph_utils.py
import networkx as nx
import plotly.graph_objects as go
import numpy as np

def visualize_graph(graph):
    """
    Visualize a causal graph using Plotly.
    Returns Plotly figure as JSON.
    """
    # Use a fixed seed for layout reproducibility (optional)
    pos = nx.spring_layout(graph, seed=42)
    
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        mode='lines',
        hoverinfo='none'
    )

    node_x, node_y = [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(graph.nodes()),
        textposition='bottom center',
        marker=dict(size=15, color='lightblue', line=dict(width=2, color='DarkSlateGrey')),
        hoverinfo='text'
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="Python Causal Graph",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y= -0.002,
                font=dict(size=14, color="lightgray")
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            title=dict(text="Causal Graph Visualization", font=dict(size=16)) # Corrected line
        )
    )
    return fig.to_json()

def get_graph_summary_for_chatbot(graph_adj, nodes):
    """
    Generates a text summary of the causal graph for the chatbot.
    """
    if not graph_adj or not nodes:
        return "No causal graph discovered yet."

    adj_matrix = np.array(graph_adj)
    G = nx.DiGraph(adj_matrix)
    
    # Relabel nodes with actual names
    mapping = {i: node_name for i, node_name in enumerate(nodes)}
    G = nx.relabel_nodes(G, mapping)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    summary = (
        f"The causal graph has {num_nodes} variables (nodes) and {num_edges} causal relationships (directed edges).\n"
        "The variables are: " + ", ".join(nodes) + ".\n"
    )
    
    # Add some basic structural info
    if nx.is_directed_acyclic_graph(G):
        summary += "The graph is a Directed Acyclic Graph (DAG), which is typical for causal models.\n"
    else:
        summary += "The graph contains cycles, which might indicate feedback loops or issues with the discovery algorithm for a DAG model.\n"
    
    # Smallest graphs: list all edges
    if num_edges > 0 and num_edges < 10: # Avoid listing too many edges for large graphs
        edge_list = [f"{u} -> {v}" for u, v in G.edges()]
        summary += "The discovered relationships are: " + ", ".join(edge_list) + ".\n"
    elif num_edges >= 10:
        summary += "There are many edges; you can ask for specific relationships (e.g., 'What are the direct causes of X?').\n"

    # Identify source and sink nodes (if any)
    source_nodes = [n for n, d in G.in_degree() if d == 0]
    sink_nodes = [n for n, d in G.out_degree() if d == 0]
    
    if source_nodes:
        summary += f"Variables with no known causes (source nodes): {', '.join(source_nodes)}.\n"
    if sink_nodes:
        summary += f"Variables with no known effects (sink nodes): {', '.join(sink_nodes)}.\n"
        
    return summary