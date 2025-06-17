# utils/graph_utils.py
import networkx as nx
import plotly.graph_objects as go

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