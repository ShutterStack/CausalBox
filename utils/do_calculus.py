# utils/do_calculus.py
import pandas as pd
import numpy as np
import networkx as nx

class DoCalculus:
    def __init__(self, graph):
        self.graph = graph

    def intervene(self, data, intervention_var, intervention_value):
        """
        Simulate do(X=x) intervention on a variable.
        Returns intervened DataFrame.
        This is a simplified implementation.
        """
        intervened_data = data.copy()
        
        # Direct intervention: set the value
        intervened_data[intervention_var] = intervention_value

        # Propagate effects (simplified linear model) - needs graph
        # For a true do-calculus, you'd prune the graph and re-estimate based on parents
        # For demonstration, this still uses a simplified propagation.
        try:
            # Ensure graph is connected and topological sort is possible
            if self.graph and not nx.is_directed_acyclic_graph(self.graph):
                print("Warning: Graph is not a DAG. Topological sort may fail or be incorrect for do-calculus.")
            
            # This simplified propagation is a conceptual placeholder
            for node in nx.topological_sort(self.graph):
                if node == intervention_var:
                    continue # Do not propagate back to the intervened variable
                
                parents = list(self.graph.predecessors(node))
                if parents:
                    # Very simplified linear model to show propagation
                    # In reality, you'd use learned coefficients or structural equations
                    combined_effect = np.zeros(len(intervened_data))
                    for p in parents:
                        if p in intervened_data.columns:
                            # Use a fixed random coefficient for demonstration
                            coeff = 0.5 
                            combined_effect += intervened_data[p].to_numpy() * coeff
                    
                    # Add a small random noise to simulate uncertainty
                    intervened_data[node] += combined_effect + np.random.normal(0, 0.1, len(intervened_data))
        except Exception as e:
            print(f"Could not perform full propagation due to graph issues or simplification: {e}")
            # Fallback to direct intervention only if graph logic fails
            pass # The direct intervention `intervened_data[intervention_var] = intervention_value` is already applied

        return intervened_data