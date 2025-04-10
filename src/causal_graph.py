import os
import pickle
import graphviz

class CausalGraph:
    def __init__(self, fragment_id, ego_id, output_dir):
        """
        Initialize the CausalGraph class.

        Args:
            fragment_id: ID of the scenario fragment.
            ego_id: ID of the ego vehicle.
            output_dir: Directory where outputs will be saved.
        """
        self.fragment_id = fragment_id
        self.ego_id = ego_id
        self.output_dir = output_dir
        self.graph = {}
        self.first_critical_frames = {}
    
    def add_edge(self, source, target, ssm_type, critical_frames):
        """
        Add an edge to the causal graph.

        Args:
            source: Source vehicle ID.
            target: Target vehicle ID.
            ssm_type: Type of the surrogate safety metric.
            critical_frames: List of critical frames for this edge.
        """
        if target not in self.graph:
            self.graph[target] = []
        self.graph[target].append((source, ssm_type, critical_frames))

    def save_graph(self):
        """
        Save the causal graph to a file.
        """
        save_path = os.path.join(self.output_dir, f"{self.fragment_id}_{self.ego_id}")
        os.makedirs(save_path, exist_ok=True)
        graph_file = os.path.join(save_path, f"causal_graph_{self.fragment_id}_{self.ego_id}.pkl")
        with open(graph_file, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"Causal graph saved to {graph_file}")

    def visualize(self, save_pic=True, save_pdf=False):
        """
        Visualize the causal graph using Graphviz.
        
        Args:
            save_pic: Whether to save the graph as a PNG.
            save_pdf: Whether to save the graph as a PDF.
        """
        dot = graphviz.Digraph(comment=f'Causal Graph for Fragment {self.fragment_id}, Ego {self.ego_id}')
        dot.attr(rankdir='LR', size='12,8', dpi='300', fontname='Arial', bgcolor='white', concentrate='true')
        
        # Add nodes and edges
        for agent, influences in self.graph.items():
            for source, ssm_type, critical_frames in influences:
                dot.node(str(agent), str(agent))
                dot.node(str(source), str(source))
                dot.edge(str(source), str(agent), label=f"{ssm_type}: {critical_frames[0]}-{critical_frames[-1]}")
        
        if save_pic:
            dot.render(os.path.join(self.output_dir, f"causal_graph_{self.fragment_id}_{self.ego_id}"), format='png', cleanup=True)
            print(f"Causal graph saved as PNG.")
        if save_pdf:
            dot.render(os.path.join(self.output_dir, f"causal_graph_{self.fragment_id}_{self.ego_id}"), format='pdf', cleanup=True)
            print(f"Causal graph saved as PDF.")

    def load_graph(self):
        """
        Load the causal graph from a file.
        
        Returns:
            The loaded causal graph.
        """
        graph_file = os.path.join(self.output_dir, f"{self.fragment_id}_{self.ego_id}/causal_graph_{self.fragment_id}_{self.ego_id}.pkl")
        
        if not os.path.exists(graph_file):
            print(f"Causal graph file not found: {graph_file}")
            return None
        
        with open(graph_file, "rb") as f:
            self.graph = pickle.load(f)
        print(f"Successfully loaded causal graph: {graph_file}")
        return self.graph