import random
from Gene import NodeGene, ConnectionGene, NodeType


class Genome:
    def __init__(self):
        self.nodes = {}         # {id: NodeGene}
        self.connections = {}   # {innovation_number: ConnectionGene}
        self.fitness = 0.0

    def add_node(self, node_gene):
        self.nodes[node_gene.id] = node_gene

    def add_connection(self, conn_gene):
        self.connections[conn_gene.innovation_number] = conn_gene

    def feed_forward(self, input_values):
        # Set input values
        input_nodes = [n for n in self.nodes.values() if n.type == NodeType.INPUT]
        for node, value in zip(input_nodes, input_values):
            node.value = value

        # Set bias (optional)
        for node in self.nodes.values():
            if node.type == NodeType.BIAS:
                node.value = 1.0

        # Compute values in topological order (simplified)
        for node in self.nodes.values():
            if node.type == NodeType.HIDDEN or node.type == NodeType.OUTPUT:
                incoming = [c for c in self.connections.values()
                            if c.out_node == node.id and c.enabled]
                total = 0.0
                for conn in incoming:
                    total += self.nodes[conn.in_node].value * conn.weight
                node.value = self.sigmoid(total)

        # Return outputs
        output_nodes = [n for n in self.nodes.values() if n.type == NodeType.OUTPUT]
        return [n.value for n in output_nodes]

    def sigmoid(self, x):
        return 1 / (1 + pow(2.71828, -x))

    def __repr__(self):
        return f"Genome(nodes={len(self.nodes)}, conns={len(self.connections)})"
