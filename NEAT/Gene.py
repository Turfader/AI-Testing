from enum import Enum, auto
import uuid


class NodeType(Enum):
    INPUT = auto()
    HIDDEN = auto()
    OUTPUT = auto()
    BIAS = auto()


class NodeGene:
    def __init__(self, id, node_type):
        self.id = id
        self.type = node_type
        self.value = 0.0  # for forward pass

    def __repr__(self):
        return f"NodeGene(id={self.id}, type={self.type.name})"


class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled, innovation_number):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number

    def __repr__(self):
        return (f"ConnGene({self.in_node} -> {self.out_node}, "
                f"w={self.weight:.2f}, enabled={self.enabled}, "
                f"innov={self.innovation_number})")
