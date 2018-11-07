import numpy as np


class Operation:

    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def compute(self, x_var, y_var):
        pass


class Add(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])
        self.inputs = []

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class Multiply(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])
        self.inputs = []

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


class MatMul(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])
        self.inputs = []

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)


class Placeholder:

    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)


class Variable:

    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)


class Graph:

    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self


def traverse_post_order(operation):
    """
    Post-order Traversal of nodes.
    Makes sure computations are done in the correct order (Ax first, then Ax + b).
    :param operation:
    :return:
    """

    nodes_post_order = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_post_order.append(node)

    recurse(operation)

    return nodes_post_order


class Session:

    def run(self, operation, feed_dict={}):
        nodes_post_order = traverse_post_order(operation)

        for node in nodes_post_order:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else:
                # Operation
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)
        return operation.output
