import numpy as np
import numpy.typing as npt

from scipy import sparse


class Tree:

    def __init__(self, node: int):
        self.node = node
        self.children: list[Tree] = []
        self.weight = None

    def add_child(self, child):
        self.children.append(child)


def neighbours(orig_node: int, adj_matrix: sparse.csr_matrix) -> npt.NDArray:
    return adj_matrix[orig_node].nonzero()[1]


def n_neighbours(orig_node: int, adj_matrix: sparse.csr_matrix) -> int:
    return adj_matrix[orig_node].count_nonzero()


def ij_to_node(i: int, j: int, n_cols: int) -> int:
    return (j + i * n_cols) + 2

# build sparse adjacency matrix


def build_adj_matrix(vacancy_matrix: npt.NDArray) -> sparse.csr_matrix:
    n_rows, n_cols = vacancy_matrix.shape
    # n_
    n_nodes = n_rows * n_cols + 2

    adj_matrix = sparse.lil_matrix((n_nodes, n_nodes), dtype=int)

    connections = [None] * 4
    for i in range(n_rows):
        for j in range(n_cols):
            if vacancy_matrix[i, j] == 1:
                node_idx = ij_to_node(i, j, n_cols)
                # check if there is a vacancy to the right
                connections[0] = None
                if j < n_cols - 1 and vacancy_matrix[i, j + 1] == 1:
                    connections[0] = ij_to_node(i, j + 1, n_cols)

                # check if there is a vacancy to the left
                connections[1] = None
                if j > 0 and vacancy_matrix[i, j - 1] == 1:
                    connections[1] = ij_to_node(i, j - 1, n_cols)

                # check if there is a vacancy on the node below
                connections[2] = None
                if i < n_rows - 1 and vacancy_matrix[i + 1, j] == 1:
                    connections[2] = ij_to_node(i + 1, j, n_cols)

                # check if there is a vacancy on the node above
                connections[3] = None
                if i > 0 and vacancy_matrix[i - 1, j] == 1:
                    connections[3] = ij_to_node(i - 1, j, n_cols)

                if j == 0:
                    connections[1] = 0
                if j == n_cols - 1:
                    connections[0] = 1

                for c in connections:
                    if c is not None:
                        # add edge to adjacency matrix
                        adj_matrix[node_idx, c] = 1
                        adj_matrix[c, node_idx] = 1

    return adj_matrix.tocsr()


def remove_edges(edges: list[(int, int)], adj_matrix: sparse.csr_matrix):
    for edge in edges:
        adj_matrix[edge[0], edge[1]] = 0
        adj_matrix[edge[1], edge[0]] = 0


def remove_loose_edges_without_0_or_1(adj_matrix: sparse.csr_matrix):
    n_nodes = adj_matrix.shape[0]

    while True:
        loose_nodes = []

        for node in range(n_nodes):
            if node == 0 or node == 1:
                continue
            # get num of neighbors
            if n_neighbours(node, adj_matrix) == 1:
                loose_nodes.append(node)

        if len(loose_nodes) == 0:
            break

        to_remove = []
        for node in loose_nodes:
            neighbours = adj_matrix[node].nonzero()[1]
            for neighbour in neighbours:
                to_remove.append((node, neighbour))

        remove_edges(to_remove, adj_matrix)


def vacancy_matrix_from_adj_matrix(adj_matrix: sparse.csr_matrix, n_rows: int, n_cols: int) -> npt.NDArray:
    n_nodes = n_rows * n_cols
    vacancy_matrix = np.zeros((n_rows, n_cols), dtype=int)

    for i in range(n_rows):
        for j in range(n_cols):
            node_idx = ij_to_node(i, j, n_cols)
            if n_neighbours(node_idx, adj_matrix) > 0:
                vacancy_matrix[i, j] = 1

    return vacancy_matrix


def pretty_print_tree(tree: Tree):

    def pretty_print_tree_aux(tree: Tree, level: int):
        print(" " * level + str(tree.node) + " (" + str(tree.weight) + ")")
        for child in tree.children:
            pretty_print_tree_aux(child, level + 1)

    pretty_print_tree_aux(tree, 0)


def find_path(vacancy_matrix: npt.NDArray) -> npt.NDArray:
    n_rows, n_cols = vacancy_matrix.shape
    n_nodes = n_rows * n_cols + 2

    adj_matrix = build_adj_matrix(vacancy_matrix)

    remove_loose_edges_without_0_or_1(adj_matrix)

    # return vacancy_matrix_from_adj_matrix(adj_matrix, n_rows, n_cols)

    adj_matrix_aux = adj_matrix.copy()

    n_nodes = adj_matrix.shape[0]

    start_node = 0
    end_node = 1

    visited = []
    to_visit = [start_node]

    weights = np.zeros(n_nodes)
    # tree = [[]] * n_nodes
    start_tree = Tree(start_node)
    trees = {start_node: start_tree}
    while len(to_visit) > 0:
        node = to_visit.pop()
        tree = trees[node]

        visited.append(node)
        if node == end_node:
            continue

        # unvisited_neighbours = neighbours_unvisited(node, visited, to_visit, adj_matrix_aux)
        neighs = neighbours(node, adj_matrix_aux)
        weights[node] = weights[node] - len(neighs)

        for neighbour in neighs:
            weights[neighbour] += 1

            if adj_matrix_aux[node, neighbour] == 1:
                adj_matrix_aux[node, neighbour] = 0
                adj_matrix_aux[neighbour, node] = 0

            # tree[node].append(neighbour)

        new_neighbours = list(set(neighs).difference(set(visited)).difference(set(to_visit)))
        for neighbour in new_neighbours:
            trees[neighbour] = Tree(neighbour)
            tree.add_child(trees[neighbour])

        to_visit.extend(new_neighbours)
        if end_node in to_visit:
            to_visit.remove(end_node)

    for node, weight in enumerate(weights):
        if node in trees:
            trees[node].weight = weight

    start_node_reverse = end_node
    end_node_reverse = start_node
    to_visit_reverse = [start_node_reverse]
    visited_reverse = []

    while len(to_visit_reverse) > 0:
        node = to_visit_reverse.pop()

        visited_reverse.append(node)
        if node == end_node_reverse:
            continue

        unvisited_neighbours = neighbours_unvisited(node, visited_reverse, to_visit_reverse, adj_matrix)
        to_visit_reverse.extend(unvisited_neighbours)

    visited_reverse = set(visited_reverse)
    visited = list(set(visited).intersection(visited_reverse))

    # print(f"Visited: {visited}")

    edges_to_remove = []
    for src_node in range(n_nodes):
        for dest_node in neighbours(src_node, adj_matrix_aux):
            weights[dest_node] += 1
            weights[src_node] -= 1

            # trees[src_node].weight -= 1
            # trees[dest_node].weight += 1

            # edges_to_remove.append((src_node, dest_node))

            adj_matrix_aux[src_node, dest_node] = 0
            adj_matrix_aux[dest_node, src_node] = 0

    for node, weight in enumerate(weights):
        if node in trees:
            trees[node].weight = weight

 # remove_edges(edges_to_remove, adj_matrix_aux)
 # pretty_print_tree(start_tree)
    deletefrom = []
    find_loops(0, start_tree, weights, adj_matrix, deletefrom)

    for node in deletefrom:
        to_remove = [c.node for c in trees[node].children]
        if not any([n == 0 or n == 1 for n in to_remove]):
            for loop_node in to_remove:
                if loop_node != node:
                    for neighbour in neighbours(loop_node, adj_matrix):
                        # if neighbour != node:
                        edges_to_remove.append((loop_node, neighbour))

    # print(edges_to_remove)

    remove_edges(edges_to_remove, adj_matrix)
    remove_loose_edges_without_0_or_1(adj_matrix)

    for node in range(n_nodes):
        if node not in visited:
            for neighbour in neighbours(node, adj_matrix):
                adj_matrix[node, neighbour] = 0
                adj_matrix[neighbour, node] = 0

    return vacancy_matrix_from_adj_matrix(adj_matrix, n_rows, n_cols)


def neighbours_unvisited(node: int, visited: list[int], to_visit: list[int], adj_matrix: sparse.csr_matrix) -> list[int]:
    neighbours_l = []
    for neighbour in neighbours(node, adj_matrix):
        if neighbour not in visited and neighbour not in to_visit:
            neighbours_l.append(neighbour)

    return neighbours_l


def find_loops(tree_node: int, tree: Tree, weights: list[int], adj_matrix: sparse.csr_matrix, deletefrom: list[int]) -> int:

    if tree_node == 1:
        # weight = 0
        return None
    else:
        # weight = tree.weight
        weight = weights[tree.node]
    # weight = tree.weight
    if len(tree.children) > 0:
        # for node in tree[tree_node]:
        for child in tree.children:
            w = find_loops(child.node, child, weights, adj_matrix, deletefrom)
            if w is not None:
                weight += w
            else:
                continue

        # if tree_node == 51:
        # print(f"Node {tree_node} has weight {weight} with weights[tree.node] = {weights[tree.node]}")

        if (weight == 1) and tree_node != 0 and tree_node != 1 and weights[tree.node] < 1.0:
            outside_connections = 0

            # childs = [c.node for c in tree.children]

            # for child in childs:

            #     if child != tree_node and any([(n in childs) for n in neighbours(child, adj_matrix)]):
            #         outside_connections += 1
            #         break

            # if outside_connections == 0:
            deletefrom.append(tree_node)

    return weight


def children(tree_node: int, tree: list[list[int]]) -> list[int]:
    this_children = []
    to_visit = [tree_node]
    while len(to_visit) > 0:
        node = to_visit.pop()

        this_children.append(node)

        for neighbour in tree[node]:
            to_visit.append(neighbour)

    return this_children
