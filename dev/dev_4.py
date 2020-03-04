import lxml
import lxml.html
import lxml.etree
import os
from collections import defaultdict
from graphviz import Digraph
import yaml


HIERARCHICAL = 'hierarchical'
SEQUENTIAL = 'sequential'


def open_doc(folder, filename):
    folder = os.path.abspath(folder)
    filepath = os.path.join(folder, filename)

    with open(filepath, 'r') as file:
        doc = lxml.html.fromstring(
            lxml.etree.tostring(
                lxml.html.parse(file), method='html'
            )
        )
    return doc


def html_to_dot_sequential_name(html, with_text=False):
    graph = Digraph(name='html')
    tag_counts = defaultdict(int)
    
    def add_node(html_node):
        tag = html_node.tag
        tag_sequential = tag_counts[tag]
        tag_counts[tag] += 1
        node_name = "{}-{}".format(tag, tag_sequential)
        graph.node(node_name, node_name)
        
        if len(html_node) > 0:
            for child in html_node.iterchildren():
                child_name = add_node(child)
                graph.edge(node_name, child_name)
        else:
            child_name = "-".join([node_name, "txt"])
            graph.node(child_name, html_node.text)
            graph.edge(node_name, child_name)
        return node_name
    add_node(html)
    return graph


def html_to_dot_hierarchical_name(html, with_text=False):
    graph = Digraph(name='html')
    
    def add_node(html_node, parent_suffix, brotherhood_index):
        tag = html_node.tag
        if parent_suffix is None and brotherhood_index is None:
            node_suffix = ""
            node_name = tag
        else:
            node_suffix = (
                "-".join([parent_suffix, str(brotherhood_index)]) 
                if parent_suffix else 
                str(brotherhood_index)
            )
            node_name = "{}-{}".format(tag, node_suffix)
        graph.node(node_name, node_name, path=node_suffix)
        
        if len(html_node) > 0:
            for child_index, child in enumerate(html_node.iterchildren()):
                child_name = add_node(child, node_suffix, child_index)
                graph.edge(node_name, child_name)
        else:
            child_name = "-".join([node_name, "txt"])
            child_path = "-".join([node_suffix, "txt"])
            graph.node(child_name, html_node.text, path=child_path)
            graph.edge(node_name, child_name)
        return node_name
    add_node(html, None, None)
    return graph


def html_to_dot(html, name_option='hierarchical', with_text=False):
    if name_option == SEQUENTIAL:
        return html_to_dot_sequential_name(html, with_text=with_text)
    elif name_option == HIERARCHICAL:
        return html_to_dot_hierarchical_name(html, with_text=with_text)
    else:
        raise Exception('No name option `{}`'.format(name_option))
        
        
def depth(node):
    d = 0
    while node is not None:
        d += 1
        node = node.getparent()
    return d


from similarity.normalized_levenshtein import NormalizedLevenshtein
import lxml
import lxml.etree
from pprint import pprint
from collections import defaultdict
from copy import deepcopy

normalized_levenshtein = NormalizedLevenshtein()
TAG_NAME_ATTRIB = '___tag_name___'


class MDR:

    MINIMUM_DEPTH = 3

    def __init__(self, max_tag_per_gnode, verbose=False):
        self.max_tag_per_gnode = max_tag_per_gnode
        self.verbose = verbose

    def _debug(self, msg, tabs=0):
        if self.verbose:
            if type(msg) == str:
                print(tabs * '\t' + msg)
            else:
                pprint(msg)

    def __call__(self, node):
        # when in the root
        if node.getparent() is None:
            self.distances = {}
            self.tag_counts = defaultdict(int)
            self.shallow_html_copy = deepcopy(node)

        # each tag is named sequentially
        tag = node.tag
        tag_name = "{}-{}".format(tag, self.tag_counts[tag])
        self.tag_counts[tag] += 1

        self._debug(tag_name)

        node_depth = depth(node)

        if node_depth >= MDR.MINIMUM_DEPTH:
            # get all possible distances of the n-grams of children
            distances = self.compare_combinations(node.getchildren())

            self._debug("{} - distances".format(tag_name))
            self._debug(distances)
            
            # !!! ATTENTION !!! this modifies the input HTML 
            # it is important that this comes after `compare_combinations` because 
            # otherwise the edit distances would change
            # todo: remember, in the last phase, to clear the `TAG_NAME_ATTRIB` from all tags
            node.set(TAG_NAME_ATTRIB, tag_name)
            self.distances[tag_name] = distances

        self._debug("\n\n")

        for child in node:
            self(child)

    def compare_combinations(self, node_list):
        """
        Notation: gnode = "generalized node"

        :param node_list:
        :return:
        """

        self._debug("in compare_combinations")

        if not node_list:
            return {}

        # version 1: {gnode_size: {((,), (,)): float}}
#         distances = defaultdict(dict)  
        # version 2: {gnode_size: {starting_tag: {{ ((,), (,)): float }}}}
        distances = defaultdict(lambda: defaultdict(dict))  
        
        n_nodes = len(node_list)

        # for (i = 1; i <= K; i++)  /* start from each node */
        for starting_tag in range(1, self.max_tag_per_gnode + 1):
            self._debug('starting_tag (i): {}'.format(starting_tag), 1)

            # for (j = i; j <= K; j++) /* comparing different combinations */
            for gnode_size in range(starting_tag, self.max_tag_per_gnode + 1):  # j
                self._debug('gnode_size (j): {}'.format(gnode_size), 2)

                # if NodeList[i+2*j-1] exists then
                if (starting_tag + 2 * gnode_size - 1) < n_nodes + 1:  # +1 for pythons open set notation
                    self._debug(" ")
                    self._debug(">>> if 1 <<<", 3)

                    left_gnode_start = starting_tag - 1  # st

                    # for (k = i+j; k < Size(NodeList); k+j)
                    # for k in range(i + j, n, j):
                    for right_gnode_start in range(starting_tag + gnode_size - 1, n_nodes, gnode_size):  # k
                        self._debug('left_gnode_start (st): {}'.format(left_gnode_start), 4)
                        self._debug('right_gnode_start (k): {}'.format(right_gnode_start), 4)

                        # if NodeList[k+j-1] exists then
                        if right_gnode_start + gnode_size < n_nodes + 1:
                            self._debug(" ")
                            self._debug(">>> if 2 <<<", 5)
                            # todo: avoid recomputing strings?
                            # todo: avoid recomputing edit distances?
                            # todo: check https://pypi.org/project/strsim/ ?

                            # NodeList[St..(k-1)]
                            left_gnode_indices = (left_gnode_start, right_gnode_start)
                            left_gnode = node_list[left_gnode_indices[0]:left_gnode_indices[1]]
                            left_gnode_str = MDR.gnode_to_string(left_gnode)
                            self._debug('left_gnode_indices: {}'.format(left_gnode_indices), 5)

                            # NodeList[St..(k-1)]
                            right_gnode_indices = (right_gnode_start, right_gnode_start + gnode_size)
                            right_gnode = node_list[right_gnode_indices[0]:right_gnode_indices[1]]
                            right_gnode_str = MDR.gnode_to_string(right_gnode)
                            self._debug('right_gnode_indices: {}'.format(right_gnode_indices), 5)

                            # edit distance
                            edit_distance = normalized_levenshtein.distance(left_gnode_str, right_gnode_str)
                            self._debug('edit_distance: {}'.format(edit_distance), 5)
                            
                            # version 1
#                             distances[gnode_size][(left_gnode_indices, right_gnode_indices)] = edit_distance
                            # version 2
                            distances[gnode_size][starting_tag][
                                (left_gnode_indices, right_gnode_indices)
                            ] = edit_distance
    
    
                            left_gnode_start = right_gnode_start
                        else:
                            self._debug("skipped\n", 5)
                        self._debug(' ')
                else:
                    self._debug("skipped\n", 3)
                self._debug(' ')
                
        # version 1
#         return dict(distances)
        # version 2
        return {k: dict(v) for k, v in distances.items()}
                            
    @staticmethod
    def gnode_to_string(list_of_nodes):
        return " ".join([
            lxml.etree.tostring(child).decode('utf-8') for child in list_of_nodes
        ])
