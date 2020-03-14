import os
from collections import defaultdict, namedtuple
from copy import deepcopy
from pprint import pprint

import lxml
import lxml.html
import lxml.etree
from graphviz import Digraph
from similarity.normalized_levenshtein import NormalizedLevenshtein


normalized_levenshtein = NormalizedLevenshtein()
TAG_NAME_ATTRIB = '___tag_name___'
HIERARCHICAL = 'hierarchical'
SEQUENTIAL = 'sequential'

DataRegion = namedtuple(
    "DataRegion", ["n_nodes_per_region", "start_child_index", "n_nodes_covered"]
)


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


class MDR:

    MINIMUM_DEPTH = 3

    def __init__(self, max_tag_per_gnode, edit_distance_threshold, verbose=(False, False, False)):
        self.max_tag_per_gnode = max_tag_per_gnode
        self.edit_distance_threshold = edit_distance_threshold
        self._verbose = verbose
        self._phase = None

    def _debug(self, msg, tabs=0, force=False):
        if self._verbose[self._phase] or (any(self._verbose) and force):
            if type(msg) == str:
                print(tabs * '\t' + msg)
            else:
                pprint(msg)

    @staticmethod
    def depth(node):
        d = 0
        while node is not None:
            d += 1
            node = node.getparent()
        return d

    @staticmethod
    def gnode_to_string(list_of_nodes):
        return " ".join([
            lxml.etree.tostring(child).decode('utf-8') for child in list_of_nodes
        ])
    
    def __call__(self, root):
        self.distances = {}
        self.data_regions = {}
        self.tag_counts = defaultdict(int)
        self.root_copy = deepcopy(root)
        self._checked_data_regions = defaultdict(set)

        self._phase = 0
        self._debug(
            ">" * 20 + " COMPUTE DISTANCES PHASE ({}) ".format(self._phase) + "<" * 20, force=True
        )
        self._compute_distances(root)
        self._debug(
            "<" * 20 + " COMPUTE DISTANCES PHASE ({}) ".format(self._phase) + ">" * 20, force=True
        )
        self._debug("\n\nself.distances\n", force=True)
        self._debug(self.distances, force=True)
        self._debug("\n\n", force=True)

        self._phase = 1
        self._debug(
            ">" * 20 + " FIND DATA REGIONS PHASE ({}) ".format(self._phase) + "<" * 20, force=True
        )
        self._find_data_regions(root)
        self._debug(
            "<" * 20 + " FIND DATA REGIONS PHASE ({}) ".format(self._phase) + ">" * 20, force=True
        )

        self._phase = 2

    def _compute_distances(self, node):
        # each tag is named sequentially
        tag = node.tag
        tag_name = "{}-{}".format(tag, self.tag_counts[tag])
        self.tag_counts[tag] += 1

        self._debug("in _compute_distances of `{}`".format(tag_name))

        # todo: stock depth in attrib???
        node_depth = MDR.depth(node)

        if node_depth >= MDR.MINIMUM_DEPTH:
            # get all possible distances of the n-grams of children
            distances = self._compare_combinations(node.getchildren())

            self._debug("`{}` distances".format(tag_name))
            self._debug(distances)
        else:
            distances = None
            
        # !!! ATTENTION !!! this modifies the input HTML 
        # it is important that this comes after `compare_combinations` because 
        # otherwise the edit distances would change
        # todo: remember, in the last phase, to clear the `TAG_NAME_ATTRIB` from all tags
        node.set(TAG_NAME_ATTRIB, tag_name)
        self.distances[tag_name] = distances

        self._debug("\n\n")

        for child in node:
            self._compute_distances(child)

    def _compare_combinations(self, node_list):
        """
        Notation: gnode = "generalized node"

        :param node_list:
        :return:
        """

        self._debug("in _compare_combinations")

        if not node_list:
            return {}

        # version 1: {gnode_size: {((,), (,)): float}}
        distances = defaultdict(dict)
        # version 2: {gnode_size: {starting_tag: {{ ((,), (,)): float }}}}
        # distances = defaultdict(lambda: defaultdict(dict))
        
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
                            distances[gnode_size][(left_gnode_indices, right_gnode_indices)] = edit_distance
                            # version 2
                            # distances[gnode_size][starting_tag][
                            #     (left_gnode_indices, right_gnode_indices)
                            # ] = edit_distance
    
                            left_gnode_start = right_gnode_start
                        else:
                            self._debug("skipped\n", 5)
                        self._debug(' ')
                else:
                    self._debug("skipped\n", 3)
                self._debug(' ')
                
        # version 1
        return dict(distances)
        # version 2
        # return {k: dict(v) for k, v in distances.items()}
    
    def _find_data_regions(self, node):
        tag_name = node.attrib[TAG_NAME_ATTRIB]
        node_depth = MDR.depth(node)
        
        self._debug("in _find_data_regions of `{}`".format(tag_name))
        
        # if TreeDepth(Node) => 3 then
        if node_depth >= MDR.MINIMUM_DEPTH:
            
            # Node.DRs = IdenDRs(1, Node, K, T);
            # data_regions = self._identify_data_regions(1, node)  # 0 or 1???
            data_regions = self._identify_data_regions(0, node)
            self.data_regions[tag_name] = data_regions

            self._debug(" ")
            # tempDRs = ∅;
            temp_data_regions = set()
            
            # for each Child ∈ Node.Children do
            for child in node.getchildren():
                
                # FindDRs(Child, K, T);
                self._find_data_regions(child)
                
                # tempDRs = tempDRs ∪ UnCoveredDRs(Node, Child);
                uncovered_data_regions = self._uncovered_data_regions(node, child)
                temp_data_regions = temp_data_regions | uncovered_data_regions
                
            # Node.DRs = Node.DRs ∪ tempDRs
            self.data_regions[tag_name] |= temp_data_regions
        else:
            self._debug("skipped", 1)
            for child in node.getchildren():
                self._find_data_regions(child)
        
        self._debug(" ")
            
    def _identify_data_regions(self, start_index, node):
        """
        Notation: dr = data_region
        """
        tag_name = node.attrib[TAG_NAME_ATTRIB]
        self._debug("in _identify_data_regions node:{}".format(tag_name))
        self._debug("start_index:{}".format(start_index), 1)

        # 1 maxDR = [0, 0, 0];
        max_dr = DataRegion(0, 0, 0)

        # todo remove this
        debug_dist = 1

        # todo check if i need this
        current_dr = None

        # 2 for (i = 1; i <= K; i++) /* compute for each i-combination */
        for gnode_size in range(1, self.max_tag_per_gnode + 1):
            self._debug('gnode_size (i): {}'.format(gnode_size), 2)

            # 3 for (f = start; f <= start+i; f++) /* start from each node */
            for first_gnode_start_index in range(start_index, start_index + gnode_size + 1):
                self._debug('first_gnode_start_index (f): {}'.format(first_gnode_start_index), 3)

                # 4 flag = true;
                is_first_pair = True

                # 5 for (j = f; j < size(Node.Children); j+i)
                # for left_gnode_start in range(start_node, len(node) , gnode_size):
                for last_gnode_start_index in range(
                    first_gnode_start_index + gnode_size, len(node) - gnode_size + 1, gnode_size  # todo: recheck limits
                    # first_gnode_start_index, len(node) - gnode_size + 1, gnode_size  # todo: recheck limits
                    # first_gnode_start_index, len(node) - 2 * gnode_size + 1, gnode_size  # todo: recheck limits
                ):
                    self._debug('last_gnode_start_index (j): {}'.format(last_gnode_start_index), 4)

                    # 6 if Distance(Node, i, j) <= T then

                    # todo: correct here
                    # from _compare_combinations
                    #   left_gnode_indices = (left_gnode_start, right_gnode_start)
                    #   right_gnode_indices = (right_gnode_start, right_gnode_start + gnode_size)
                    left_gnode_indices = (last_gnode_start_index - gnode_size, last_gnode_start_index)
                    right_gnode_indices = (last_gnode_start_index, last_gnode_start_index + gnode_size)

                    self._debug('left_gnode_indices : {}'.format(left_gnode_indices), 5)
                    self._debug('right_gnode_indices : {}'.format(right_gnode_indices), 5)

                    # todo: check if this is necessary
                    gnode_pair = (left_gnode_indices, right_gnode_indices)
                    dist = self.distances[tag_name][gnode_size][gnode_pair]

                    # todo remove this
                    self._checked_data_regions[tag_name].add(gnode_pair)
                    
                    # todo REMOVE THIS >>>> DEBUG ONLY
                    self._debug('dist (temp before debug_dist) : {}'.format(dist), 5)
                    dist = debug_dist
                    debug_dist *= 0.98
                    self._debug('dist : {}'.format(dist), 4)

                    if dist <= self.edit_distance_threshold:
                        self._debug('>>> in dist if <<<', 6)

                        # 7 if flag=true then
                        if is_first_pair:
                            self._debug('>>> is first pair <<<', 7)

                            # 8 curDR = [i, j, 2*i];
                            current_dr = DataRegion(gnode_size, first_gnode_start_index, 2 * gnode_size)

                            self._debug('current_dr : {}'.format(current_dr), 7)

                            # 9 flag = false;
                            is_first_pair = False

                        # 10 else curDR[3] = curDR[3] + i;
                        else:
                            self._debug('>>> is NOT first pair <<<', 7)
                            current_dr = DataRegion(
                                current_dr[0], current_dr[1], current_dr[2] + gnode_size
                            ) 
                            self._debug('current_dr : {}'.format(current_dr), 6)

                    # 11 elseif flag = false then Exit-inner-loop;
                    elif not is_first_pair:
                        self._debug('>>> is NOT first pair and failed distance TH <<<', 5)
                        break

                    self._debug(" ")

                # 13 if (maxDR[3] < curDR[3]) and (maxDR[2] = 0 or (curDR[2]<= maxDR[2]) then
                if (current_dr is not None) and (max_dr[2] < current_dr[2]) and (max_dr[1] == 0 or current_dr[1] <= max_dr[1]):
                    self._debug('!!! replacing max dr !!!', 4)
                    self._debug(" ")

                    # 14 maxDR = curDR;
                    max_dr = current_dr
                    self._debug('max_dr : {}'.format(max_dr), 4)

        # 16 if ( maxDR[3] != 0 ) then
        if max_dr.n_nodes_covered > 0:

            # 17 if (maxDR[2]+maxDR[3]-1 != size(Node.Children)) then
            last_covered_tag_index = max_dr.start_child_index + max_dr.n_nodes_covered
            if last_covered_tag_index < len(node):  # todo check this condition !!!!

                # 18 return {maxDR} ∪ IdentDRs(maxDR[2]+maxDR[3], Node, K, T)
                return {max_dr} | self._identify_data_regions(last_covered_tag_index, node)

            # 19 else return {maxDR}
            else:
                return {max_dr}

        # 21 return ∅;
        return set()

    def _uncovered_data_regions(self, node, child):
        return set()
