"""
Module dependencies:
    all - {utils} -> core


Notation:
    gn = gnode = generalized node
    dr = data region

References:
    [1] Liu, Bing & Grossman, Robert & Zhai, Yanhong. (2003). Mining data records in Web pages.
        Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
        601-606. 10.1145/956750.956826.
"""

import copy
import logging
from collections import defaultdict, namedtuple, UserList
from typing import Set, List, Dict, Union, Optional

import Levenshtein
import lxml
import lxml.etree
import lxml.html

from utils import generate_random_colors

NODE_NAME_ATTRIB = "___tag_name___"

logging.basicConfig(
    level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s"
)


class WithBasicFormat(object):
    """Define a basic __format__ with !s, !r and ''."""

    def _extra_format(self, format_spec: str) -> str:
        raise NotImplementedError()

    def __format__(self, format_spec: str) -> str:
        if format_spec == "!s":
            return str(self)
        elif format_spec in ("!r", ""):
            return repr(self)
        else:
            try:
                return self._extra_format(format_spec)
            except NotImplementedError:
                raise TypeError(
                    "unsupported format string passed to {}.__format__".format(type(self).__name__)
                )


class GNode(namedtuple("GNode", ["parent", "start", "end"]), WithBasicFormat):
    """Generalized Node - start/end are indexes of sibling nodes relative to their parent node."""

    def __str__(self):
        return "GN({start:>2}, {end:>2})".format(start=self.start, end=self.end)

    def __len__(self):
        return self.size

    def _extra_format(self, format_spec):
        if format_spec == "!S":
            return "GN({parent}, {start:>2}, {end:>2})".format(
                parent=self.parent, start=self.start, end=self.end
            )
        else:
            raise NotImplementedError()

    @property
    def size(self):
        return self.end - self.start


# noinspection PyAbstractClass
class GNodePair(namedtuple("GNodePair", ["left", "right"]), WithBasicFormat):
    """Generalized Node Pair - pair of adjacent GNodes, used for stocking the edit distances between them."""

    def __str__(self):
        return "{left:!s} - {right:!s}".format(left=self.left, right=self.right)


# noinspection PyArgumentList
class DataRegion(
    namedtuple(
        "DataRegion", ["parent", "gnode_size", "first_gnode_start_index", "n_nodes_covered"],
    ),
    WithBasicFormat,
):
    """Data Region - a continuous sequence of GNode's."""

    def _extra_format(self, format_spec):
        if format_spec == "!S":
            return "DR({0}, {1}, {2}, {3})".format(
                self.parent, self.gnode_size, self.first_gnode_start_index, self.n_nodes_covered,
            )
        else:
            raise NotImplementedError()

    def __str__(self):
        return "DR({0}, {1}, {2})".format(
            self.gnode_size, self.first_gnode_start_index, self.n_nodes_covered
        )

    def __contains__(self, child_index: int):
        """todo(doc)"""
        msg = (
            "DataRegion contains the indexes of a node relative to its parent list of children. "
            "Type `{}` not supported.".format(type(child_index).__name__)
        )
        assert isinstance(child_index, int), msg
        return self.first_gnode_start_index <= child_index <= self.last_covered_tag_index

    def get_gnode_iterator(self):
        class GNodeIterator:
            def __init__(self, dr):
                self.dr = dr

            def __iter__(self):
                self._iter_i = 0
                return self

            def __next__(self):
                if self._iter_i < self.dr.n_gnodes:
                    start = self.dr.first_gnode_start_index + self._iter_i * self.dr.gnode_size
                    end = start + self.dr.gnode_size
                    gnode = GNode(self.dr.parent, start, end)
                    self._iter_i += 1
                    return gnode
                else:
                    raise StopIteration

        return GNodeIterator(self)

    @classmethod
    def empty(cls):
        return cls(None, None, None, 0)

    @classmethod
    def binary_from_last_gnode(cls, gnode: GNode):
        """(Joao: I know the name is confusing...) It is the DR of 2 GNodes where the last one is `gnode`."""
        gnode_size = gnode.end - gnode.start
        return cls(gnode.parent, gnode_size, gnode.start - gnode_size, 2 * gnode_size)

    @property
    def is_empty(self):
        return self[0] is None

    @property
    def n_gnodes(self):
        return self.n_nodes_covered // self.gnode_size

    @property
    def last_covered_tag_index(self):
        return self.first_gnode_start_index + self.n_nodes_covered - 1

    def extend_one_gnode(self):
        return self.__class__(
            self.parent,
            self.gnode_size,
            self.first_gnode_start_index,
            self.n_nodes_covered + self.gnode_size,
        )


# noinspection PyAbstractClass
class DataRecord(UserList, WithBasicFormat):
    @property
    def is_non_contiguous(self) -> bool:
        return len(self) > 1

    def __hash__(self):
        return hash(tuple(self))

    def __repr__(self):
        return "DataRecord({})".format(", ".join([repr(gn) for gn in self.data]))

    def __str__(self):
        return "DataRecord({})".format(", ".join([str(gn) for gn in self.data]))


class MDREditDistanceThresholds(
    namedtuple("MDREditDistanceThresholds", ["data_region", "find_records_1", "find_records_n"],)
):
    @classmethod
    def all_equal(cls, threshold):
        return cls(threshold, threshold, threshold)


class UsedMDRException(Exception):
    default_message = "This MDR instance has already been used. Please instantiate another one."

    def __init__(self):
        super(Exception, self).__init__(self.default_message)


class NodeNamer(object):
    """todo(doc)
    # todo(improvement) change the other naming method to use this
    todo(unittest)"""

    def __init__(self, for_loaded_file: bool = False):
        self.tag_counts = defaultdict(int)
        self._is_loaded = for_loaded_file

    def __call__(self, node: lxml.html.HtmlElement, *args, **kwargs):
        assert self._is_loaded, "Must load the node namer first!!!"
        assert NODE_NAME_ATTRIB in node.attrib, "The given node has not been seen during load."
        return node.attrib[NODE_NAME_ATTRIB]

    @staticmethod
    def cleanup_all(root: lxml.html.HtmlElement) -> None:
        for node in root.getiterator():
            del node.attrib[NODE_NAME_ATTRIB]

    def load(self, root: lxml.html.HtmlElement) -> None:
        if self._is_loaded:
            return
        # each tag is named sequentially
        for node in root.getiterator():
            tag = node.tag
            tag_sequential = self.tag_counts[tag]
            self.tag_counts[tag] += 1
            node_name = "{0}-{1:0>5}".format(tag, tag_sequential)
            node.set(NODE_NAME_ATTRIB, node_name)
            self._is_loaded = True


# typing consts
NODE_DISTANCES_DICT_FORMAT = Dict[int, Dict[GNodePair, float]]
DISTANCES_DICT_FORMAT = Dict[str, Union[int, Optional[NODE_DISTANCES_DICT_FORMAT]]]
DATA_REGION_DICT_FORMAT = Dict[str, Union[int, float, Set[DataRegion]]]
DATA_RECORDS = Set[DataRecord]


# noinspection PyArgumentList
def get_data_records_as_nodes(
    doc: lxml.html.HtmlElement, data_records: DATA_RECORDS
) -> List[List[lxml.html.HtmlElement]]:
    """
    Returns:
        List[DataRecord]  ==
        List[List[HtmlElement]]  ==
    """
    # List[]
    return [
        _get_node(doc, gn.parent).getchildren()[gn.start : gn.end]
        for data_record in data_records
        for gn in data_record
    ]


class MDR:

    data_records: DATA_RECORDS

    def __init__(
        self,
        root,
        minimum_depth=3,
        max_tag_per_gnode: int = 10,
        edit_distance_threshold: MDREditDistanceThresholds = MDREditDistanceThresholds.all_equal(
            0.3
        ),
        precomputed_distances: DISTANCES_DICT_FORMAT = None,
    ):
        """ The default values are from [1]."""
        self.root_original = root
        self.root = copy.deepcopy(root)
        self.minimum_depth = minimum_depth
        self.max_tag_per_gnode = max_tag_per_gnode
        self.edit_distance_threshold = edit_distance_threshold
        self.precomputed_distances = precomputed_distances or {}

        self.distances: Dict[str, Union[int, Optional[Dict[int, Dict[GNodePair, float]]]]] = {}
        self.data_regions = {}
        self.node_namer = NodeNamer()
        self.node_namer.load(self.root)

        # {node_name(str): set(GNode)}  only retains the max data regions
        self._used = False

    @classmethod
    def with_defaults(cls, root, precomputed_distances: DISTANCES_DICT_FORMAT = None):
        return cls(root, precomputed_distances=precomputed_distances)

    def __call__(self) -> DATA_RECORDS:  # todo remove none
        if self._used:
            raise UsedMDRException()
        self._used = True

        logging.info("STARTING COMPUTE DISTANCES PHASE")
        compute_distances(
            self.root,
            self.distances,
            self.precomputed_distances,
            self.node_namer,
            self.minimum_depth,
            self.max_tag_per_gnode,
        )

        logging.info("STARTING FIND DATA REGIONS PHASE")
        find_data_regions(
            self.root,
            self.node_namer,
            self.minimum_depth,
            self.distances,
            self.data_regions,
            self.edit_distance_threshold.data_region,
            self.max_tag_per_gnode,
        )

        logging.info("STARTING FIND DATA RECORDS PHASE")
        self.data_records = find_data_records(
            self.root,
            self.data_regions,
            self.distances,
            self.node_namer,
            self.edit_distance_threshold,
            self.max_tag_per_gnode,
        )

        return self.data_records


def compute_distances(
    node,
    distances: DISTANCES_DICT_FORMAT,
    precomputed: DISTANCES_DICT_FORMAT,
    node_namer: NodeNamer,
    minimum_depth: int,
    max_tag_per_gnode: int,
) -> None:
    # todo create dry run to get the size of list/dicts and then rerun --> faster by avoiding allocation
    node_name = node_namer(node)
    node_depth = depth(node)
    logging.debug("node_name=%s depth=%d)", node_name, node_depth)

    if node_depth >= minimum_depth and should_process_node(node):
        # get all possible node_distances of the n-grams of children
        # {gnode_size: {GNode: float}}
        # todo put these strings in consts
        precomputed_min_depth = precomputed.get("minimum_depth")
        precomputed_max_tag_per_gnode = precomputed.get("max_tag_per_gnode")
        # todo(improvement) use as much as possible if it's partially computed...
        precomputed_is_compatible = (
            precomputed_min_depth is not None
            and precomputed_max_tag_per_gnode is not None
            and precomputed_min_depth <= minimum_depth
            and precomputed_max_tag_per_gnode >= max_tag_per_gnode
        )
        precomputed_node_distances = (
            precomputed.get(node_name) if precomputed_is_compatible else None
        )

        if precomputed_node_distances is None:
            node_distances = _compare_combinations(node.getchildren(), node_name, max_tag_per_gnode)
        else:
            node_distances = precomputed_node_distances
    else:
        logging.debug("skipped (less than min depth = %d)", minimum_depth)
        node_distances = None

    distances[node_name] = node_distances

    for child in node:
        compute_distances(
            child, distances, precomputed, node_namer, minimum_depth, max_tag_per_gnode
        )


def _compare_combinations(
    node_list: List[lxml.html.HtmlElement], parent_name, max_tag_per_gnode, only_1b1=False
) -> NODE_DISTANCES_DICT_FORMAT:
    """
    Notation: gnode = "generalized node"
    :returns
        {gnode_size: {GNode: float}}
    """

    if not node_list:
        logging.debug("empty list --> return {}")
        return {}

    # {gnode_size: {GNode: float}}
    distances = defaultdict(dict)
    n_nodes = len(node_list)
    logging.debug("n_nodes: %d", n_nodes)

    # 1) for (i = 1; i <= K; i++)  /* start from each node */
    for starting_tag in range(1, max_tag_per_gnode + 1):
        # 2) for (j = i; j <= K; j++) /* comparing different combinations */
        gnode_size_range = range(starting_tag, max_tag_per_gnode + 1) if not only_1b1 else [1]
        for gnode_size in gnode_size_range:  # j
            # 3) if NodeList[i+2*j-1] exists then
            there_are_pairs_to_look = (starting_tag + 2 * gnode_size - 1) < n_nodes + 1
            if there_are_pairs_to_look:  # +1 for pythons open set notation
                logging.debug(
                    "starting_tag(i)=%d | gnode_size(j)=%d | if(there_are_pairs_to_look) == True",
                    starting_tag,
                    gnode_size,
                )

                # 4) St = i;
                left_gnode_start = starting_tag - 1  # st

                # 5) for (k = i+j; k < Size(NodeList); k+j)
                for right_gnode_start in range(
                    starting_tag + gnode_size - 1, n_nodes, gnode_size
                ):  # k
                    # 6)  if NodeList[k+j-1] exists then
                    right_gnode_exists = right_gnode_start + gnode_size < n_nodes + 1

                    if right_gnode_exists:
                        logging.debug(
                            "starting_tag(i)=%d | gnode_size(j)=%d | "
                            "left_gnode_start(st)=%d | right_gnode_start(k)=%d | "
                            "if(right_gnode_exists) == True",
                            starting_tag,
                            gnode_size,
                            left_gnode_start,
                            right_gnode_start,
                        )

                        # todo(improvement): avoid recomputing strings?
                        # todo(improvement): avoid recomputing edit distances?
                        # todo(improvement): check https://pypi.org/project/strsim/ ?

                        # NodeList[St..(k-1)]
                        left_gnode = GNode(parent_name, left_gnode_start, right_gnode_start,)
                        left_gnode_nodes = node_list[left_gnode.start : left_gnode.end]
                        left_gnode_str = nodes_to_string(left_gnode_nodes)

                        # NodeList[St..(k-1)]
                        right_gnode = GNode(
                            parent_name, right_gnode_start, right_gnode_start + gnode_size,
                        )
                        right_gnode_nodes = node_list[right_gnode.start : right_gnode.end]
                        right_gnode_str = nodes_to_string(right_gnode_nodes)

                        # 7) EditDist(NodeList[St..(k-1), NodeList[k..(k+j-1)])
                        edit_distance = Levenshtein.ratio(left_gnode_str, right_gnode_str)
                        gnode_pair = GNodePair(left_gnode, right_gnode)

                        logging.debug(
                            "starting_tag(i)=%d | gnode_size(j)=%d | "
                            "left_gnode_start(st)=%d | right_gnode_start(k)=%d | "
                            "dist(%s) = %.2f",
                            starting_tag,
                            gnode_size,
                            left_gnode_start,
                            right_gnode_start,
                            gnode_pair,
                            edit_distance,
                        )

                        # {gnode_size: {GNode: float}}
                        distances[gnode_size][gnode_pair] = edit_distance

                        # 8) St = k+j
                        left_gnode_start = right_gnode_start
                    else:
                        logging.debug(
                            "starting_tag(i)=%d | gnode_size(j)=%d | "
                            "left_gnode_start(st)=%d | right_gnode_start(k)=%d | "
                            "if(right_gnode_exists) == False --> skipped",
                            starting_tag,
                            gnode_size,
                            left_gnode_start,
                            right_gnode_start,
                        )
            else:
                logging.debug(
                    "starting_tag(i)=%d | gnode_size(j)=%d | "
                    "if(there_are_pairs_to_look) == False --> skipped",
                    starting_tag,
                    gnode_size,
                )

    return dict(distances)


def find_data_regions(
    node: lxml.html.HtmlElement,
    node_namer: NodeNamer,
    minimum_depth: int,
    distances: DISTANCES_DICT_FORMAT,
    all_data_regions: DATA_REGION_DICT_FORMAT,
    distance_threshold: float,
    max_tag_per_gnode: int,
) -> None:
    node_depth = depth(node)

    # 1) if TreeDepth(Node) => 3 then
    if node_depth >= minimum_depth and should_process_node(node):
        # todo(log) add here

        # 2) Node.DRs = IdenDRs(1, Node, K, T);
        node_name = node_namer(node)
        n_children = len(node)
        node_distances = distances.get(node_name)
        data_regions = _identify_data_regions(
            start_index=0,
            node_name=node_name,
            n_children=n_children,
            node_distances=node_distances,
            distance_threshold=distance_threshold,
            max_tag_per_gnode=max_tag_per_gnode,
        )
        all_data_regions[node_name] = data_regions

        # 3) tempDRs = ∅;
        temp_data_regions = set()

        # 4) for each Child ∈ Node.Children do
        for child_idx, child in enumerate(node.getchildren()):

            child_name = node_namer(child)

            # 5) FindDRs(Child, K, T);
            find_data_regions(
                child,
                node_namer,
                minimum_depth,
                distances,
                all_data_regions,
                distance_threshold,
                max_tag_per_gnode,
            )

            # 6) tempDRs = tempDRs ∪ UnCoveredDRs(Node, Child);
            uncovered_data_regions = (
                all_data_regions[child_name]
                if child_name in all_data_regions
                and _uncovered_data_regions(all_data_regions[node_name], child_idx)
                else set()
            )
            temp_data_regions = temp_data_regions | uncovered_data_regions

        # 7) Node.DRs = Node.DRs ∪ tempDRs
        all_data_regions[node_name] |= temp_data_regions

    else:
        # todo(log) add here
        for child in node.getchildren():
            find_data_regions(
                child,
                node_namer,
                minimum_depth,
                distances,
                all_data_regions,
                distance_threshold,
                max_tag_per_gnode,
            )


def _identify_data_regions(
    start_index: int,
    node_name: str,
    n_children: int,
    node_distances: NODE_DISTANCES_DICT_FORMAT,
    distance_threshold: float,
    max_tag_per_gnode: int,
) -> Set[DataRegion]:

    # todo(log) add here
    # logging.debug("in _identify_data_regions node: {}".format(node_name))
    # logging.debug("start_index:{}".format(start_index), 1)

    if not node_distances:
        # todo(log) add here
        # logging.debug("no distances, returning empty set")
        return set()

    # 1 maxDR = [0, 0, 0];
    max_dr = DataRegion.empty()
    current_dr = DataRegion.empty()

    # 2 for (i = 1; i <= K; i++) /* compute for each i-combination */
    for gnode_size in range(1, max_tag_per_gnode + 1):
        # todo(log) add here
        # logging.debug("gnode_size (i): {}".format(gnode_size), 2)

        # 3 for (f = start; f <= start+i; f++) /* start from each node */
        # for start_gnode_start_index in range(start_index, start_index + gnode_size + 1):
        for first_gn_start_idx in range(start_index, start_index + gnode_size):
            # todo(log) add here
            # logging.debug("first_gn_start_idx (f): {}".format(first_gn_start_idx), 3)

            # 4 flag = true;
            dr_has_started = False

            # 5 for (j = f; j < size(Node.Children); j+i)
            # for left_gnode_start in range(start_node, len(node) , gnode_size):
            for last_gn_start_idx in range(
                # start_gnode_start_index, len(node) - gnode_size + 1, gnode_size
                first_gn_start_idx + gnode_size,
                n_children - gnode_size + 1,
                gnode_size,
            ):
                # todo(log) add here
                # logging.debug(
                #     "last_gn_start_idx (j): {}".format(last_gn_start_idx), 4,
                # )

                # 6 if Distance(Node, i, j) <= T then
                gn_last = GNode(node_name, last_gn_start_idx, last_gn_start_idx + gnode_size,)
                gn_before_last = GNode(
                    node_name, last_gn_start_idx - gnode_size, last_gn_start_idx,
                )
                gn_pair = GNodePair(gn_before_last, gn_last)
                distance = node_distances[gnode_size][gn_pair]

                # todo(log) add here
                # logging.debug(
                #     "gn_pair (bef last, last): {!s} = {:.2f}".format(gn_pair, distance), 5,
                # )

                if distance <= distance_threshold:
                    # todo(log) add here
                    # logging.debug("dist passes the threshold!".format(distance), 6)

                    # 7 if flag=true then
                    if not dr_has_started:

                        # todo(log) add here
                        # logging.debug(
                        #     "it is the first pair, init the `current_dr`...".format(distance),
                        #     6,
                        # )

                        # 8 curDR = [i, j, 2*i];
                        # current_dr = DataRegion(gnode_size, first_gn_start_idx - gnode_size, 2 * gnode_size)
                        # current_dr = DataRegion(gnode_size, first_gn_start_idx, 2 * gnode_size)
                        current_dr = DataRegion.binary_from_last_gnode(gn_last)

                        # todo(log) add here
                        # logging.debug("current_dr: {}".format(current_dr), 6)

                        # 9 flag = false;
                        dr_has_started = True

                    # 10 else curDR[3] = curDR[3] + i;
                    else:
                        # todo(log) add here
                        # logging.debug("extending the DR...".format(distance), 6)
                        # current_dr = DataRegion(
                        #     current_dr[0], current_dr[1], current_dr[2] + gnode_size
                        # )
                        current_dr = current_dr.extend_one_gnode()
                        # todo(log) add here
                        # logging.debug("current_dr: {}".format(current_dr), 6)

                # 11 elseif flag = false then Exit-inner-loop;
                elif dr_has_started:
                    # todo(log) add here
                    # logging.debug("above the threshold, breaking the loop...", 6)
                    break

            # 13 if (maxDR[3] < curDR[3]) and (maxDR[2] = 0 or (curDR[2]<= maxDR[2]) then
            # todo(improvement) add a criteria that checks the avg distance when
            #  n_nodes_covered is the same and it starts at the same node
            current_is_strictly_larger = max_dr.n_nodes_covered < current_dr.n_nodes_covered
            current_starts_at_same_node_or_before = (
                max_dr.is_empty
                or current_dr.first_gnode_start_index <= max_dr.first_gnode_start_index
            )

            if current_is_strictly_larger and current_starts_at_same_node_or_before:
                # todo(log) add here
                # logging.debug("current DR is bigger than max! replacing...", 3)

                # 14 maxDR = curDR;
                # todo(log) add here
                # logging.debug(
                #     "old max_dr: {}, new max_dr: {}".format(max_dr, current_dr), 3,
                # )
                max_dr = current_dr

    # todo(log) add here
    # logging.debug("max_dr: {}\n".format(max_dr))

    # 16 if ( maxDR[3] != 0 ) then
    if not max_dr.is_empty:

        # 17 if (maxDR[2]+maxDR[3]-1 != size(Node.Children)) then
        last_covered_idx = max_dr.last_covered_tag_index
        # todo(log) add here
        # logging.debug("max_dr.last_covered_tag_index: {}".format(last_covered_idx))

        if last_covered_idx < n_children - 1:
            # todo(log) add here
            # logging.debug("calling recursion! \n")

            # 18 return {maxDR} ∪ IdentDRs(maxDR[2]+maxDR[3], Node, K, T)
            return {max_dr} | _identify_data_regions(
                start_index=last_covered_idx + 1,
                node_name=node_name,
                n_children=n_children,
                node_distances=node_distances,
                distance_threshold=distance_threshold,
                max_tag_per_gnode=max_tag_per_gnode,
            )

        # 19 else return {maxDR}
        else:
            # todo(log) add here
            # logging.debug("returning {{max_dr}}")
            return {max_dr}

    # 21 return ∅;
    # todo(log) add here
    # logging.debug("max_dr is empty, returning empty set")
    return set()


def _uncovered_data_regions(node_drs: Set[DataRegion], child_idx: int) -> bool:
    # 1) for each data region DR in Node.DRs do
    for dr in node_drs:
        # 2) if Child in range DR[2] .. (DR[2] + DR[3] - 1) then
        if child_idx in dr:
            # todo(unittest) test case where child idx is in the limit
            # 3) return null
            return False
    # 4) return Child.DRs
    return True


def find_data_records(
    root: lxml.html.HtmlElement,
    data_regions_per_node: DATA_REGION_DICT_FORMAT,
    distances: DISTANCES_DICT_FORMAT,
    node_namer: NodeNamer,
    edit_distance_threshold: MDREditDistanceThresholds,
    max_tag_per_gnode: int,
) -> DATA_RECORDS:
    all_data_regions: Set[DataRegion] = set.union(
        *(v for v in data_regions_per_node.values() if isinstance(v, set))
    )

    # todo(log) add here and the rest of the  method
    # self._debug("total nb of data regions to check: {}".format(len(all_data_regions)))
    data_records = set()

    for dr in all_data_regions:
        gn_is_of_size_1 = dr.gnode_size == 1
        dr_parent_node = _get_node(root, dr.parent)
        dr_data_records = set()

        gnode: GNode
        for gnode in dr.get_gnode_iterator():
            gnode_nodes = dr_parent_node[gnode.start : gnode.end]
            if gn_is_of_size_1:
                gn_data_records = _find_records_1(
                    gnode,
                    gnode_nodes[0],
                    distances,
                    node_namer,
                    edit_distance_threshold.find_records_1,
                    max_tag_per_gnode,
                )
            else:
                gn_data_records = _find_records_n(
                    gnode,
                    gnode_nodes,
                    distances,
                    node_namer,
                    edit_distance_threshold.find_records_n,
                    max_tag_per_gnode,
                )

            dr_data_records.update(gn_data_records)
            # todo(log) add here

        # check disconnected data records
        # see section '3.4 Data Records not in Data Regions' of [1]
        # todo(test)
        if gn_is_of_size_1 and len(dr_data_records) > 0:
            there_are_nodes_out_of_dr = len(dr_parent_node) > dr.n_nodes_covered
            # if dr and data records have different parents, it's because they are in different levels
            drecs_parents_names = {drec[0].parent for drec in dr_data_records}
            all_data_records_are_in_level_below = all(
                name != dr.parent for name in drecs_parents_names
            )

            if there_are_nodes_out_of_dr and all_data_records_are_in_level_below:
                nodes_with_data_records: List[lxml.html.HtmlElement] = [
                    nd
                    for nd in dr_parent_node.getchildren()
                    if node_namer(nd) in drecs_parents_names
                ]
                a_data_record_node = nodes_with_data_records[0][0]
                a_drec_str = nodes_to_string([a_data_record_node])

                not_covered_nodes: List[lxml.html.HtmlElement] = [
                    dr_parent_node[idx] for idx in range(len(dr_parent_node)) if idx not in dr
                ]
                for nd in not_covered_nodes:
                    candidate_drec_node: lxml.html.HtmlElement
                    for idx, candidate_drec_node in enumerate(nd.getchildren()):
                        dist = Levenshtein.ratio(a_drec_str, nodes_to_string([candidate_drec_node]))
                        if dist <= edit_distance_threshold.find_records_1:
                            new_drec = DataRecord([GNode(node_namer(nd), idx, idx + 1)])
                            dr_data_records.add(new_drec)

        data_records.update(dr_data_records)

    return data_records
    # todo: add the retrieval of data records out of data regions (technical report)


def _find_records_1(
    gnode: GNode,
    gnode_node: lxml.html.HtmlElement,
    distances: DISTANCES_DICT_FORMAT,
    node_namer: NodeNamer,
    edit_distance_threshold: float,  # edit_distance_threshold.find_records_1
    max_tag_per_gnode: int,
) -> DATA_RECORDS:
    """Finding data records in a one-component generalized gnode_node."""
    # todo(log) add here
    # self._debug("in `_find_records_1` ", 2)

    has_children = len(gnode_node) > 1

    node_name = node_namer(gnode_node)

    if has_children:
        node_children_distances = (distances.get(node_name) or {}).get(1, None)
        if node_children_distances is None:
            node_children_distances = _compare_combinations(
                list(gnode_node.getchildren()), node_name, max_tag_per_gnode, only_1b1=True
            ).get(1, {})
    else:
        node_children_distances = None

        # 1) If all children nodes of G are similar
    # it is not well defined what "all .. similar" means - I consider that "similar" means "edit_dist < TH"
    #       hyp 1: it means that every combination 2 by 2 is similar
    #       hyp 2: it means that all the computed edit distances (every sequential pair...) is similar
    # for the sake of practicality and speed, I'll choose the hypothesis 2
    all_children_are_similar = has_children and all(
        d <= edit_distance_threshold for d in node_children_distances.values()
    )

    # 2) AND G is not a data table row then
    node_is_table_row = gnode_node.tag == "tr"

    data_records_found = set()
    if all_children_are_similar and not node_is_table_row:
        # todo(log) add here
        # self._debug("its children are data records", 3)
        # 3) each child node of R is a data record
        for i in range(len(gnode_node)):
            data_records_found.add(DataRecord([GNode(node_name, i, i + 1)]))

    # 4) else G itself is a data record.
    else:
        # todo(log) add here
        # self._debug("it is a data record itself", 3)
        data_records_found.add(DataRecord([gnode]))

    return data_records_found
    # todo(unittest): debug this implementation with examples in the technical paper


def _find_records_n(
    gnode: GNode,
    gnode_nodes: List[lxml.html.HtmlElement],
    distances: DISTANCES_DICT_FORMAT,
    node_namer: NodeNamer,
    distance_threshold: float,  # edit_distance_threshold.find_records_n
    max_tag_per_gnode: int,
) -> DATA_RECORDS:
    """Finding data records in an n-component generalized node."""
    # todo(log) add here
    # self._debug("in `_find_records_n` ", 2)

    numbers_children = [len(n) for n in gnode_nodes]
    childrens_distances = []
    for nd in gnode_nodes:
        nd_name = node_namer(nd)
        nd_dists = (distances.get(nd_name) or {}).get(1, None)
        if (nd_dists is None or len(nd_dists) == 0) and len(nd) > 1:
            nd_dists = _compare_combinations(
                list(nd.getchildren()), nd_name, max_tag_per_gnode, only_1b1=True
            ).get(1, {})
        childrens_distances.append(nd_dists)
    # childrens_distances = [distances[node_namer(n)].get(1, None) for n in gnode_nodes]

    all_have_same_nb_children = len(set(numbers_children)) == 1
    childrens_are_similar = None not in childrens_distances and all(
        child_distances and all(d <= distance_threshold for d in child_distances.values())
        for child_distances in childrens_distances
    )

    # 1) If the children gnode_nodes of each node in G are similar
    # 1...)   AND each node also has the same number of children then
    data_records_found = set()
    if not (all_have_same_nb_children and childrens_are_similar):
        # todo(log) add here

        # 3) else G itself is a data record.
        data_records_found.add(DataRecord([gnode]))

    else:
        # todo(log) add here
        # 2) The corresponding children gnode_nodes of every node in G form a non-contiguous object description
        n_children = numbers_children[0]
        for i in range(n_children):
            data_records_found.add(
                DataRecord([GNode(node_namer(n), i, i + 1) for n in gnode_nodes])
            )
        # todo(unittest) check a case like this

    return data_records_found
    # todo(unittest): debug this implementation


def _get_node(root: lxml.html.HtmlElement, node_name: str) -> lxml.html.HtmlElement:
    tag = node_name.split("-")[0]
    # todo add some security stuff here???
    # this depends on the implementation of `NodeNamer`
    nodes = root.xpath(
        "//{tag}[@___tag_name___='{node_name}']".format(tag=tag, node_name=node_name)
    )
    if len(nodes) == 0:
        raise Exception("node not found node_name={}".format(node_name))
    return nodes[0]


def nodes_to_string(list_of_nodes: List[lxml.html.HtmlElement]) -> str:
    return " ".join([lxml.etree.tostring(child).decode("utf-8").strip() for child in list_of_nodes])


def depth(node):
    d = 0
    while node is not None:
        d += 1
        node = node.getparent()
    return d - 1


def should_process_node(node: lxml.html.HtmlElement):
    return node.tag in (
        "table",
        "tr",
        "th",
        "td",
        "thead",
        "tbody",
        "tfoot",
        "form",
        # "ol",
        # "ul",
        # "li",
    )


def is_processable(node: lxml.html.HtmlElement):
    return "___should_process___" in node.attrib


def paint_data_records(data_records_nodes: List[List[lxml.html.HtmlElement]]):
    """todo(unittest)"""
    colors = generate_random_colors(len(data_records_nodes))
    for record_nodes, color in zip(data_records_nodes, colors):
        for e in record_nodes:
            e.set("style", "background-color: #{} !important;".format(color))
