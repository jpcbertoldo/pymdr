"""

Module dependencies:
    all - {utils} -> core

Notation:
    gn = gnode = generalized node
    dr = data region
    drec = data record

References:
    [1] Liu, Bing & Grossman, Robert & Zhai, Yanhong. (2003). Mining data records in Web pages.
        Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
        601-606. 10.1145/956750.956826.
    Note: refer to the technical report version.
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


STR_DIST_USE_NODE_NAME_CLEANUP = True

# these are used for finding parameters of previous runs for preloaded (intermediate) results
DICT_PARAM_TAG_PER_GNODE = "max_tag_per_gnode"
DICT_PARAM_MINIMUM_DEPTH = "minimum_depth"

# for typing
HTML_ELEMENT = lxml.html.HtmlElement

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

    def __str__(self) -> str:
        return "GN({start:>2}, {end:>2})".format(start=self.start, end=self.end)

    def __len__(self) -> int:
        """ Number of nodes in the generalized node. """
        return self.size

    def _extra_format(self, format_spec: str):
        if format_spec == "!S":
            return "GN({parent}, {start:>2}, {end:>2})".format(
                parent=self.parent, start=self.start, end=self.end
            )
        else:
            raise NotImplementedError()

    @property
    def size(self) -> int:
        """ Number of nodes in the generalized node. """
        return self.end - self.start


# noinspection PyAbstractClass
class GNodePair(namedtuple("GNodePair", ["left", "right"]), WithBasicFormat):
    """Generalized Node Pair - pair of adjacent GNodes, used for stocking the edit distances between them."""

    def __str__(self) -> str:
        return "{left:!s} - {right:!s}".format(left=self.left, right=self.right)


# noinspection PyArgumentList
class DataRegion(
    namedtuple(
        "DataRegion", ["parent", "gnode_size", "first_gnode_start_index", "n_nodes_covered"],
    ),
    WithBasicFormat,
):
    """Data Region - a continuous sequence of GNode's."""

    def _extra_format(self, format_spec: str):
        if format_spec == "!S":
            return "DR({0}, {1}, {2}, {3})".format(
                self.parent, self.gnode_size, self.first_gnode_start_index, self.n_nodes_covered,
            )
        else:
            raise NotImplementedError()

    def __str__(self) -> str:
        return "DR({0}, {1}, {2})".format(
            self.gnode_size, self.first_gnode_start_index, self.n_nodes_covered
        )

    def __contains__(self, child_index: int) -> bool:
        """ True if the index (supposedly of a child) is inside the interval of children covered by the DR."""
        msg = (
            "DataRegion contains the indexes of a node relative to its parent list of children. "
            "Type `{}` not supported.".format(type(child_index).__name__)
        )
        assert isinstance(child_index, int), msg
        return self.first_gnode_start_index <= child_index <= self.last_covered_tag_index

    def get_gnode_iterator(self):
        """ Return an iterator that yields the GNode objects under the data region. """

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
        """ Initializer that represents an invalid/empty data region. """
        return cls(None, None, None, 0)

    @classmethod
    def binary_from_last_gnode(cls, gnode: GNode):
        """
            (Joao: I know the name is confusing...) It is the DR of 2 GNodes where the last one is `gnode`.
            It's just an utility for the distance computing routine.
        """
        gnode_size = gnode.end - gnode.start
        return cls(gnode.parent, gnode_size, gnode.start - gnode_size, 2 * gnode_size)

    @property
    def is_empty(self) -> bool:
        return self[0] is None

    @property
    def n_gnodes(self) -> int:
        return self.n_nodes_covered // self.gnode_size

    @property
    def last_covered_tag_index(self) -> int:
        """ Last index of the parent node whose node is inside the data region. """
        return self.first_gnode_start_index + self.n_nodes_covered - 1

    def extend_one_gnode(self) -> "DataRegion":
        """ Return a data region that has the same characteristics but one GNode bigger. """
        return self.__class__(
            self.parent,
            self.gnode_size,
            self.first_gnode_start_index,
            self.n_nodes_covered + self.gnode_size,
        )


# noinspection PyAbstractClass
class DataRecord(UserList, WithBasicFormat):
    """
        A data record is a list of GNodes.
        Most of the data records have a single data region and, therefore, are 'equivalent'.
        It is necessary to consider have a list to cover the cases where a data record has disconnected fields.
        This notion is detected by the term of 'contiguity'.
    """

    @property
    def is_non_contiguous(self) -> bool:
        """ It is non contiguous if it has disconnected data regions (see the class' doc). """
        return len(self) > 1

    def __hash__(self) -> int:
        """ Necessary for dedupling. """
        return hash(tuple(self))

    def __repr__(self) -> str:
        return "DataRecord({})".format(", ".join([repr(gn) for gn in self.data]))

    def __str__(self) -> str:
        return "DataRecord({})".format(", ".join([str(gn) for gn in self.data]))


class MDREditDistanceThresholds(
    namedtuple("MDREditDistanceThresholds", ["data_region", "find_records_1", "find_records_n"],)
):
    """
        Utility class for hold the three (possibly) different distance thresholds.
    """

    @classmethod
    def all_equal(cls, threshold):
        return cls(threshold, threshold, threshold)


class UsedMDRException(Exception):
    default_message = "This MDR instance has already been used. Please instantiate another one."

    def __init__(self):
        super(Exception, self).__init__(self.default_message)


class NodeNamer(object):
    """
        This class is an utility for finding the node name of a given node's HtmlElement.
        On init it will right the nodes' names sequentially on themselves as an attribute.
        Then, when called again with an HtmlNode, it retrieves this attribute and returns it.
        # todo(improvement)(?) change the other naming method to use this
        # improvement
    """

    def __init__(self, for_loaded_file: bool = False):
        self.tag_counts = defaultdict(int)
        self._is_loaded = for_loaded_file

    def __call__(self, node: HTML_ELEMENT, *args, **kwargs):
        assert self._is_loaded, "Must load the node namer first!!!"
        assert NODE_NAME_ATTRIB in node.attrib, "The given node has not been seen during load."
        return node.attrib[NODE_NAME_ATTRIB]

    @staticmethod
    def cleanup_all(root: HTML_ELEMENT) -> None:
        """ Remove the name attributes from all the nodes of an html tree. """
        for node in root.getiterator():
            if NODE_NAME_ATTRIB in node.attrib:
                del node.attrib[NODE_NAME_ATTRIB]

    def load(self, root: HTML_ELEMENT) -> None:
        """ Write down the name attribute in the nodes of an html tree. """
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

# dict that stocks all the distances pairs of a given node - the first level (int) is the size of the gnodes
NODE_DISTANCES_DICT_FORMAT = Dict[int, Dict[GNodePair, float]]

# keeps all the NODE_DISTANCES_DICT_FORMAT and metadata about the specs of
# how they were computed (min depth, max nodes per gnode)
DISTANCES_DICT_FORMAT = Dict[str, Union[int, Optional[NODE_DISTANCES_DICT_FORMAT]]]

# keeps all the sets of data regions of each node and metadata about the specs of
# how they were computed (min depth, max nodes per gnode)
DATA_REGION_DICT_FORMAT = Dict[str, Union[int, float, Set[DataRegion]]]

DATA_RECORDS = Set[DataRecord]


# noinspection PyArgumentList
def get_data_records_as_nodes(
    doc: HTML_ELEMENT, data_records: DATA_RECORDS
) -> List[List[HTML_ELEMENT]]:
    """
    Returns:
        List[DataRecord]  ==
        List[List[HtmlElement]]  ==
    """
    return [
        _get_node(doc, gn.parent).getchildren()[gn.start : gn.end]
        for data_record in data_records
        for gn in data_record
    ]


class MDR:
    """
    An MDR object will put together the three main parts of the algorithm and stock all
     the intermediate entities inside so that it can be inspected. An instance represents
     an execution, so it cannot be called twice.
    """

    # the main output of the algorithm
    data_records: DATA_RECORDS

    def __init__(
        self,
        root: HTML_ELEMENT,
        minimum_depth: int = 3,
        max_tag_per_gnode: int = 10,
        edit_distance_threshold: MDREditDistanceThresholds = MDREditDistanceThresholds.all_equal(
            0.3
        ),
        precomputed_distances: DISTANCES_DICT_FORMAT = None,
    ):
        """
        The default values are from [1].

        Args:
            root: root of the html tree
            minimum_depth: bellow this depth (i.g. d < min), the distances are not computed
                            and it is ignored when looking for data regions
            max_tag_per_gnode: consider gnodes of size up to this
            edit_distance_threshold: this defines what "close" is in terms of str distance for html nodes
            precomputed_distances: cache mechanism for the distances because it is the longest part of the algorithm
        """
        self.root_original = root
        self.root = copy.deepcopy(root)
        self.minimum_depth = minimum_depth
        self.max_tag_per_gnode = max_tag_per_gnode
        self.edit_distance_threshold = edit_distance_threshold
        self.precomputed_distances = precomputed_distances or {}

        self.distances: DISTANCES_DICT_FORMAT = {}
        self.data_regions: DATA_REGION_DICT_FORMAT = {}
        self.node_namer: NodeNamer = NodeNamer()
        self.node_namer.load(self.root)

        self._used = False

    @classmethod
    def with_defaults(cls, root: HTML_ELEMENT, precomputed_distances: DISTANCES_DICT_FORMAT = None):
        """ Shortcut for using the default parameters. """
        return cls(root, precomputed_distances=precomputed_distances)

    def __call__(self) -> DATA_RECORDS:
        """ Launches the algorithm execution. """

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
    """
        See pseudo code in Figure 5 in [1].
        It fills in the given `distances` dict reusing the `precomputed` to accelerate if possible.
        todo(improvement) create dry run to get the size of list/dicts and then rerun --> faster by avoiding allocation
    """

    node_name = node_namer(node)
    node_depth = depth(node)
    logging.debug("node_name=%s depth=%d)", node_name, node_depth)

    if node_depth >= minimum_depth and should_process_node(node):
        # get all possible node_distances of the n-grams of children
        # {gnode_size: {GNode: float}}
        precomputed_min_depth = precomputed.get(DICT_PARAM_MINIMUM_DEPTH)
        precomputed_max_tag_per_gnode = precomputed.get(DICT_PARAM_TAG_PER_GNODE)
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
    node_list: List[HTML_ELEMENT], parent_name: str, max_tag_per_gnode: int, only_1b1: bool = False,
) -> NODE_DISTANCES_DICT_FORMAT:
    """
    See pseudo algorithm in Figure 6 in [1].

    Args:
        node_list:
        parent_name:
        max_tag_per_gnode:
        only_1b1: it might happen that a node is skipped for performance reasons and it is later needed in the
                  data records finding algorithm. So this allows to compute only the necessary in that case.

    Returns:

    """
    logging.debug(
        "in %s. parent_name=%s only_1b1=%s",
        _compare_combinations.__name__,
        parent_name,
        str(only_1b1),
    )

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

                        # NodeList[St..(k-1)]
                        left_gnode = GNode(parent_name, left_gnode_start, right_gnode_start,)
                        left_gnode_nodes = node_list[left_gnode.start : left_gnode.end]
                        left_gnode_str = nodes_to_string(
                            left_gnode_nodes, STR_DIST_USE_NODE_NAME_CLEANUP
                        )

                        # NodeList[St..(k-1)]
                        right_gnode = GNode(
                            parent_name, right_gnode_start, right_gnode_start + gnode_size,
                        )
                        right_gnode_nodes = node_list[right_gnode.start : right_gnode.end]
                        right_gnode_str = nodes_to_string(
                            right_gnode_nodes, STR_DIST_USE_NODE_NAME_CLEANUP
                        )

                        # check https://pypi.org/project/strsim/
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
    node: HTML_ELEMENT,
    node_namer: NodeNamer,
    minimum_depth: int,
    distances: DISTANCES_DICT_FORMAT,
    all_data_regions: DATA_REGION_DICT_FORMAT,
    distance_threshold: float,
    max_tag_per_gnode: int,
) -> None:
    """
    See pseudo code in Figure 8 in [1].
    This will make use of the distances computed previously to find data regions under each node such that
     the most number of children nodes are covered by the region.

    !!! IMPORTANT !!! it modifies the given `all_data_regions`, nothing is returned.

    Args:
          max_tag_per_gnode:
          distance_threshold:
          distances:
          minimum_depth:
          node_namer:
          node:
          all_data_regions: the dict where the sets of data regions (per node) will be stored.
    """
    node_depth = depth(node)

    # 1) if TreeDepth(Node) => 3 then
    if node_depth >= minimum_depth and should_process_node(node):

        # 2) Node.DRs = IdenDRs(1, Node, K, T);
        node_name = node_namer(node)
        n_children = len(node)
        node_distances = distances.get(node_name)

        logging.debug(
            "Will identify data regions. node_depth=%d node_name=%s n_children=%d",
            node_depth,
            node_name,
            n_children,
        )

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
        logging.debug(
            "saving data regions. node_depth=%d node_name=%s n_data_regions=%d",
            node_depth,
            node_name,
            len(temp_data_regions),
        )
        all_data_regions[node_name] |= temp_data_regions

    else:
        logging.debug("skipped node because of min depth. node_depth=%d", node_depth)
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
    """
    See pseudo code in Figure 9 in [1].
    This use scan all the possibilities of continuous data regions by iteratively adding gnodes until the
     min distance to find one is passed.
    Then, it will recursively call it when it will have scanned what is relevant to find other data regions
     under the same node.
    The goal is to find the biggest DR with the earliest node included.

    Args:
        start_index: only consider the nodes from this index and on (supposing whatever is behind already belongs
                      to a data region.
        node_name:
        n_children:
        node_distances:
        distance_threshold:
        max_tag_per_gnode:
    """

    if not node_distances:
        logging.debug("no distances, returning empty set. node_name=%s", node_name)
        return set()

    # 1 maxDR = [0, 0, 0];
    max_dr = DataRegion.empty()
    current_dr = DataRegion.empty()

    # 2 for (i = 1; i <= K; i++) /* compute for each i-combination */
    for gnode_size in range(1, max_tag_per_gnode + 1):

        # 3 for (f = start; f <= start+i; f++) /* start from each node */
        # for start_gnode_start_index in range(start_index, start_index + gnode_size + 1):
        for first_gn_start_idx in range(start_index, start_index + gnode_size):

            # 4 flag = true;
            dr_has_started = False
            logging.debug(
                "set up the dr_has_started to False. gnode_size=%d first_gn_start_idx=%d",
                gnode_size,
                first_gn_start_idx,
            )

            # 5 for (j = f; j < size(Node.Children); j+i)
            # for left_gnode_start in range(start_node, len(node) , gnode_size):
            for last_gn_start_idx in range(
                # start_gnode_start_index, len(node) - gnode_size + 1, gnode_size
                first_gn_start_idx + gnode_size,
                n_children - gnode_size + 1,
                gnode_size,
            ):

                # 6 if Distance(Node, i, j) <= T then
                gn_last = GNode(node_name, last_gn_start_idx, last_gn_start_idx + gnode_size,)
                gn_before_last = GNode(
                    node_name, last_gn_start_idx - gnode_size, last_gn_start_idx,
                )
                gn_pair = GNodePair(gn_before_last, gn_last)
                distance = node_distances[gnode_size][gn_pair]

                if distance <= distance_threshold:

                    # 7 if flag=true then
                    if not dr_has_started:
                        logging.debug(
                            "distance is BELLOW the threshold ==> close enough; "
                            "and the DR has NOT started ==> initializing it. "
                            "gnode_size=%d first_gn_start_idx=%d last_gn_start_idx=%d gn_pair=%s distance=%.2f ",
                            gnode_size,
                            first_gn_start_idx,
                            last_gn_start_idx,
                            gn_pair,
                            distance,
                        )

                        # 8 curDR = [i, j, 2*i];
                        current_dr = DataRegion.binary_from_last_gnode(gn_last)

                        # 9 flag = false;
                        dr_has_started = True

                    # 10 else curDR[3] = curDR[3] + i;
                    else:

                        logging.debug(
                            "distance is BELLOW the threshold ==> close enough; "
                            "and the DR HAS started ==> extending it. "
                            "gnode_size=%d first_gn_start_idx=%d last_gn_start_idx=%d gn_pair=%s distance=%.2f ",
                            gnode_size,
                            first_gn_start_idx,
                            last_gn_start_idx,
                            gn_pair,
                            distance,
                        )

                        current_dr = current_dr.extend_one_gnode()

                    logging.debug(
                        "current_dr=%s gnode_size=%d first_gn_start_idx=%d "
                        "last_gn_start_idx=%d gn_pair=%s distance=%.2f ",
                        current_dr,
                        gnode_size,
                        first_gn_start_idx,
                        last_gn_start_idx,
                        gn_pair,
                        distance,
                    )

                # 11 elseif flag = false then Exit-inner-loop;
                elif dr_has_started:
                    logging.debug(
                        "distance is ABOVE the threshold => too far; "
                        "and the DR has started => breaking it. "
                        "gnode_size=%d first_gn_start_idx=%d last_gn_start_idx=%d gn_pair=%s distance=%.2f ",
                        gnode_size,
                        first_gn_start_idx,
                        last_gn_start_idx,
                        gn_pair,
                        distance,
                    )
                    break

            # 13 if (maxDR[3] < curDR[3]) and (maxDR[2] = 0 or (curDR[2]<= maxDR[2]) then
            current_is_strictly_larger = max_dr.n_nodes_covered < current_dr.n_nodes_covered
            current_starts_at_same_node_or_before = (
                max_dr.is_empty
                or current_dr.first_gnode_start_index <= max_dr.first_gnode_start_index
            )

            if current_is_strictly_larger and current_starts_at_same_node_or_before:
                logging.debug(
                    "current_dr is larger than max_dr, replacing it. gnode_size=%d first_gn_start_idx=%d",
                    gnode_size,
                    first_gn_start_idx,
                )
                # 14 maxDR = curDR;
                max_dr = current_dr

            logging.debug(
                "current max_dr=%s. gnode_size=%d first_gn_start_idx=%d",
                max_dr,
                gnode_size,
                first_gn_start_idx,
            )

    logging.debug("final max_dr=%s", max_dr)

    # 16 if ( maxDR[3] != 0 ) then
    if not max_dr.is_empty:

        # 17 if (maxDR[2]+maxDR[3]-1 != size(Node.Children)) then
        last_covered_idx = max_dr.last_covered_tag_index

        if last_covered_idx < n_children - 1:

            recursion_start_index = last_covered_idx + 1
            logging.debug("calling recursion. recursion_start_index=%d", recursion_start_index)

            # 18 return {maxDR} ∪ IdentDRs(maxDR[2]+maxDR[3], Node, K, T)
            return {max_dr} | _identify_data_regions(
                start_index=recursion_start_index,
                node_name=node_name,
                n_children=n_children,
                node_distances=node_distances,
                distance_threshold=distance_threshold,
                max_tag_per_gnode=max_tag_per_gnode,
            )

        # 19 else return {maxDR}
        else:
            logging.debug("returning {{max_dr}}")
            return {max_dr}

    # 21 return ∅;
    logging.debug("max_dr is empty, returning empty set")
    return set()


def _uncovered_data_regions(node_drs: Set[DataRegion], child_idx: int) -> bool:
    """ True if child_idx is not covered any of the data regions `node_drs`."""
    # 1) for each data region DR in Node.DRs do
    for dr in node_drs:
        # 2) if Child in range DR[2] .. (DR[2] + DR[3] - 1) then
        if child_idx in dr:
            # 3) return null
            return False
    # 4) return Child.DRs
    return True


def find_data_records(
    root: HTML_ELEMENT,
    data_regions_per_node: DATA_REGION_DICT_FORMAT,
    distances: DISTANCES_DICT_FORMAT,
    node_namer: NodeNamer,
    edit_distance_threshold: MDREditDistanceThresholds,
    max_tag_per_gnode: int,
) -> DATA_RECORDS:
    """
    No pseudo code is given in [1] for this method. Read the description in section `3.3 Identify Data Records`.

    Args:
        root:
        data_regions_per_node:
        distances:
        node_namer:
        edit_distance_threshold:
        max_tag_per_gnode:

    Returns:
        all data records based on the given data regions (and distances)
    """

    all_data_regions: Set[DataRegion] = set.union(
        *(v for v in data_regions_per_node.values() if isinstance(v, set))
    )

    logging.debug("total nb of data regions to check: %d", len(all_data_regions))

    data_records = set()

    for dr in all_data_regions:
        gn_is_of_size_1 = dr.gnode_size == 1
        dr_parent_node = _get_node(root, dr.parent)
        dr_data_records = set()

        gnode: GNode
        for gnode in dr.get_gnode_iterator():
            logging.debug("checking gnode. data_region=%s gnode=%s", dr, gnode)

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
            logging.debug(
                "data records in gnode n_data_records=%d. data_region=%s gnode=%s",
                len(gn_data_records),
                dr,
                gnode,
            )

        # check disconnected data records
        # see section '3.4 Data Records not in Data Regions' of [1]
        if gn_is_of_size_1 and len(dr_data_records) > 0:
            logging.debug("checking for disconnected data records. data_region=%s", dr)

            there_are_nodes_out_of_dr = len(dr_parent_node) > dr.n_nodes_covered
            # if dr and data records have different parents, it's because they are in different levels
            drecs_parents_names = {drec[0].parent for drec in dr_data_records}
            all_data_records_are_in_level_below = all(
                name != dr.parent for name in drecs_parents_names
            )

            if there_are_nodes_out_of_dr and all_data_records_are_in_level_below:
                nodes_with_data_records: List[HTML_ELEMENT] = [
                    nd
                    for nd in dr_parent_node.getchildren()
                    if node_namer(nd) in drecs_parents_names
                ]
                a_data_record_node = nodes_with_data_records[0][0]
                a_drec_str = nodes_to_string([a_data_record_node], STR_DIST_USE_NODE_NAME_CLEANUP)

                not_covered_nodes: List[HTML_ELEMENT] = [
                    dr_parent_node[idx] for idx in range(len(dr_parent_node)) if idx not in dr
                ]
                for nd in not_covered_nodes:
                    candidate_drec_node: HTML_ELEMENT
                    for idx, candidate_drec_node in enumerate(nd.getchildren()):
                        dist = Levenshtein.ratio(
                            a_drec_str,
                            nodes_to_string([candidate_drec_node], STR_DIST_USE_NODE_NAME_CLEANUP),
                        )
                        if dist <= edit_distance_threshold.find_records_1:
                            new_drec = DataRecord([GNode(node_namer(nd), idx, idx + 1)])
                            dr_data_records.add(new_drec)
                            logging.debug(
                                "new disconnected data record found. data_region=%s new_drec=%s",
                                dr,
                                new_drec,
                            )
            else:
                logging.debug("no disconnected data record possible. data_region=%s", dr)

        data_records.update(dr_data_records)
        logging.debug(
            "data records in data region n_data_records=%d. data_region=%s",
            len(dr_data_records),
            dr,
        )

    logging.debug("total data records n_data_records=%d. data_region=%s", len(data_records))

    return data_records


def _find_records_1(
    gnode: GNode,
    gnode_node: HTML_ELEMENT,
    distances: DISTANCES_DICT_FORMAT,
    node_namer: NodeNamer,
    edit_distance_threshold: float,  # edit_distance_threshold.find_records_1
    max_tag_per_gnode: int,
) -> DATA_RECORDS:
    """
    Finding data records in a one-component generalized gnode_node.
    See pseudo-code in Figure 12 in [1].
    """

    has_children = len(gnode_node) > 1
    node_name = node_namer(gnode_node)

    logging.debug("in %s. gnode=%s node_name=%s", _find_records_1.__name__, gnode, node_name)

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
        logging.debug(
            "all children are similar and it is a table row ==> will create an individual data records for each child. "
            "gnode=%s node_name=%s",
            gnode,
            node_name,
        )

        # 3) each child node of R is a data record
        for i in range(len(gnode_node)):
            data_records_found.add(DataRecord([GNode(node_name, i, i + 1)]))

    # 4) else G itself is a data record.
    else:
        logging.debug("the gnode itself is a data record. gnode=%s node_name=%s", gnode, node_name)
        data_records_found.add(DataRecord([gnode]))

    return data_records_found


def _find_records_n(
    gnode: GNode,
    gnode_nodes: List[HTML_ELEMENT],
    distances: DISTANCES_DICT_FORMAT,
    node_namer: NodeNamer,
    distance_threshold: float,  # edit_distance_threshold.find_records_n
    max_tag_per_gnode: int,
) -> DATA_RECORDS:
    """
    Finding data records in an n-component generalized node.
    See pseudo code in Figure 15 in [1].
    """

    logging.debug("in %s. gnode=%s ", _find_records_n.__name__, gnode)

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

    all_have_same_nb_children = len(set(numbers_children)) == 1
    childrens_are_similar = None not in childrens_distances and all(
        child_distances and all(d <= distance_threshold for d in child_distances.values())
        for child_distances in childrens_distances
    )

    # 1) If the children gnode_nodes of each node in G are similar
    # 1...)   AND each node also has the same number of children then
    data_records_found = set()
    if not (all_have_same_nb_children and childrens_are_similar):
        logging.debug(
            "all nodes have the same number of children and they are all similar. "
            "will create a single data record for the gnode"
            "gnode=%s ",
            gnode,
        )

        # 3) else G itself is a data record.
        data_records_found.add(DataRecord([gnode]))

    else:

        logging.debug("will create a data record with the respective children. gnode=%s ", gnode)

        # 2) The corresponding children gnode_nodes of every node in G form a non-contiguous object description
        n_children = numbers_children[0]
        for i in range(n_children):
            data_records_found.add(
                DataRecord([GNode(node_namer(n), i, i + 1) for n in gnode_nodes])
            )

    return data_records_found


def _get_node(root: HTML_ELEMENT, node_name: str) -> HTML_ELEMENT:
    # todo(improvement) add some safety to this

    tag = node_name.split("-")[0]

    # this depends on the implementation of `NodeNamer`
    nodes = root.xpath(
        "//{tag}[@___tag_name___='{node_name}']".format(tag=tag, node_name=node_name)
    )
    if len(nodes) == 0:
        raise Exception("node not found node_name={}".format(node_name))
    return nodes[0]


def nodes_to_string(list_of_nodes: List[HTML_ELEMENT], use_node_name_cleanup: bool = False) -> str:
    if use_node_name_cleanup:
        list_of_nodes = [copy.deepcopy(c) for c in list_of_nodes]
        for c in list_of_nodes:
            NodeNamer.cleanup_all(c)
    return " ".join([lxml.etree.tostring(child).decode("utf-8").strip() for child in list_of_nodes])


def depth(node: HTML_ELEMENT) -> int:
    """ The tree depth. The root has depth = 0."""
    d = 0
    while node is not None:
        d += 1
        node = node.getparent()
    return d - 1


def should_process_node(node: HTML_ELEMENT):
    """ This defines which types of html nodes will be considered as a potential data record. """
    return node.tag in (
        "table",
        "tr",
        "th",
        "td",
        "thead",
        "tbody",
        "tfoot",
        "form",
        # [algo-list-elements-considered]
        # this makes the algo considerably slower
        "ol",
        "ul",
        "li",
    )


def paint_data_records(data_records_nodes: List[List[HTML_ELEMENT]]):
    """ This will put a random color as the background of nodes in the same data record. """
    colors = generate_random_colors(len(data_records_nodes))
    for record_nodes, color in zip(data_records_nodes, colors):
        for e in record_nodes:
            e.set("style", "background-color: #{} !important;".format(color))
