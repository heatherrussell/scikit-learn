"""
This module defines export functions for decision trees.
"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Trevor Stephens <trev.stephens@gmail.com>
# Licence: BSD 3 clause

import numpy as np

from ..externals import six

from . import _tree


def export_graphviz(decision_tree, out_file="tree.dot", max_depth=None,
                    feature_names=None, class_names=None, label='all',
                    filled=False, leaf=False, metric=True, node_ids=False,
                    proportion=False, ps=True, rotate=False, rounded=False):
    """Export a decision tree in DOT format.

    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)

    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported to GraphViz.

    out_file : file object or string, optional (default="tree.dot")
        Handle or name of the output file.

    max_depth : int, optional (default=None)
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : list of strings, optional (default=None)
        Names of each of the features.

    class_names : list of strings, bool or None, optional (default=None)
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and unsupported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, optional (default='all')
        Whether to show informative labels for error metric, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, optional (default=False)
        When set to ``True``, paint nodes to indicate majority class.

    leaf : bool, optional (default=False)
        When set to ``True``, draw the leaf nodes at the bottom of the tree.

    metric : bool, optional (default=True)
        When set to ``True``, show the error metric at each node.

    node_ids : bool, optional (default=False)
        When set to ``True``, show the ID number on each node.

    proportion : bool, optional (default=False)
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    ps : bool, optional (default=True)
        When set to ``True``, ignore special characters for PostScript
        compatibility.

    rotate : bool, optional (default=False)
        When set to ``True``, orient tree left to right rather than top-down.

    rounded : bool, optional (default=False)
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier()
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.export_graphviz(clf,
    ...     out_file='tree.dot')                # doctest: +SKIP
    """

    def colour_brew(n):
        # Initialize saturation & value; calculate chroma & value shift
        s, v = 0.75, 0.9
        c = s * v
        m = v - c

        colours = []
        # Loop through n equally spaced hues
        for h in np.arange(25, 385, 360./n).astype(int):

            # Calculate some intermediate values
            h_bar = h / 60.
            x = c * (1 - abs((h_bar % 2) - 1))

            # Initialize RGB with same hue & chroma as our colour
            if h_bar < 1:
                r, g, b = c, x, 0
            elif h_bar < 2:
                r, g, b = x, c, 0
            elif h_bar < 3:
                r, g, b = 0, c, x
            elif h_bar < 4:
                r, g, b = 0, x, c
            elif h_bar < 5:
                r, g, b = x, 0, c
            elif h_bar < 6:
                r, g, b = c, 0, x
            else:
                r, g, b = c, x, 0

            # Shift the initial RGB values to match value and store
            rgb = [(int(255 * (r + m))),
                   (int(255 * (g + m))),
                   (int(255 * (b + m)))]
            colours.append(rgb)

        return colours

    def get_colour(colours, value, bounds):
        # Find the appropriate colour & intensity for a node
        if bounds is None:
            # Classification tree
            colour = [c for c in colours[list(value).index(value.max())]]
            sorted_values = sorted(value, reverse=True)
            alpha = int(255 * (sorted_values[0] - sorted_values[1]) /
                        (1 - sorted_values[1]))
        else:
            # Regression tree
            colour = [c for c in colours[0]]
            alpha = int(255 * ((value - bounds[0]) / (bounds[1] - bounds[0])))

        # Return html colour code in #RRGGBBAA format
        colour.append(alpha)
        hex_codes = [str(i) for i in range(10)]
        hex_codes.extend(['a', 'b', 'c', 'd', 'e', 'f'])
        colour = [hex_codes[c // 16] + hex_codes[c % 16] for c in colour]

        return '#' + ''.join(colour)

    def recurse_subtree(tree, node_id, values=None):
        # Gather the leaf node classifications for a subtree
        if values is None:
            values = []

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if left_child != _tree.TREE_LEAF:
            recurse_subtree(tree, left_child, values=values)
            recurse_subtree(tree, right_child, values=values)
        else:
            if tree.n_classes[0] != 1:
                # Classification tree
                values.append(tree.value[node_id][0, :])
            else:
                # Regression tree, need a weighted average
                values.append(tree.value[node_id][0, :] *
                              float(tree.weighted_n_node_samples[node_id]))

        return values

    def node_to_str(tree, node_id, criterion, value=None):

        # Should labels be shown?
        labels = (label == 'root' and node_id == 0) or label == 'all'

        # PostScript compatibility for special characters
        string_segments = ['<', '&#35;', '<SUB>', '</SUB>',
                           '&le;', '<br/>', '>']
        if ps:
            string_segments = ['"', '#', '[', ']', '<=', '\\n', '"']

        # Build up node string as determined by plot_options
        node_string = string_segments[0]

        # Write node ID
        if node_ids:
            if labels:
                node_string += 'node '
            node_string += (string_segments[1] + str(node_id) +
                            string_segments[5])

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (string_segments[2],
                                       tree.feature[node_id],
                                       string_segments[3])
            node_string += '%s %s %s%s' % (feature,
                                           string_segments[4],
                                           round(tree.threshold[node_id], 4),
                                           string_segments[5])

        # Write error metric
        if metric:
            if not isinstance(criterion, six.string_types):
                criterion = "impurity"
            if labels:
                node_string += '%s = ' % criterion
            node_string += (str(round(tree.impurity[node_id], 4)) +
                            string_segments[5])

        # Write node sample count
        if labels:
            node_string += 'samples = '
        if proportion:
            percent = (100. * tree.n_node_samples[node_id] /
                       float(tree.n_node_samples[0]))
            node_string += (str(round(percent, 1)) + '%' +
                            string_segments[5])
        else:
            node_string += (str(tree.n_node_samples[node_id]) +
                            string_segments[5])

        # Write node class distribution / regression value
        if (tree.n_outputs == 1 or
                tree.children_left[node_id] == _tree.TREE_LEAF):
            # Format value string depending on classification/regression
            if value is None:
                value = tree.value[node_id]
                if tree.n_outputs == 1:
                    value = value[0, :]
                if proportion and tree.n_classes[0] != 1:
                    value = value / tree.weighted_n_node_samples[node_id]
            elif proportion or tree.n_classes[0] == 1:
                # For classification this will show the proportion of samples
                # For regression this will find the weighted average
                value = value / tree.weighted_n_node_samples[node_id]
            if labels:
                node_string += 'value = '
            if tree.n_classes[0] == 1:
                # Regression
                value_text = np.around(value, 4)
            elif proportion:
                # Classification
                value_text = np.around(value, 2)
            elif np.all(np.equal(np.mod(value, 1), 0)):
                # Classification without floating-point weights
                value_text = value.astype(int)
            else:
                # Classification with floating-point weights
                value_text = np.around(value, 4)
            # Strip whitespace
            value_text = str(value_text.astype('S32')).replace("b'", "'")
            value_text = value_text.replace("' '", ", ").replace("'", "")
            if tree.n_classes[0] == 1 and tree.n_outputs == 1:
                value_text = value_text.replace("[", "").replace("]", "")
            value_text = value_text.replace("\n", string_segments[5])
            node_string += value_text + string_segments[5]

        # Write node majority class
        if (class_names is not None and
                tree.n_classes[0] != 1 and
                tree.n_outputs == 1):
            # Only done for single-output classification trees
            if labels:
                node_string += 'class = '
            if class_names is not True:
                class_name = class_names[np.argmax(value)]
            else:
                class_name = "y%s%s%s" % (string_segments[2],
                                          np.argmax(value),
                                          string_segments[3])
            node_string += class_name

        # Clean up any trailing newlines
        if node_string[-2:] == '\\n':
            node_string = node_string[:-2]
        if node_string[-5:] == '<br/>':
            node_string = node_string[:-5]

        return node_string + string_segments[-1]

    def recurse(tree, node_id, criterion, parent=None, depth=0, colours=None):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Gather sub-tree's leaf node classifications if required
        leaves = None
        if tree.n_outputs == 1:
            leaves = np.sum(np.array(recurse_subtree(tree, node_id)), axis=0)

        # Get colours if required, only performed once
        if filled and colours is None and tree.n_outputs == 1:
            colours = colour_brew(tree.n_classes[0])

        # Add node with description
        if max_depth is None or depth <= max_depth:

            # Collect ranks for 'leaf' option in plot_options
            if left_child == _tree.TREE_LEAF:
                ranks['leaves'].append(str(node_id))
            elif str(depth) not in ranks:
                ranks[str(depth)] = [str(node_id)]
            else:
                ranks[str(depth)].append(str(node_id))

            out_file.write('%d [label=%s'
                           % (node_id,
                              node_to_str(tree, node_id, criterion, leaves)))

            if filled:
                # Fetch appropriate colour for node
                if tree.n_outputs == 1:
                    bounds = None
                    if tree.n_classes == 1:
                        # Find max and min values in leaf nodes for regression
                        bounds = (np.min(tree.value[tree.feature < 0]),
                                  np.max(tree.value[tree.feature < 0]))
                    samples = tree.weighted_n_node_samples[node_id]
                    node_colour = get_colour(colours,
                                             (leaves / samples),
                                             bounds)
                else:
                    # If multi-output colour node as white
                    node_colour = '#FFFFFF'
                out_file.write(', fillcolor="%s"' % node_colour)
            out_file.write('] ;\n')

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d' % (parent, node_id))
                if parent == 0:
                    # Draw True/False labels if parent is root node
                    angles = np.array([45, -45]) * ((rotate - .5) * -2)
                    out_file.write(' [labeldistance=2.5, labelangle=')
                    if node_id == 1:
                        out_file.write('%d, headlabel="True"]' % angles[0])
                    else:
                        out_file.write('%d, headlabel="False"]' % angles[1])
                out_file.write(' ;\n')

            if left_child != _tree.TREE_LEAF:
                recurse(tree, left_child, criterion=criterion, parent=node_id,
                        depth=depth + 1, colours=colours)
                recurse(tree, right_child, criterion=criterion, parent=node_id,
                        depth=depth + 1, colours=colours)

        else:
            ranks['leaves'].append(str(node_id))

            out_file.write('%d [label="(...)"' % node_id)
            if filled:
                # Colour cropped nodes grey
                out_file.write(', fillcolor="#C0C0C0"')
            out_file.write('] ;\n' % node_id)

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d ;\n' % (parent, node_id))

    own_file = False
    try:
        if isinstance(out_file, six.string_types):
            if six.PY3:
                out_file = open(out_file, "w", encoding="utf-8")
            else:
                out_file = open(out_file, "wb")
            own_file = True

        # The depth of each node for plotting with 'leaf' option
        ranks = {'leaves': []}

        out_file.write('digraph Tree {\n')

        # Specify node aesthetics
        out_file.write('node [shape=box')
        rounded_filled = []
        if filled:
            rounded_filled.append('filled')
        if rounded:
            rounded_filled.append('rounded')
        if len(rounded_filled) > 0:
            out_file.write(', style="%s", color="black"'
                           % ", ".join(rounded_filled))
        if rounded:
            out_file.write(', fontname=helvetica')
        out_file.write('] ;\n')

        # Specify graph & edge aesthetics
        if leaf:
            out_file.write('graph [ranksep=equally, splines=polyline] ;\n')
        if rounded:
            out_file.write('edge [fontname=helvetica] ;\n')
        if rotate:
            out_file.write('rankdir=LR ;\n')

        # Now recurse the tree and add node & edge attributes
        if isinstance(decision_tree, _tree.Tree):
            recurse(decision_tree, 0, criterion="impurity")
        else:
            recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

        # If required, draw leaf nodes at same depth as each other
        if leaf:
            for rank in sorted(ranks):
                out_file.write("{rank=same ; " +
                               "; ".join(r for r in ranks[rank]) + "} ;\n")
        out_file.write("}")

    finally:
        if own_file:
            out_file.close()
