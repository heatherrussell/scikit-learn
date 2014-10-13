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


def export_graphviz(decision_tree, out_file="tree.dot", feature_names=None,
                    max_depth=None, plot_options=None):
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

    feature_names : list of strings, optional (default=None)
        Names of each of the features.

    max_depth : int, optional (default=None)
        The maximum depth of the representation. If None, the tree is fully
        generated.

    plot_options : list of strings, optional (default=None)
                   (which implies ['samples', 'metric', 'labels'])
        Plot the tree with aesthetic options. Available keywords include:
            - 'filled' : Paint nodes to indicate majority class
            - 'rounded' : Draw node boxes with rounded corners
            - 'helvetica' : Replace Times-Roman font with Helvetica
            - 'leaf' : Draw the leaf nodes at the bottom of the tree
            - 'rotate' : Orient the tree left to right (default is top-down)
            - 'yes' : Show Yes/No labels at first split
            - 'true' : Show True/False labels at first split
            - 'id' : Show the ID number on each node
            - 'metric' : Show the error metric at each node
            - 'class' : Show majority class at each node (classification only)
            - 'values' : Show the class breakdown at each node
                - default is to show this only for the leaf nodes
            - 'samples' : Show the number of samples in each node
            - 'proportion' : Change the display of 'values' and/or 'samples'
                             to be proportions and percentages respectively
            - 'labels' : Show informative labels for 'samples', 'metric', etc
            - 'label' : Same as 'labels', but for root node only

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

    def parse_options():
        # Clean up plot_options and determine if subtree recursion is required
        final_options = set(['samples', 'metric', 'labels'])
        valid_options = set(['class', 'filled', 'helvetica', 'id', 'label',
                             'labels', 'leaf', 'metric', 'proportion',
                             'rotate', 'rounded', 'samples', 'true', 'values',
                             'yes'])
        # The set of options that require subtree recursion
        recurse_options = set(['filled', 'class', 'values'])

        if (hasattr(plot_options, '__iter__') and
                not isinstance(plot_options, str)):
            bad_options = set(plot_options) - valid_options
            good_options = set(plot_options) & valid_options
            if len(bad_options) != 0:
                raise ValueError("Invalid plot options provided: " +
                                 str(list(bad_options)))
            if len(good_options) != 0:
                # If empty list supplied, use defaults
                final_options = good_options
        else:
            raise ValueError("Expected list for plot_options.")

        return final_options, len(recurse_options & final_options) > 0

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
            if len(tree.value[node_id][0, :]) != 1:
                # Classification tree
                values.append(tree.value[node_id][0, :])
            else:
                # Regression tree
                values.append(tree.value[node_id][0, :] *
                              float(tree.n_node_samples[node_id]))

        return values

    def node_to_str(tree, node_id, criterion, value=None):

        # Build up node string as determined by plot_options
        node_string = ''

        # Should labels be shown?
        labels = (('label' in plot_options and node_id == 0) or
                  'labels' in plot_options)

        if 'id' in plot_options:
            if labels:
                node_string += 'node '
            node_string += '#' + str(node_id) + '\\n'

        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = "X[%s]" % tree.feature[node_id]
            node_string += '%s <= %s\\n' % (feature,
                                            round(tree.threshold[node_id], 4))

        if 'metric' in plot_options:
            if not isinstance(criterion, six.string_types):
                criterion = "impurity"
            if labels:
                node_string += '%s = ' % criterion
            node_string += str(round(tree.impurity[node_id], 4)) + '\\n'

        if 'samples' in plot_options:
            if labels:
                node_string += 'samples = '
            if 'proportion' in plot_options:
                percent = (100. * tree.n_node_samples[node_id] /
                           float(tree.n_node_samples[0]))
                node_string += str(round(percent, 1)) + '%\\n'
            else:
                node_string += str(tree.n_node_samples[node_id]) + '\\n'

        if (('values' in plot_options and tree.n_outputs == 1) or
                tree.children_left[node_id] == _tree.TREE_LEAF):
            # Format value string depending on classification/regression
            if value is None:
                value = tree.value[node_id]
                if tree.n_outputs == 1:
                    value = value[0, :]
                if 'proportion' in plot_options and tree.n_classes[0] != 1:
                    value = value / tree.n_node_samples[node_id]
            elif 'proportion' in plot_options or tree.n_classes[0] == 1:
                value = value / tree.n_node_samples[node_id]
            if labels:
                node_string += 'value = '
            if tree.n_classes[0] != 1:
                node_string += str(np.around(value, 2)) + '\\n'
            else:
                node_string += str(np.around(value, 4)) + '\\n'

        if ('class' in plot_options and
                tree.n_classes[0] != 1 and
                tree.n_outputs == 1):
            # Only done for single-output classification trees
            if labels:
                node_string += 'class = '
            node_string += str(list(value).index(value.max()))

        if node_string[-2:] == '\\n':
            # Clean up any trailing newlines
            node_string = node_string[:-2]

        return node_string

    def recurse(tree, node_id, criterion, parent=None, depth=0, colours=None):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Gather sub-tree's leaf node classifications if required
        leaves = None
        if subtree_required and tree.n_outputs == 1:
            leaves = np.sum(np.array(recurse_subtree(tree, node_id)), axis=0)
        if ('filled' in plot_options and
                colours is None and
                tree.n_outputs == 1):
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

            out_file.write('%d [label="%s"'
                           % (node_id,
                              node_to_str(tree, node_id, criterion, leaves)))
            if 'filled' in plot_options:
                if tree.n_outputs == 1:
                    bounds = None
                    if tree.n_classes == 1:
                        bounds = (np.min(tree.value[tree.feature < 0]),
                                  np.max(tree.value[tree.feature < 0]))
                    node_colour = get_colour(colours,
                                             (leaves /
                                              tree.n_node_samples[node_id]),
                                             bounds)
                else:
                    node_colour = '#FFFFFF'
                out_file.write(', fillcolor="%s"' % node_colour)
            out_file.write('] ;\n')

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d' % (parent, node_id))
                root_labels = None
                if 'yes' in plot_options:
                    root_labels = ('Yes', 'No')
                if 'true' in plot_options:
                    root_labels = ('True', 'False')
                if parent == 0 and root_labels is not None:
                    root_angles = np.array([45, -45])
                    if 'rotate' in plot_options:
                        root_angles *= -1
                    out_file.write(' [labeldistance=2.5, labelangle=')
                    if node_id == 1:
                        out_file.write('%d, headlabel="%s"]'
                                       % (root_angles[0], root_labels[0]))
                    else:
                        out_file.write('%d, headlabel="%s"]'
                                       % (root_angles[1], root_labels[1]))
                out_file.write(' ;\n')

            if left_child != _tree.TREE_LEAF:
                recurse(tree, left_child, criterion=criterion, parent=node_id,
                        depth=depth + 1, colours=colours)
                recurse(tree, right_child, criterion=criterion, parent=node_id,
                        depth=depth + 1, colours=colours)

        else:
            ranks['leaves'].append(str(node_id))

            out_file.write('%d [label="(...)"' % node_id)
            if 'filled' in plot_options:
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

        if plot_options is None:
            plot_options = ['samples', 'metric', 'labels']
        plot_options, subtree_required = parse_options()

        out_file.write('digraph Tree {\n')

        # Specify node aesthetics
        out_file.write('node [shape=box')
        rounded_filled = set(['rounded', 'filled']) & plot_options
        if len(rounded_filled) > 0:
            out_file.write(', style="%s", color="black"'
                           % ",".join(rounded_filled))
        if 'helvetica' in plot_options:
            out_file.write(', fontname=helvetica')
        out_file.write('] ;\n')

        # Specify graph & edge aesthetics
        if 'leaf' in plot_options:
            out_file.write('graph [ranksep=equally, splines=polyline] ;\n')
        if 'helvetica' in plot_options:
            out_file.write('edge [fontname=helvetica] ;\n')
        if 'rotate' in plot_options:
            out_file.write('rankdir=LR ;\n')

        # Now recurse the tree and add node & edge attributes
        if isinstance(decision_tree, _tree.Tree):
            recurse(decision_tree, 0, criterion="impurity")
        else:
            recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

        # If required, draw leaf nodes at same depth as each other
        if 'leaf' in plot_options:
            for rank in sorted(ranks):
                out_file.write("{rank=same ; " +
                               "; ".join(r for r in ranks[rank]) + "} ;\n")
        out_file.write("}")

    finally:
        if own_file:
            out_file.close()
