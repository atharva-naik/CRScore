from __future__ import division

from collections import defaultdict

import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa (For 3D plots)
from matplotlib import pyplot as plt
from matplotlib import gridspec, animation
import param
from ...core import (OrderedDict, HoloMap, AdjointLayout, NdLayout,
                     GridSpace, Element, CompositeOverlay, Element3D,
                     Empty, Collator)
from ...core.options import Store, Compositor
from ...core.util import int_to_roman, int_to_alpha, basestring
from ...core import traversal
from ..plot import DimensionedPlot, GenericLayoutPlot, GenericCompositePlot
from ..util import get_dynamic_mode, initialize_sampled
from .renderer import MPLRenderer


class MPLPlot(DimensionedPlot):
    """
    An MPLPlot object draws a matplotlib figure object when called or
    indexed but can also return a matplotlib animation object as
    appropriate. MPLPlots take element objects such as Image, Contours
    or Points as inputs and plots them in the appropriate format using
    matplotlib. As HoloMaps are supported, all plots support animation
    via the anim() method.
    """

    renderer = MPLRenderer
    sideplots = {}

    fig_alpha = param.Number(default=1.0, bounds=(0, 1), doc="""
        Alpha of the overall figure background.""")

    fig_bounds = param.NumericTuple(default=(0.15, 0.15, 0.85, 0.85),
                                       doc="""
        The bounds of the overall figure as a 4-tuple of the form
        (left, bottom, right, top), defining the size of the border
        around the subplots.""")

    fig_inches = param.Parameter(default=4, doc="""
        The overall matplotlib figure size in inches.  May be set as
        an integer in which case it will be used to autocompute a
        size. Alternatively may be set with an explicit tuple or list,
        in which case it will be applied directly after being scaled
        by fig_size. If either the width or height is set to None,
        it will be computed automatically.""")

    fig_latex = param.Boolean(default=False, doc="""
        Whether to use LaTeX text in the overall figure.""")

    fig_rcparams = param.Dict(default={}, doc="""
        matplotlib rc parameters to apply to the overall figure.""")

    fig_size = param.Integer(default=100, bounds=(1, None), doc="""
        Size relative to the supplied overall fig_inches in percent.""")

    initial_hooks = param.HookList(default=[], doc="""
        Optional list of hooks called before plotting the data onto
        the axis. The hook is passed the plot object and the displayed
        object, other plotting handles can be accessed via plot.handles.""")

    final_hooks = param.HookList(default=[], doc="""
        Optional list of hooks called when finalizing an axis.
        The hook is passed the plot object and the displayed
        object, other plotting handles can be accessed via plot.handles.""")

    finalize_hooks = param.HookList(default=[], doc="""
        Optional list of hooks called when finalizing an axis.
        The hook is passed the plot object and the displayed
        object, other plotting handles can be accessed via plot.handles.""")

    sublabel_format = param.String(default=None, allow_None=True, doc="""
        Allows labeling the subaxes in each plot with various formatters
        including {Alpha}, {alpha}, {numeric} and {roman}.""")

    sublabel_position = param.NumericTuple(default=(-0.35, 0.85), doc="""
         Position relative to the plot for placing the optional subfigure label.""")

    sublabel_size = param.Number(default=18, doc="""
         Size of optional subfigure label.""")

    projection = param.ObjectSelector(default=None,
                                      objects=['3d', 'polar', None], doc="""
        The projection of the plot axis, default of None is equivalent to
        2D plot, '3d' and 'polar' are also supported.""")

    show_frame = param.Boolean(default=True, doc="""
        Whether or not to show a complete frame around the plot.""")

    _close_figures = True

    def __init__(self, fig=None, axis=None, **params):
        self._create_fig = True
        super(MPLPlot, self).__init__(**params)
        # List of handles to matplotlib objects for animation update
        scale = self.fig_size/100.
        if isinstance(self.fig_inches, (tuple, list)):
            self.fig_inches = [None if i is None else i*scale
                               for i in self.fig_inches]
        else:
            self.fig_inches *= scale
        fig, axis = self._init_axis(fig, axis)
        self.handles['fig'] = fig
        self.handles['axis'] = axis

        if self.final_hooks and self.finalize_hooks:
            self.warning('Set either final_hooks or deprecated '
                         'finalize_hooks, not both.')
        self.finalize_hooks = self.final_hooks


    def _init_axis(self, fig, axis):
        """
        Return an axis which may need to be initialized from
        a new figure.
        """
        if not fig and self._create_fig:
            rc_params = self.fig_rcparams
            if self.fig_latex:
                rc_params['text.usetex'] = True
            with mpl.rc_context(rc=rc_params):
                fig = plt.figure()
                l, b, r, t = self.fig_bounds
                inches = self.fig_inches
                fig.subplots_adjust(left=l, bottom=b, right=r, top=t)
                fig.patch.set_alpha(self.fig_alpha)
                if isinstance(inches, (tuple, list)):
                    inches = list(inches)
                    if inches[0] is None:
                        inches[0] = inches[1]
                    elif inches[1] is None:
                        inches[1] = inches[0]
                    fig.set_size_inches(list(inches))
                else:
                    fig.set_size_inches([inches, inches])
                axis = fig.add_subplot(111, projection=self.projection)
                axis.set_aspect('auto')

        return fig, axis


    def _subplot_label(self, axis):
        layout_num = self.layout_num if self.subplot else 1
        if self.sublabel_format and not self.adjoined and layout_num > 0:
            from mpl_toolkits.axes_grid1.anchored_artists import AnchoredText
            labels = {}
            if '{Alpha}' in self.sublabel_format:
                labels['Alpha'] = int_to_alpha(layout_num-1)
            elif '{alpha}' in self.sublabel_format:
                labels['alpha'] = int_to_alpha(layout_num-1, upper=False)
            elif '{numeric}' in self.sublabel_format:
                labels['numeric'] = self.layout_num
            elif '{Roman}' in self.sublabel_format:
                labels['Roman'] = int_to_roman(layout_num)
            elif '{roman}' in self.sublabel_format:
                labels['roman'] = int_to_roman(layout_num).lower()
            at = AnchoredText(self.sublabel_format.format(**labels), loc=3,
                              bbox_to_anchor=self.sublabel_position, frameon=False,
                              prop=dict(size=self.sublabel_size, weight='bold'),
                              bbox_transform=axis.transAxes)
            at.patch.set_visible(False)
            axis.add_artist(at)


    def _finalize_axis(self, key):
        """
        General method to finalize the axis and plot.
        """
        if 'title' in self.handles:
            self.handles['title'].set_visible(self.show_title)

        self.drawn = True
        if self.subplot:
            return self.handles['axis']
        else:
            fig = self.handles['fig']
            if self._close_figures: plt.close(fig)
            return fig


    @property
    def state(self):
        return self.handles['fig']

    def anim(self, start=0, stop=None, fps=30):
        """
        Method to return a matplotlib animation. The start and stop
        frames may be specified as well as the fps.
        """
        figure = self.initialize_plot()
        anim = animation.FuncAnimation(figure, self.update_frame,
                                       frames=self.keys,
                                       interval = 1000.0/fps)
        # Close the figure handle
        if self._close_figures: plt.close(figure)
        return anim

    def update(self, key):
        rc_params = self.fig_rcparams
        if self.fig_latex:
            rc_params['text.usetex'] = True
        mpl.rcParams.update(rc_params)
        if len(self) == 1 and key == 0 and not self.drawn:
            return self.initialize_plot()
        return self.__getitem__(key)



class CompositePlot(GenericCompositePlot, MPLPlot):
    """
    CompositePlot provides a baseclass for plots coordinate multiple
    subplots to form a Layout.
    """

    def update_frame(self, key, ranges=None):
        ranges = self.compute_ranges(self.layout, key, ranges)
        for subplot in self.subplots.values():
            subplot.update_frame(key, ranges=ranges)
        axis = self.handles['axis']
        self.update_handles(axis, self.layout, key, ranges)



class GridPlot(CompositePlot):
    """
    Plot a group of elements in a grid layout based on a GridSpace element
    object.
    """

    aspect = param.Parameter(default='equal', doc="""
        Aspect ratios on GridPlot should be automatically determined.""")

    padding = param.Number(default=0.1, doc="""
        The amount of padding as a fraction of the total Grid size""")

    shared_xaxis = param.Boolean(default=False, doc="""
        If enabled the x-axes of the GridSpace will be drawn from the
        objects inside the Grid rather than the GridSpace dimensions.""")

    shared_yaxis = param.Boolean(default=False, doc="""
        If enabled the x-axes of the GridSpace will be drawn from the
        objects inside the Grid rather than the GridSpace dimensions.""")

    show_frame = param.Boolean(default=False, doc="""
        Whether to draw a frame around the Grid.""")

    show_legend = param.Boolean(default=False, doc="""
        Legends add to much clutter in a grid and are disabled by default.""")

    tick_format = param.String(default="%.2f", doc="""
        Formatting string for the GridPlot ticklabels.""")

    xaxis = param.ObjectSelector(default='bottom',
                                 objects=['bottom', 'top', None], doc="""
        Whether and where to display the xaxis, supported options are
        'bottom', 'top' and None.""")

    yaxis = param.ObjectSelector(default='left',
                                 objects=['left', 'right', None], doc="""
        Whether and where to display the yaxis, supported options are
        'left', 'right' and None.""")

    xrotation = param.Integer(default=0, bounds=(0, 360), doc="""
        Rotation angle of the xticks.""")

    yrotation = param.Integer(default=0, bounds=(0, 360), doc="""
        Rotation angle of the yticks.""")

    def __init__(self, layout, axis=None, create_axes=True, ranges=None,
                 keys=None, dimensions=None, layout_num=1, **params):
        if not isinstance(layout, GridSpace):
            raise Exception("GridPlot only accepts GridSpace.")
        self.layout = layout
        self.cols, self.rows = layout.shape
        self.layout_num = layout_num
        extra_opts = self.lookup_options(layout, 'plot').options
        if not keys or not dimensions:
            dimensions, keys = traversal.unique_dimkeys(layout)
        if 'uniform' not in params:
            params['uniform'] = traversal.uniform(layout)
        dynamic, sampled = get_dynamic_mode(layout)
        if sampled:
            initialize_sampled(layout, dimensions, keys[0])
        super(GridPlot, self).__init__(keys=keys, dimensions=dimensions,
                                       dynamic=dynamic,
                                       **dict(extra_opts, **params))
        # Compute ranges layoutwise
        grid_kwargs = {}
        if axis is not None:
            bbox = axis.get_position()
            l, b, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height
            grid_kwargs = {'left': l, 'right': l+w, 'bottom': b, 'top': b+h}
            self.position = (l, b, w, h)

        self.fig_inches = self._get_size()
        self._layoutspec = gridspec.GridSpec(self.rows, self.cols, **grid_kwargs)
        self.subplots, self.subaxes, self.layout = self._create_subplots(layout, axis, ranges, create_axes)


    def _get_size(self):
        max_dim = max(self.layout.shape)
        # Reduce plot size as GridSpace gets larger
        shape_factor = 1. / max_dim
        # Expand small grids to a sensible viewing size
        expand_factor = 1 + (max_dim - 1) * 0.1
        scale_factor = expand_factor * shape_factor
        cols, rows = self.layout.shape
        if isinstance(self.fig_inches, (tuple, list)):
            fig_inches = list(self.fig_inches)
            if fig_inches[0] is None:
                fig_inches[0] = fig_inches[1] * (cols/rows)
            if fig_inches[1] is None:
                fig_inches[1] = fig_inches[0] * (rows/cols)
            return fig_inches
        else:
            fig_inches = (self.fig_inches,)*2
            return (scale_factor * cols * fig_inches[0],
                    scale_factor * rows * fig_inches[1])


    def _create_subplots(self, layout, axis, ranges, create_axes):
        layout = layout.map(Compositor.collapse_element, [CompositeOverlay],
                            clone=False)
        norm_opts = self._deep_options(layout, 'norm', ['axiswise'], [Element])
        axiswise = all(v.get('axiswise', False) for v in norm_opts.values())

        if not ranges:
            self.handles['fig'].set_size_inches(self.fig_inches)
        subplots, subaxes = OrderedDict(), OrderedDict()
        frame_ranges = self.compute_ranges(layout, None, ranges)
        frame_ranges = OrderedDict([(key, self.compute_ranges(layout, key, frame_ranges))
                                    for key in self.keys])
        collapsed_layout = layout.clone(shared_data=False, id=layout.id)
        r, c = (0, 0)
        for coord in layout.keys(full_grid=True):
            if not isinstance(coord, tuple): coord = (coord,)
            view = layout.data.get(coord, None)
            # Create subplot
            if view is not None:
                vtype = view.type if isinstance(view, HoloMap) else view.__class__
                opts = self.lookup_options(view, 'plot').options
            else:
                continue

            # Create axes
            kwargs = {}
            if create_axes:
                threed = issubclass(vtype, Element3D)
                subax = plt.subplot(self._layoutspec[r, c],
                                    projection='3d' if threed else None)

                if not axiswise and self.shared_xaxis and self.xaxis is not None:
                    self.xaxis = 'top'
                if not axiswise and self.shared_yaxis and self.yaxis is not None:
                    self.yaxis = 'right'

                # Disable subplot axes depending on shared axis options
                # and the position in the grid
                if (self.shared_xaxis or self.shared_yaxis) and not axiswise:

                    if c == 0 and r != 0:
                        subax.xaxis.set_ticks_position('none')
                        kwargs['xaxis'] = 'bottom-bare'
                    if c != 0 and r == 0 and not layout.ndims == 1:
                        subax.yaxis.set_ticks_position('none')
                        kwargs['yaxis'] = 'left-bare'
                    if r != 0 and c != 0:
                        kwargs['xaxis'] = 'bottom-bare'
                        kwargs['yaxis'] = 'left-bare'
                    if not self.shared_xaxis:
                        kwargs['xaxis'] = 'bottom-bare'
                    if not self.shared_yaxis:
                        kwargs['yaxis'] = 'left-bare'
                else:
                    kwargs['xaxis'] = 'bottom-bare'
                    kwargs['yaxis'] = 'left-bare'
                subaxes[(r, c)] = subax
            else:
                subax = None
            if issubclass(vtype, CompositeOverlay) and (c == self.cols - 1 and
                                                        r == self.rows//2):
                kwargs['show_legend'] = self.show_legend
                kwargs['legend_position'] = 'right'

            # Create subplot
            if view is not None:
                params = dict(fig=self.handles['fig'], axis=subax,
                              dimensions=self.dimensions, show_title=False,
                              subplot=not create_axes, ranges=frame_ranges,
                              uniform=self.uniform, keys=self.keys,
                              show_legend=False)
                plotting_class = Store.registry['matplotlib'][vtype]
                subplot = plotting_class(view,  **dict(opts, **dict(params, **kwargs)))
                collapsed_layout[coord] = subplot.layout if isinstance(subplot, CompositePlot) else subplot.hmap
                subplots[(r, c)] = subplot
            else:
                subax.set_visible(False)
            if r != self.rows-1:
                r += 1
            else:
                r = 0
                c += 1
        if create_axes:
            self.handles['axis'] = self._layout_axis(layout, axis)
            self._adjust_subplots(self.handles['axis'], subaxes)

        return subplots, subaxes, collapsed_layout


    def initialize_plot(self, ranges=None):
        # Get the extent of the layout elements (not the whole layout)
        key = self.keys[-1]
        axis = self.handles['axis']
        subplot_kwargs = dict()
        ranges = self.compute_ranges(self.layout, key, ranges)
        for subplot in self.subplots.values():
            subplot.initialize_plot(ranges=ranges, **subplot_kwargs)

        if self.show_title:
            title = axis.set_title(self._format_title(key),
                                   **self._fontsize('title'))
            self.handles['title'] = title

        self._readjust_axes(axis)
        self.drawn = True
        if self.subplot: return self.handles['axis']
        if self._close_figures: plt.close(self.handles['fig'])
        return self.handles['fig']


    def _readjust_axes(self, axis):
        if self.subplot:
            axis.set_position(self.position)
            if self.aspect == 'equal':
                axis.set_aspect(float(self.rows)/self.cols)
            self.handles['fig'].canvas.draw()
            self._adjust_subplots(self.handles['axis'], self.subaxes)


    def update_handles(self, axis, view, key, ranges=None):
        """
        Should be called by the update_frame class to update
        any handles on the plot.
        """
        if self.show_title:
            title = axis.set_title(self._format_title(key),
                                   **self._fontsize('title'))
            self.handles['title'] = title


    def _layout_axis(self, layout, axis):
        fig = self.handles['fig']
        axkwargs = {'gid': str(self.position)} if axis else {}
        layout_axis = fig.add_subplot(1,1,1, **axkwargs)

        if axis:
            axis.set_visible(False)
            layout_axis.set_position(self.position)
        layout_axis.patch.set_visible(False)

        tick_fontsize = self._fontsize('ticks','labelsize',common=False)
        if tick_fontsize: layout_axis.tick_params(**tick_fontsize)

        # Set labels
        layout_axis.set_xlabel(str(layout.kdims[0]),
                               **self._fontsize('xlabel'))
        if layout.ndims == 2:
            layout_axis.set_ylabel(str(layout.kdims[1]),
                               **self._fontsize('ylabel'))

        # Compute and set x- and y-ticks
        dims = layout.kdims
        keys = layout.keys()
        if layout.ndims == 1:
            dim1_keys = keys
            dim2_keys = [0]
            layout_axis.get_yaxis().set_visible(False)
        else:
            dim1_keys, dim2_keys = zip(*keys)
            layout_axis.set_ylabel(str(dims[1]))
            layout_axis.set_aspect(float(self.rows)/self.cols)

        # Process ticks
        plot_width = (1.0 - self.padding) / self.cols
        border_width = self.padding / (self.cols-1)
        xticks = [(plot_width/2)+(r*(plot_width+border_width)) for r in range(self.cols)]
        plot_height = (1.0 - self.padding) / self.rows
        border_height = self.padding / (self.rows-1) if layout.ndims > 1 else 0
        yticks = [(plot_height/2)+(r*(plot_height+border_height)) for r in range(self.rows)]

        layout_axis.set_xticks(xticks)
        layout_axis.set_xticklabels(self._process_ticklabels(sorted(set(dim1_keys)), dims[0]))
        for tick in layout_axis.get_xticklabels():
            tick.set_rotation(self.xrotation)

        ydim = dims[1] if layout.ndims > 1 else None
        layout_axis.set_yticks(yticks)
        layout_axis.set_yticklabels(self._process_ticklabels(sorted(set(dim2_keys)), ydim))
        for tick in layout_axis.get_yticklabels():
            tick.set_rotation(self.yrotation)

        if not self.show_frame:
            layout_axis.spines['right' if self.yaxis == 'left' else 'left'].set_visible(False)
            layout_axis.spines['bottom' if self.xaxis == 'top' else 'top'].set_visible(False)

        axis = layout_axis
        if self.xaxis is not None:
            axis.xaxis.set_ticks_position(self.xaxis)
            axis.xaxis.set_label_position(self.xaxis)
        else:
            axis.xaxis.set_visible(False)

        if self.yaxis is not None:
            axis.yaxis.set_ticks_position(self.yaxis)
            axis.yaxis.set_label_position(self.yaxis)
        else:
            axis.yaxis.set_visible(False)

        for pos in ['left', 'right', 'top', 'bottom']:
            axis.spines[pos].set_visible(False)

        return layout_axis


    def _process_ticklabels(self, labels, dim):
        formatted_labels = []
        for k in labels:
            if dim and dim.value_format:
                k = dim.value_format(k)
            elif not isinstance(k, (str, type(None))):
                k = self.tick_format % k
            elif k is None:
                k = ''
            formatted_labels.append(k)
        return formatted_labels


    def _adjust_subplots(self, axis, subaxes):
        bbox = axis.get_position()
        l, b, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height

        if self.padding:
            width_padding = w/(1./self.padding)
            height_padding = h/(1./self.padding)
        else:
            width_padding, height_padding = 0, 0

        if self.cols == 1:
            b_w = 0
        else:
            b_w = width_padding / (self.cols - 1)

        if self.rows == 1:
            b_h = 0
        else:
            b_h = height_padding / (self.rows - 1)
        ax_w = (w - (width_padding if self.cols > 1 else 0)) / self.cols
        ax_h = (h - (height_padding if self.rows > 1 else 0)) / self.rows

        r, c = (0, 0)
        for ax in subaxes.values():
            xpos = l + (c*ax_w) + (c * b_w)
            ypos = b + (r*ax_h) + (r * b_h)
            if r != self.rows-1:
                r += 1
            else:
                r = 0
                c += 1
            if not ax is None:
                ax.set_position([xpos, ypos, ax_w, ax_h])



class AdjointLayoutPlot(CompositePlot):
    """
    LayoutPlot allows placing up to three Views in a number of
    predefined and fixed layouts, which are defined by the layout_dict
    class attribute. This allows placing subviews next to a main plot
    in either a 'top' or 'right' position.

    Initially, a LayoutPlot computes an appropriate layout based for
    the number of Views in the AdjointLayout object it has been given, but
    when embedded in a NdLayout, it can recompute the layout to
    match the number of rows and columns as part of a larger grid.
    """

    layout_dict = {'Single': ['main'],
                   'Dual': ['main', 'right'],
                   'Triple': ['top', None, 'main', 'right'],
                   'Embedded Dual': [None, 'main']}

    def __init__(self, layout, layout_type, subaxes, subplots, **params):
        # The AdjointLayout ViewableElement object
        self.layout = layout
        # Type may be set to 'Embedded Dual' by a call it grid_situate
        self.layout_type = layout_type
        self.view_positions = self.layout_dict[self.layout_type]

        # The supplied (axes, view) objects as indexed by position
        self.subaxes = {pos: ax for ax, pos in zip(subaxes, self.view_positions)}
        super(AdjointLayoutPlot, self).__init__(subplots=subplots, **params)


    def initialize_plot(self, ranges=None):
        """
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        for pos in self.view_positions:
            # Pos will be one of 'main', 'top' or 'right' or None
            view = self.layout.get(pos, None)
            subplot = self.subplots.get(pos, None)
            ax = self.subaxes.get(pos, None)
            # If no view object or empty position, disable the axis
            if None in [view, pos, subplot]:
                ax.set_axis_off()
                continue
            subplot.initialize_plot(ranges=ranges)

        self.adjust_positions()
        self.drawn = True


    def adjust_positions(self):
        """
        Make adjustments to the positions of subplots (if available)
        relative to the main plot axes as required.

        This method is called by LayoutPlot after an initial pass
        used to position all the Layouts together. This method allows
        LayoutPlots to make final adjustments to the axis positions.
        """
        checks = [self.view_positions, self.subaxes, self.subplots]
        right = all('right' in check for check in checks)
        top = all('top' in check for check in checks)
        if not 'main' in self.subplots or not (top or right):
            return
        self.handles['fig'].canvas.draw()
        main_ax = self.subplots['main'].handles['axis']
        bbox = main_ax.get_position()
        if right:
            ax = self.subaxes['right']
            subplot = self.subplots['right']
            ax.set_position([bbox.x1 + bbox.width * subplot.border_size,
                             bbox.y0,
                             bbox.width * subplot.subplot_size, bbox.height])
            if isinstance(subplot, GridPlot):
                ax.set_aspect('equal')
        if top:
            ax = self.subaxes['top']
            subplot = self.subplots['top']
            ax.set_position([bbox.x0,
                             bbox.y1 + bbox.height * subplot.border_size,
                             bbox.width, bbox.height * subplot.subplot_size])
            if isinstance(subplot, GridPlot):
                ax.set_aspect('equal')


    def update_frame(self, key, ranges=None):
        for pos in self.view_positions:
            subplot = self.subplots.get(pos)
            if subplot is not None:
                subplot.update_frame(key, ranges)


    def __len__(self):
        return max([1 if self.keys is None else len(self.keys), 1])


class LayoutPlot(GenericLayoutPlot, CompositePlot):
    """
    A LayoutPlot accepts either a Layout or a NdLayout and
    displays the elements in a cartesian grid in scanline order.
    """

    aspect_weight = param.Number(default=0, doc="""
      Weighting of the individual aspects when computing the Layout
      grid aspects and overall figure size.""")

    fig_bounds = param.NumericTuple(default=(0.05, 0.05, 0.95, 0.95), doc="""
      The bounds of the figure as a 4-tuple of the form
      (left, bottom, right, top), defining the size of the border
      around the subplots.""")

    tight = param.Boolean(default=False, doc="""
      Tightly fit the axes in the layout within the fig_bounds
      and tight_padding.""")

    tight_padding = param.Parameter(default=3, doc="""
      Integer or tuple specifying the padding in inches in a tight layout.""")

    hspace = param.Number(default=0.5, doc="""
      Specifies the space between horizontally adjacent elements in the grid.
      Default value is set conservatively to avoid overlap of subplots.""")

    vspace = param.Number(default=0.1, doc="""
      Specifies the space between vertically adjacent elements in the grid.
      Default value is set conservatively to avoid overlap of subplots.""")

    fontsize = param.Parameter(default={'title':16}, allow_None=True)

    def __init__(self, layout, **params):
        super(LayoutPlot, self).__init__(layout=layout, **params)
        self.subplots, self.subaxes, self.layout = self._compute_gridspec(layout)


    def _compute_gridspec(self, layout):
        """
        Computes the tallest and widest cell for each row and column
        by examining the Layouts in the GridSpace. The GridSpec is then
        instantiated and the LayoutPlots are configured with the
        appropriate embedded layout_types. The first element of the
        returned tuple is a dictionary of all the LayoutPlots indexed
        by row and column. The second dictionary in the tuple supplies
        the grid indicies needed to instantiate the axes for each
        LayoutPlot.
        """
        layout_items = layout.grid_items()
        layout_dimensions = layout.kdims if isinstance(layout, NdLayout) else None

        layouts = {}
        row_heightratios, col_widthratios = {}, {}
        col_aspects, row_aspects = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for (r, c) in self.coords:
            # Get view at layout position and wrap in AdjointLayout
            _, view = layout_items.get((r, c), (None, None))
            layout_view = view if isinstance(view, AdjointLayout) else AdjointLayout([view])
            layouts[(r, c)] = layout_view

            # Compute shape of AdjointLayout element
            layout_lens = {1:'Single', 2:'Dual', 3:'Triple'}
            layout_type = layout_lens[len(layout_view)]

            # Get aspects
            main = layout_view.main
            main = main.last if isinstance(main, HoloMap) else main
            main_options = self.lookup_options(main, 'plot').options if main else {}
            if main and not isinstance(main_options.get('aspect', 1), basestring):
                main_aspect = main_options.get('aspect', 1)
                main_aspect = self.aspect_weight*main_aspect + 1-self.aspect_weight
            else:
                main_aspect = 1

            if layout_type in ['Dual', 'Triple']:
                el = layout_view.get('right', None)
                eltype = type(el)
                if el and eltype in MPLPlot.sideplots:
                    plot_type = MPLPlot.sideplots[type(el)]
                    ratio = plot_type.border_size + plot_type.subplot_size
                    width_ratios = [4, 4*ratio]
                else:
                    width_ratios = [4, 1]
                col_aspect = [main_aspect, 1/(4/width_ratios[1])]
            else:
                width_ratios = [4]
                col_aspect = [main_aspect, 0]

            if layout_type in ['Embedded Dual', 'Triple']:
                el = layout_view.get('top', None)
                eltype = type(el)
                if el and eltype in MPLPlot.sideplots:
                    plot_type = MPLPlot.sideplots[type(el)]
                    ratio = plot_type.border_size + plot_type.subplot_size
                    height_ratios = [4*ratio, 4]
                else:
                    height_ratios = [1, 4]
                row_aspect = [1/(4/height_ratios[0]), 1./main_aspect]
                hidx = 1
            else:
                height_ratios = [4]
                row_aspect = [0, 1./main_aspect]
                hidx = 0

            if not isinstance(main_aspect, (basestring, type(None))):
                width_ratios[0] = (width_ratios[0] * main_aspect)
                height_ratios[-1] = (height_ratios[-1] * 1./main_aspect)
            layout_shape = (len(width_ratios), len(height_ratios))

            # For each row and column record the width and height ratios
            # of the LayoutPlot with the most horizontal or vertical splits
            # and largest aspect
            if layout_shape[1] > row_heightratios.get(r, (0, None))[0]:
                row_heightratios[r] = [layout_shape[1], height_ratios]
            if height_ratios[hidx] > row_heightratios[r][1][hidx]:
                row_heightratios[r][1][-1] = height_ratios[hidx]

            if layout_shape[0] > col_widthratios.get(c, (0, None))[0]:
                col_widthratios[c] = (layout_shape[0], width_ratios)
            if width_ratios[0] > col_widthratios[c][1][0]:
                col_widthratios[c][1][0] = width_ratios[0]

            for i in range(2):
                if col_aspect[i] > col_aspects.get(c, [0,0])[i]:
                    col_aspects[c][i] = col_aspect[i]
                if row_aspect[i] > row_aspects.get(r, [0,0])[i]:
                    row_aspects[r][i] = row_aspect[i]

        # In order of row/column collect the largest width and height ratios
        height_ratios = [v[1] for k, v in sorted(row_heightratios.items())]
        width_ratios = [v[1] for k, v in sorted(col_widthratios.items())]
        col_aspect_ratios = [v for k, v in sorted(col_aspects.items())]
        row_aspect_ratios = [v for k, v in sorted(row_aspects.items())]

        # Compute the number of rows and cols
        cols = np.sum([len(wr) for wr in width_ratios])
        rows = np.sum([len(hr) for hr in height_ratios])

        # Flatten the width and height ratio lists
        wr_list = [wr for wrs in width_ratios for wr in wrs]
        hr_list = [hr for hrs in height_ratios for hr in hrs]

        # Compute and set the plot size if not explicitly supplied
        col_ars = [ar for ars in col_aspect_ratios for ar in ars]
        row_ars = [ar for ars in row_aspect_ratios for ar in ars]
        width = len(col_ars[::2]) + sum(col_ars[1::2])
        yscale = sum(col_ars)/sum(row_ars)
        xinches, yinches = None, None
        if not isinstance(self.fig_inches, (tuple, list)):
            xinches = self.fig_inches * width
            yinches = xinches/yscale
        elif self.fig_inches[0] is None:
            xinches = self.fig_inches[1] * yscale
            yinches = self.fig_inches[1]
        elif self.fig_inches[1] is None:
            xinches = self.fig_inches[0]
            yinches = self.fig_inches[0] / yscale
        if xinches and yinches:
            self.handles['fig'].set_size_inches([xinches, yinches])

        self.gs = gridspec.GridSpec(rows, cols,
                                    width_ratios=wr_list,
                                    height_ratios=hr_list,
                                    wspace=self.hspace,
                                    hspace=self.vspace)

        # Situate all the Layouts in the grid and compute the gridspec
        # indices for all the axes required by each LayoutPlot.
        gidx = 0
        layout_count = 0
        tight = self.tight
        collapsed_layout = layout.clone(shared_data=False, id=layout.id)
        frame_ranges = self.compute_ranges(layout, None, None)
        frame_ranges = OrderedDict([(key, self.compute_ranges(layout, key, frame_ranges))
                                    for key in self.keys])
        layout_subplots, layout_axes = {}, {}
        for r, c in self.coords:
            # Compute the layout type from shape
            wsplits = len(width_ratios[c])
            hsplits = len(height_ratios[r])
            if (wsplits, hsplits) == (1,1):
                layout_type = 'Single'
            elif (wsplits, hsplits) == (2,1):
                layout_type = 'Dual'
            elif (wsplits, hsplits) == (1,2):
                layout_type = 'Embedded Dual'
            elif (wsplits, hsplits) == (2,2):
                layout_type = 'Triple'

            # Get the AdjoinLayout at the specified coordinate
            view = layouts[(r, c)]
            positions = AdjointLayoutPlot.layout_dict[layout_type]

            # Create temporary subplots to get projections types
            # to create the correct subaxes for all plots in the layout
            _, _, projs = self._create_subplots(layouts[(r, c)], positions,
                                                None, frame_ranges, create=False)
            gidx, gsinds = self.grid_situate(gidx, layout_type, cols)

            layout_key, _ = layout_items.get((r, c), (None, None))
            if isinstance(layout, NdLayout) and layout_key:
                layout_dimensions = OrderedDict(zip(layout_dimensions, layout_key))

            # Generate the axes and create the subplots with the appropriate
            # axis objects, handling any Empty objects.
            obj = layouts[(r, c)]
            empty = isinstance(obj.main, Empty)
            if empty:
                obj = AdjointLayout([])
            else:
                layout_count += 1
            subaxes = [plt.subplot(self.gs[ind], projection=proj)
                       for ind, proj in zip(gsinds, projs)]
            subplot_data = self._create_subplots(obj, positions,
                                                 layout_dimensions, frame_ranges,
                                                 dict(zip(positions, subaxes)),
                                                 num=0 if empty else layout_count)
            subplots, adjoint_layout, _ = subplot_data
            layout_axes[(r, c)] = subaxes

            # Generate the AdjointLayoutsPlot which will coordinate
            # plotting of AdjointLayouts in the larger grid
            plotopts = self.lookup_options(view, 'plot').options
            layout_plot = AdjointLayoutPlot(adjoint_layout, layout_type, subaxes, subplots,
                                            fig=self.handles['fig'], **plotopts)
            layout_subplots[(r, c)] = layout_plot
            tight = not any(type(p) is GridPlot for p in layout_plot.subplots.values()) and tight
            if layout_key:
                collapsed_layout[layout_key] = adjoint_layout

        # Apply tight layout if enabled and incompatible
        # GridPlot isn't present.
        if tight:
            if isinstance(self.tight_padding, (tuple, list)):
                wpad, hpad = self.tight_padding
                padding = dict(w_pad=wpad, h_pad=hpad)
            else:
                padding = dict(w_pad=self.tight_padding, h_pad=self.tight_padding)
            self.gs.tight_layout(self.handles['fig'], rect=self.fig_bounds, **padding)

        # Create title handle
        if self.show_title and len(self.coords) > 1:
            title = self.handles['fig'].suptitle('', **self._fontsize('title'))
            self.handles['title'] = title

        return layout_subplots, layout_axes, collapsed_layout


    def grid_situate(self, current_idx, layout_type, subgrid_width):
        """
        Situate the current AdjointLayoutPlot in a LayoutPlot. The
        LayoutPlot specifies a layout_type into which the AdjointLayoutPlot
        must be embedded. This enclosing layout is guaranteed to have
        enough cells to display all the views.

        Based on this enforced layout format, a starting index
        supplied by LayoutPlot (indexing into a large gridspec
        arrangement) is updated to the appropriate embedded value. It
        will also return a list of gridspec indices associated with
        the all the required layout axes.
        """
        # Set the layout configuration as situated in a NdLayout

        if layout_type == 'Single':
            start, inds = current_idx+1, [current_idx]
        elif layout_type == 'Dual':
            start, inds = current_idx+2, [current_idx, current_idx+1]

        bottom_idx = current_idx + subgrid_width
        if layout_type == 'Embedded Dual':
            bottom = ((current_idx+1) % subgrid_width) == 0
            grid_idx = (bottom_idx if bottom else current_idx)+1
            start, inds = grid_idx, [current_idx, bottom_idx]
        elif layout_type == 'Triple':
            bottom = ((current_idx+2) % subgrid_width) == 0
            grid_idx = (bottom_idx if bottom else current_idx) + 2
            start, inds = grid_idx, [current_idx, current_idx+1,
                              bottom_idx, bottom_idx+1]

        return start, inds


    def _create_subplots(self, layout, positions, layout_dimensions, ranges, axes={}, num=1, create=True):
        """
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        subplots = {}
        projections = []
        adjoint_clone = layout.clone(shared_data=False, id=layout.id)
        subplot_opts = dict(show_title=False, adjoined=layout)
        for pos in positions:
            # Pos will be one of 'main', 'top' or 'right' or None
            view = layout.get(pos, None)
            ax = axes.get(pos, None)
            if view is None:
                projections.append(None)
                continue

            # Determine projection type for plot
            components = view.traverse(lambda x: x)
            projs = ['3d' if isinstance(c, Element3D) else
                     self.lookup_options(c, 'plot').options.get('projection', None)
                     for c in components]
            projs = [p for p in projs if p is not None]
            if len(set(projs)) > 1:
                raise Exception("A single axis may only be assigned one projection type")
            elif projs:
                projections.append(projs[0])
            else:
                projections.append(None)

            if not create:
                continue

            # Customize plotopts depending on position.
            plotopts = self.lookup_options(view, 'plot').options

            # Options common for any subplot
            override_opts = {}
            sublabel_opts = {}
            if pos == 'main':
                own_params = self.get_param_values(onlychanged=True)
                sublabel_opts = {k: v for k, v in own_params
                                 if 'sublabel_' in k}
                if not isinstance(view, GridSpace):
                    override_opts = dict(aspect='square')
            elif pos == 'right':
                right_opts = dict(invert_axes=True,
                                  xaxis=None)
                override_opts = dict(subplot_opts, **right_opts)
            elif pos == 'top':
                top_opts = dict(yaxis=None)
                override_opts = dict(subplot_opts, **top_opts)

            # Override the plotopts as required
            plotopts = dict(sublabel_opts, **plotopts)
            plotopts.update(override_opts, fig=self.handles['fig'])
            vtype = view.type if isinstance(view, HoloMap) else view.__class__
            if isinstance(view, GridSpace):
                plotopts['create_axes'] = ax is not None
            if pos == 'main':
                plot_type = Store.registry['matplotlib'][vtype]
            else:
                plot_type = MPLPlot.sideplots[vtype]
            num = num if len(self.coords) > 1 else 0
            subplots[pos] = plot_type(view, axis=ax, keys=self.keys,
                                      dimensions=self.dimensions,
                                      layout_dimensions=layout_dimensions,
                                      ranges=ranges, subplot=True,
                                      uniform=self.uniform, layout_num=num,
                                      **plotopts)
            if isinstance(view, (Element, HoloMap, Collator, CompositeOverlay)):
                adjoint_clone[pos] = subplots[pos].hmap
            else:
                adjoint_clone[pos] = subplots[pos].layout
        return subplots, adjoint_clone, projections


    def update_handles(self, axis, view, key, ranges=None):
        """
        Should be called by the update_frame class to update
        any handles on the plot.
        """
        if self.show_title and 'title' in self.handles and len(self.coords) > 1:
            self.handles['title'].set_text(self._format_title(key))


    def initialize_plot(self):
        axis = self.handles['axis']
        self.update_handles(axis, None, self.keys[-1])

        ranges = self.compute_ranges(self.layout, self.keys[-1], None)
        for subplot in self.subplots.values():
            subplot.initialize_plot(ranges=ranges)

        return self._finalize_axis(None)



class AdjoinedPlot(DimensionedPlot):

    aspect = param.Parameter(default='auto', doc="""
        Aspect ratios on SideHistogramPlot should be determined by the
        AdjointLayoutPlot.""")

    bgcolor = param.Parameter(default=(1, 1, 1, 0), doc="""
        Make plot background invisible.""")

    border_size = param.Number(default=0.25, doc="""
        The size of the border expressed as a fraction of the main plot.""")

    show_frame = param.Boolean(default=False)

    show_title = param.Boolean(default=False, doc="""
        Titles should be disabled on all SidePlots to avoid clutter.""")

    subplot_size = param.Number(default=0.25, doc="""
        The size subplots as expressed as a fraction of the main plot.""")

    show_xlabel = param.Boolean(default=False, doc="""
        Whether to show the x-label of the plot. Disabled by default
        because plots are often too cramped to fit the title correctly.""")
