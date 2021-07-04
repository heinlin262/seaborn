from __future__ import annotations

import io
import itertools
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib as mpl

from seaborn._core.rules import categorical_order, variable_type
from seaborn._core.data import PlotData
from seaborn._core.mappings import GroupMapping, HueMapping
from seaborn._core.scales import (
    ScaleWrapper,
    CategoricalScale,
    DatetimeScale,
    norm_from_scale
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Any
    from collections.abc import Callable, Generator, Iterable, Hashable
    from pandas import DataFrame, Series, Index
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.scale import ScaleBase
    from matplotlib.colors import Normalize
    from seaborn._core.mappings import SemanticMapping
    from seaborn._marks.base import Mark
    from seaborn._stats.base import Stat
    from seaborn._core.typing import DataSource, PaletteSpec, VariableSpec, OrderSpec


class Plot:

    _data: PlotData
    _layers: list[Layer]
    _mappings: dict[str, SemanticMapping]  # TODO keys as Literal, or use TypedDict?
    _scales: dict[str, ScaleBase]

    _facetspec: dict[str, Any]  # TODO any need to be more strict on values?
    _pairspec: dict[str, Any]

    _figure: Figure

    def __init__(
        self,
        data: DataSource = None,
        **variables: VariableSpec,
    ):

        self._data = PlotData(data, variables)
        self._layers = []

        # TODO see notes in _setup_mappings I think we're going to start with this
        # empty and define the defaults elsewhere
        self._mappings = {
            "group": GroupMapping(),
            "hue": HueMapping(),
        }

        # TODO is using "unknown" here the best approach?
        # Other options would be:
        # - None as the value for type
        # - some sort of uninitialized singleton for the object,
        self._scales = {
            "x": ScaleWrapper(mpl.scale.LinearScale("x"), "unknown"),
            "y": ScaleWrapper(mpl.scale.LinearScale("y"), "unknown"),
        }

        self._facetspec = {}
        self._pairspec = {}

    def on(self) -> Plot:

        # TODO  Provisional name for a method that accepts an existing Axes object,
        # and possibly one that does all of the figure/subplot configuration

        # We should also accept an existing figure object. This will be most useful
        # in cases where users have created a *sub*figure ... it will let them facet
        # etc. within an existing, larger figure. We still have the issue with putting
        # the legend outside of the plot and that potentially causing problems for that
        # larger figure. Not sure what to do about that. I suppose existing figure could
        # disabling legend_out.

        raise NotImplementedError()
        return self

    def add(
        self,
        mark: Mark,
        stat: Stat | None = None,
        orient: Literal["x", "y", "v", "h"] = "x",  # TODO "auto" as defined by Mark?
        data: DataSource = None,
        **variables: VariableSpec,
    ) -> Plot:

        if stat is None and mark.default_stat is not None:
            # TODO We need some way to say "do no stat transformation" that is different
            # from "use the default". That's basically an IdentityStat.

            # Default stat needs to be initialized here so that its state is
            # not modified across multiple plots. If a Mark wants to define a default
            # stat with non-default params, it should use functools.partial
            stat = mark.default_stat()

        orient_map = {"v": "x", "h": "y"}
        orient = orient_map.get(orient, orient)  # type: ignore  # mypy false positive?
        mark.orient = orient  # type: ignore  # mypy false positive?
        if stat is not None:
            stat.orient = orient  # type: ignore  # mypy false positive?

        self._layers.append(Layer(mark, stat, data, variables))

        return self

    def pair(
        self,
        x: Hashable | list[Hashable] | None = None,
        y: Hashable | list[Hashable] | None = None,
        cartesian: bool = True,  # TODO bikeshed name, maybe cross?
        # TODO wrapping if only one variable is a list or not cartesian
        # TODO figure parameterization (sharex/sharey, etc.)
        # TODO other existing PairGrid things like corner?
    ) -> Plot:

        # TODO Basic idea is to implement PairGrid functionality within this interface
        # But want to be even more powerful in a few ways:
        # - combined pairing and faceting
        #   - need to decide whether rows/cols are either facets OR pairs,
        #     or if they can be composed (feasible, but more complicated)
        # - "non-product" (need a name) pairing, i.e. for len(x) == len(y) == n,
        #   make n subplots with x[0] v y[0], x[1] v y[1], etc.
        # - uni-dimensional pairing
        #   - i.e. if only x or y is assigned, to support a grid of histograms, etc.

        # Problems to solve:
        # - How to get a default square grid of all x vs all y? If x and y are None,
        #   use all variables in self._data (dropping those used for semantic mapping?)
        #   What if we want to specify the subset of variables to use for a square grid,
        #   is it necessary to specify `x=cols, y=cols`?
        # - Unclear is how to handle the diagonal plots that PairGrid offers
        # - Implementing this will require lots of downscale changes in figure setup,
        #   and especially the axis scaling, which will need to be pair specific
        # - How to resolve sharex/sharey between facet() and pair()?

        # Do we only want to allow variable keys in the x/y list? Can we have lists of
        # vectors, or is that too complicated?

        # TODO raise if pair called without source data? or add data= arg here?
        data = self._data._source_data

        # TODO default to columns of source data if x and y are both None
        # TODO (BUT we want to not use columns assigned to mappings in that case...)

        pairspec = {}
        axes = {"x": x, "y": y}
        for axis, arg in axes.items():
            # TODO more input validation
            if arg is not None:

                # Canonical way of testing hashability broken on pandas < 1.3
                # hashable_arg = isinstance(arg, abc.Hashable)
                # https://github.com/pandas-dev/pandas/pull/41283
                try:
                    hash(arg)
                    hashable_arg = True
                except TypeError:
                    hashable_arg = False

                if hashable_arg and arg in data:
                    # TODO this fails in a weird way if string is spelled wrong, etc.
                    arg = [arg]
                pairspec[axis] = list(arg)

        pairspec["variables"] = {}
        for axis in "xy":
            for i, col in enumerate(pairspec.get(axis, [])):
                pairspec["variables"][f"{axis}{i}"] = col

        pairspec["cartesian"] = cartesian

        self._pairspec.update(pairspec)
        return self

    def facet(
        self,
        col: VariableSpec = None,
        row: VariableSpec = None,
        col_order: OrderSpec = None,
        row_order: OrderSpec = None,
        wrap: int | None = None,
        data: DataSource = None,
        sharex: bool | Literal["row", "col"] = True,
        sharey: bool | Literal["row", "col"] = True,
        # TODO or sharexy: bool | Literal | tuple[bool | Literal]?
    ) -> Plot:

        # Note: can't pass `None` here or it will uninherit the `Plot()` def
        variables = {}
        if col is not None:
            variables["col"] = col
        if row is not None:
            variables["row"] = row

        # TODO raise here if col/row not defined here or in self._data?

        # TODO Alternately use the following parameterization for order
        # `order: list[Hashable] | dict[Literal['col', 'row'], list[Hashable]]
        # this is more convenient for the (dominant?) case where there is one
        # faceting variable

        # TODO Basic faceting functionality is tested, but there aren't tests
        # for all the permutations of this interface

        self._facetspec.update({
            "source": data,
            "variables": variables,
            "col_order": None if col_order is None else list(col_order),
            "row_order": None if row_order is None else list(row_order),
            "wrap": wrap,
            "sharex": sharex,
            "sharey": sharey
        })

        return self

    def map_hue(
        self,
        palette: PaletteSpec = None,
    ) -> Plot:

        # TODO we do some fancy business currently to avoid having to
        # write these ... do we want that to persist or is it too confusing?
        # ALSO TODO should these be initialized with defaults?
        self._mappings["hue"] = HueMapping(palette)
        return self

    # TODO originally we had planned to have a scale_native option that would default
    # to matplotlib. I don't fully remember why. Is this still something we need?

    # TODO related, scale_identity which uses the data as literal attribute values

    def scale_numeric(
        self,
        var: str,
        scale: str | ScaleBase = "linear",
        norm: tuple[float | None, float | None] | Normalize | None = None,
        **kwargs
    ) -> Plot:

        # TODO XXX FIXME matplotlib scales sometimes default to
        # filling invalid outputs with large out of scale numbers
        # (e.g. default behavior for LogScale is 0 -> -10000)
        # This will cause MAJOR PROBLEMS for statistical transformations
        # Solution? I think it's fine to special-case scale="log" in
        # Plot.scale_numeric and force `nonpositive="mask"` and remove
        # NAs after scaling (cf GH2454).
        # And then add a warning in the docstring that the users must
        # ensure that ScaleBase derivatives mask out of bounds data

        # TODO use norm for setting axis limits? Or otherwise share an interface?

        # TODO or separate norm as a Normalize object and limits as a tuple?
        # (If we have one we can create the other)

        # TODO expose parameter for internal dtype achieved during scale.cast?

        # TODO we want to be able to call this on numbers-as-strings data and
        # have it work the way you would expect.

        if isinstance(scale, str):
            scale = mpl.scale.scale_factory(scale, var, **kwargs)

        if norm is None:
            # TODO what about when we want to infer the scale from the norm?
            # e.g. currently you pass LogNorm to get a log normalization...
            norm = norm_from_scale(scale, norm)
        self._scales[var] = ScaleWrapper(scale, "numeric", norm=norm)
        return self

    def scale_categorical(
        self,
        var: str,
        order: Series | Index | Iterable | None = None,
        formatter: Callable | None = None,
    ) -> Plot:

        # TODO how to set limits/margins "nicely"? (i.e. 0.5 data units, past extremes)
        # TODO similarly, should this modify grid state like current categorical plots?
        # TODO "smart"/data-dependant ordering (e.g. order by median of y variable)

        if order is not None:
            order = list(order)

        scale = CategoricalScale(var, order, formatter)
        self._scales[var] = ScaleWrapper(scale, "categorical")
        return self

    def scale_datetime(self, var) -> Plot:

        scale = DatetimeScale(var)
        self._scales[var] = ScaleWrapper(scale, "datetime")

        # TODO what else should this do?
        # We should pass kwargs to the DateTime cast probably.
        # It will be nice to have more control over the formatting of the ticks
        # which is pretty annoying in standard matplotlib.
        # Should datetime data ever have anything other than a linear scale?
        # The only thing I can really think of are geologic/astro plots that
        # use a reverse log scale.

        return self

    def theme(self) -> Plot:

        # TODO Plot-specific themes using the seaborn theming system
        # TODO should this also be where custom figure size goes?
        raise NotImplementedError()
        return self

    def resize(self, val):

        # TODO I don't think this is the interface we ultimately want to use, but
        # I want to be able to do this for demonstration now. If we do want this
        # could think about how to have "auto" sizing based on number of subplots
        self._figsize = val
        return self

    def plot(self, pyplot=False) -> Plot:

        # TODO clone self here, so plot() doesn't modify the original objects?
        # (Do the clone here, or do it in show/_repr_png_?)

        self._setup_layers()
        self._setup_scales()
        self._setup_mappings()
        self._setup_figure(pyplot)

        # Abort early if we've just set up a blank figure
        if not self._layers:
            return self

        for layer in self._layers:

            layer_mappings = {k: v for k, v in self._mappings.items() if k in layer}
            self._plot_layer(layer, layer_mappings)

        # TODO this should be configurable
        self._figure.tight_layout()

        return self

    def clone(self) -> Plot:

        if hasattr(self, "_figure"):
            raise RuntimeError("Cannot clone object after calling Plot.plot")
        return deepcopy(self)

    def show(self, **kwargs) -> None:

        # Keep an eye on whether matplotlib implements "attaching" an existing
        # figure to pyplot: https://github.com/matplotlib/matplotlib/pull/14024
        self.clone().plot(pyplot=True)

        import matplotlib.pyplot as plt
        plt.show(**kwargs)

    def save(self) -> Plot:  # TODO perhaps this should not return self?

        raise NotImplementedError()
        return self

    # ================================================================================ #
    # End of public API
    # ================================================================================ #

    def _setup_layers(self):

        common_data = (
            self._data
            .concat(
                self._facetspec.get("source"),
                self._facetspec.get("variables"),
            )
            .concat(
                self._pairspec.get("source"),
                self._pairspec.get("variables"),
            )
        )

        # TODO concat with mapping spec

        for layer in self._layers:
            layer.data = common_data.concat(layer.source, layer.variables)

    def _setup_scales(self) -> None:

        # TODO We need to make sure that when using the "pair" functionality, the
        # scaling is pair-variable dependent. We can continue to use the same scale
        # (though not necessarily the same limits, or the same categories) for faceting

        layers = self._layers
        for var, scale in self._scales.items():
            if scale.type == "unknown" and any(var in layer for layer in layers):
                # TODO this is copied from _setup_mappings ... ripe for abstraction!
                all_data = pd.concat(
                    [layer.data.frame.get(var) for layer in layers]
                ).reset_index(drop=True)
                scale.type = variable_type(all_data)

    def _setup_figure(self, pyplot: bool = False) -> None:

        # TODO add external API for parameterizing figure, (size , autolayout, etc.)
        # TODO use context manager with theme that has been set
        # TODO (maybe wrap THIS function with context manager; would be cleaner)

        setup_data = self._data.concat(
            self._facetspec.get("source"),
            self._facetspec.get("variables"),
        )

        # TODO Validate that we do not have overlapping pair/facet specification
        # i.e. we cannot pair on x and facet on columns

        setup_data = setup_data.concat(None, self._pairspec.get("variables", {}))

        # TODO Ignoring col/row wrapping, but we need to deal with that

        figure_dimensions = {}
        subplot_spec = {}
        for dim, axis in zip(["col", "row"], ["x", "y"]):
            if dim in setup_data:
                data = setup_data.frame[dim]
                figure_dimensions[dim] = categorical_order(
                    data, self._facetspec.get(f"{dim}_order"),
                )
            elif axis in self._pairspec:
                # TODO different behavior for cartesian
                # TODO pass original column names? Or internal variable names?
                figure_dimensions[dim] = self._pairspec[axis]
            else:
                figure_dimensions[dim] = [None]
            subplot_spec[f"n{dim}s"] = len(figure_dimensions[dim])

            # TODO Defaults for sharex/y should be defined in one place
            if axis in self._pairspec:
                subplot_spec[f"share{axis}"] = dim  # TODO allows unshared pair axes
            else:
                subplot_spec[f"share{axis}"] = self._facetspec.get(f"share{axis}", True)

        if not self._pairspec.get("cartesian", True):
            # TODO Won't work with wrapping, and overall very confusing.
            subplot_spec["nrows"] = 1
            for axis in "xy":
                # TODO we always want axis labels for noncartesian plots, but does it
                # make sense to allow shared axes? Maybe we should turn visibility of
                # labels back on if cartesian is False, and default to unshared limits,
                # but allow them somehow...
                subplot_spec[f"share{axis}"] = False

        figsize = getattr(self, "_figsize", None)

        if pyplot:
            import matplotlib.pyplot as plt
            self._figure = plt.figure(figsize=figsize)
        else:
            self._figure = mpl.figure.Figure(figsize=figsize)

        subplots = self._figure.subplots(**subplot_spec, squeeze=False)

        self._subplot_list = []

        if not self._pairspec or self._pairspec["cartesian"]:
            iterplots = np.ndenumerate(subplots)
        else:
            indices = np.arange(subplot_spec["ncols"])
            iterplots = zip(zip(indices, indices), subplots.flat)

        for (i, j), axes in iterplots:

            self._subplot_list.append({
                "axes": axes,
                "row": figure_dimensions["row"][i],
                "col": figure_dimensions["col"][j],
            })

            for axis in "xy":
                idx = {"x": j, "y": i}[axis]
                if axis in self._pairspec:
                    label = self._pairspec.get(axis)[idx]
                else:
                    label = self._data.names.get(axis)
                axes.set(**{
                    f"{axis}scale": self._scales[axis]._scale,
                    f"{axis}label": label,
                })

            # TODO need to account for wrap
            if subplot_spec["sharex"] in (True, "col") and subplots.shape[0] - i > 1:
                axes.xaxis.label.set_visible(False)
            if subplot_spec["sharey"] in (True, "row") and j > 0:
                axes.yaxis.label.set_visible(False)

            # TODO should titles be set for each position along the pair dimension?
            # (e.g., pair on y, facet on cols, should facet titles only go on top row?)
            title_parts = []
            for idx, dim in zip([i, j], ["row", "col"]):
                if dim in setup_data:
                    name = setup_data.names.get(dim, f"_{dim}_")
                    level = figure_dimensions[dim][idx]
                    title_parts.append(f"{name} = {level}")
            title = " | ".join(title_parts)
            axes.set_title(title)

    def _setup_mappings(self) -> None:

        layers = self._layers

        # TODO we should setup default mappings here based on whether a mapping
        # variable appears in at least one of the layer data but isn't in self._mappings
        # Source of what mappings to check can be some dictionary of default mappings?

        for var, mapping in self._mappings.items():
            if any(var in layer for layer in layers):
                all_data = pd.concat(
                    [layer.data.frame.get(var) for layer in layers]
                ).reset_index(drop=True)
                scale = self._scales.get(var)
                mapping.setup(all_data, scale)

    def _plot_layer(self, layer: Layer, mappings: dict[str, SemanticMapping]) -> None:

        default_grouping_vars = ["col", "row", "group"]  # TODO where best to define?

        data = layer.data
        mark = layer.mark
        stat = layer.stat

        df = self._scale_coords(data.frame)

        if stat is not None:
            grouping_vars = stat.grouping_vars + default_grouping_vars
            df = self._apply_stat(df, grouping_vars, stat)

        df = mark._adjust(df)

        # Our statistics happen on the scale we want, but then matplotlib is going
        # to re-handle the scaling, so we need to invert before handing off
        # Note: we don't need to convert back to strings for categories (but we could?)
        df = self._unscale_coords(df)

        # TODO this might make debugging annoying ... should we create new data object?
        data.frame = df

        grouping_vars = mark.grouping_vars + default_grouping_vars
        generate_splits = self._setup_split_generator(grouping_vars, data, mappings)

        layer.mark._plot(generate_splits, mappings)

    def _apply_stat(
        self, df: DataFrame, grouping_vars: list[str], stat: Stat
    ) -> DataFrame:

        stat.setup(df)

        # TODO how can we special-case fast aggregations? (i.e. mean, std, etc.)
        # IDEA: have Stat identify as an aggregator? (Through Mixin or attribute)
        # e.g. if stat.aggregates ...
        stat_grouping_vars = [var for var in grouping_vars if var in df]
        # TODO I don't think we always want to group by the default orient axis?
        # Better to have the Stat declare when it wants that to happen
        if stat.orient not in stat_grouping_vars:
            stat_grouping_vars.append(stat.orient)

        # TODO rewrite this whole thing, I think we just need to avoid groupby/apply
        df = (
            df
            .groupby(stat_grouping_vars)
            .apply(stat)
        )
        # TODO next because of https://github.com/pandas-dev/pandas/issues/34809
        for var in stat_grouping_vars:
            if var in df.index.names:
                df = (
                    df
                    .drop(var, axis=1, errors="ignore")
                    .reset_index(var)
                )
        df = df.reset_index(drop=True)  # TODO not always needed, can we limit?
        return df

    def _get_data_for_axes(self, df: DataFrame, subplot: dict) -> DataFrame:

        # TODO should handle pair logic here too, possibly assignment of x{n} -> x, etc
        keep = pd.Series(True, df.index)
        for dim, axis in zip(["col", "row"], ["x", "y"]):
            if axis in self._pairspec:
                idx = self._pairspec[axis].index(subplot[dim])
                df[axis] = df[f"{axis}{idx}"]
            if dim in df:
                keep &= df[dim] == subplot[dim]

        return df[keep]

    def _scale_coords(self, df: DataFrame) -> DataFrame:

        # TODO the regex in filter is handy but we don't actually use the DataFrame
        # we may want to explore a way of doing this that doesn't allocate a new df
        # TODO note that this will beed to be variable-specific for pairing
        coord_cols = df.filter(regex="(^x)|(^y)").columns
        out_df = (
            df
            .drop(coord_cols, axis=1)
            .copy(deep=False)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
        )

        for subplot in self._subplot_list:
            axes_df = self._get_data_for_axes(df, subplot)[coord_cols]
            with pd.option_context("mode.use_inf_as_null", True):
                axes_df = axes_df.dropna()
            self._scale_coords_single(axes_df, out_df, subplot["axes"])
        return out_df

    def _scale_coords_single(
        self, coord_df: DataFrame, out_df: DataFrame, axes: Axes
    ) -> None:

        # TODO modify out_df in place or return and handle externally?

        for var, data in coord_df.items():

            # TODO Explain the logic of this method thoroughly
            # It is clever, but a bit confusing!

            axis = var[0]
            axis_obj = getattr(axes, f"{axis}axis")
            scale = self._scales[axis]

            if scale.order is not None:
                data = data[data.isin(scale.order)]

            # TODO wrap this in a try/except and reraise with more information
            # about what variable caused the problem (and input / desired types)
            data = scale.cast(data)
            axis_obj.update_units(categorical_order(data))

            scaled = self._scales[axis].forward(axis_obj.convert_units(data))
            out_df.loc[data.index, var] = scaled

    def _unscale_coords(self, df: DataFrame) -> DataFrame:

        # TODO copied from _scale_coords
        coord_df = df.filter(regex="(^x)|(^y)")
        out_df = (
            df
            .drop(coord_df.columns, axis=1)
            .copy(deep=False)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
        )

        for var, col in coord_df.items():
            axis = var[0]
            out_df[var] = self._scales[axis].reverse(coord_df[var])

        return out_df

    def _setup_split_generator(
        self,
        grouping_vars: list[str],
        data: PlotData,
        mappings: dict[str, SemanticMapping],
    ) -> Callable[[], Generator]:

        allow_empty = False  # TODO will need to recreate previous categorical plots

        levels = {v: m.levels for v, m in mappings.items()}
        grouping_vars = [
            var for var in grouping_vars if var in data and var not in ["col", "row"]
        ]
        grouping_keys = [levels.get(var, []) for var in grouping_vars]

        def generate_splits() -> Generator:

            for subplot in self._subplot_list:

                axes_df = self._get_data_for_axes(data.frame, subplot)

                subplot_keys = {}
                for dim in ["col", "row"]:
                    if subplot[dim] is not None:
                        subplot_keys[dim] = subplot[dim]

                if not grouping_vars or not any(grouping_keys):
                    yield subplot_keys, axes_df.copy(), subplot["axes"]
                    continue

                grouped_df = axes_df.groupby(grouping_vars, sort=False, as_index=False)

                for key in itertools.product(*grouping_keys):

                    # Pandas fails with singleton tuple inputs
                    pd_key = key[0] if len(key) == 1 else key

                    try:
                        df_subset = grouped_df.get_group(pd_key)
                    except KeyError:
                        # TODO (from initial work on categorical plots refactor)
                        # We are adding this to allow backwards compatability
                        # with the empty artists that old categorical plots would
                        # add (before 0.12), which we may decide to break, in which
                        # case this option could be removed
                        df_subset = axes_df.loc[[]]

                    if df_subset.empty and not allow_empty:
                        continue

                    sub_vars = dict(zip(grouping_vars, key))
                    sub_vars.update(subplot_keys)

                    yield sub_vars, df_subset.copy(), subplot["axes"]

        return generate_splits

    def _repr_png_(self) -> bytes:

        # TODO better to do this through a Jupyter hook?
        # TODO Would like to allow for svg too ... how to configure?

        # TODO perhaps have self.show() flip a switch to disable this, so that
        # user does not end up with two versions of the figure in the output

        # Preferred behavior is to clone self so that showing a Plot in the REPL
        # does not interfere with adding further layers onto it in the next cell.
        # But we can still show a Plot where the user has manually invoked .plot()
        if hasattr(self, "_figure"):
            figure = self._figure
        else:
            figure = self.clone().plot()._figure

        buffer = io.BytesIO()

        # TODO use bbox_inches="tight" like the inline backend?
        # pro: better results,  con: (sometimes) confusing results
        # Better solution would be to default (with option to change)
        # to using constrained/tight layout.
        figure.savefig(buffer, format="png", bbox_inches="tight")
        return buffer.getvalue()


class Layer:

    data: PlotData  # TODO added externally (bad design?)

    def __init__(
        self,
        mark: Mark,
        stat: Stat | None,
        source: DataSource | None,
        variables: VariableSpec | None,
    ):

        self.mark = mark
        self.stat = stat
        self.source = source
        self.variables = variables

    def __contains__(self, key: str) -> bool:

        if self.data is None:
            return False
        return key in self.data
