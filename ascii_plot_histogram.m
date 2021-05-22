function ascii_plot_histogram(x, varargin)

[x_counts, x_edges] = histcounts(x);

if all(is_int_val(x)) && all(diff(x_edges) == 1)
    x_edges = ceil(x_edges(1:length(x_counts)));
end

ascii_plot_bar(x_counts, x_edges, varargin{:});
