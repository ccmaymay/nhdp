function ascii_plot_histogram(x, plot_width)

if ~exist('plot_width', 'var')
    plot_width = 50;
end

[x_counts, x_edges] = histcounts(x);
plot_line_scale = plot_width / max(x_counts);

for i=1:length(x_counts)
    if is_int_val(x)
        if all(diff(x_edges) == 1)
            axis_label = sprintf('%d', ceil(x_edges(i)));
        else
            axis_label = sprintf('[%d, %d%]', ceil(x_edges(i)), floor(x_edges(i + 1)));
        end
    else
        if i == length(x_counts)
            r_bracket = ']';
        else
            r_bracket = ')';
        end
        axis_label = sprintf('[%7.2f, %7.2f%s', x_edges(i), x_edges(i + 1), r_bracket);
    end

    plot_line = repmat('*', 1, floor(x_counts(i) * plot_line_scale));
    fprintf('%s | %s\n', axis_label, plot_line);
end

function [b] = is_int_val(x)
b = isfinite(x) & x == floor(x);
