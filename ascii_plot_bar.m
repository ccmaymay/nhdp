function ascii_plot_bar(x_heights, x_edges, graph_width)

if ~exist('edge_format', 'var')
    if all(is_int_val(x_edges))
        max_digit_length = 1;
        for i=1:length(x_edges)
            max_digit_length = max(max_digit_length, length(sprintf('%d', x_edges(i))));
        end
        edge_format = sprintf('%%%dd', max_digit_length);
    else
        edge_format = '%7.2g';
    end
end

if ~exist('graph_width', 'var')
    graph_width = 50;
end

plot_line_scale = graph_width / max(x_heights);

fprintf('  total: %d\n', sum(x_heights));

for i=1:length(x_heights)
    if length(x_heights) == length(x_edges)
        axis_label = sprintf(edge_format, ceil(x_edges(i)));
    else
        if i == length(x_heights)
            r_bracket = ']';
        else
            r_bracket = ')';
        end
        axis_label = sprintf('[%s, %s%s', ...
            sprintf(edge_format, x_edges(i)), ...
            sprintf(edge_format, x_edges(i + 1)), ...
            r_bracket);
    end

    plot_line = repmat('*', 1, floor(x_heights(i) * plot_line_scale));
    fprintf('  %s | %s\n', axis_label, plot_line);
end

fprintf('\n');
