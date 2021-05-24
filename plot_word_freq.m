function plot_word_freq(X)

if ischar(X)
    fprintf('loading data from file %s...\n', X);
    X = dlmread(X);
end

if ~issparse(X)
    disp('converting data to sparse matrix...');
    X = spconvert(X);
end

word_counts = full(sum(X,1));
disp('word count distribution (log scale):');
ascii_plot_histogram(word_counts, 1);

sorted_word_counts = sort(word_counts,'descend');
sorted_word_counts = sorted_word_counts(sorted_word_counts > 5);
best_k = 0;
best_rsquared = 0;
for k=1:0.01:2
    x = sorted_word_counts.^(-1/k);
    y = 1:length(x);
    p = polyfit(x, y, 1);
    yfit = polyval(p, x);
    rsquared = 1 - sum((y - yfit).^2) / ((length(y) - 1) * var(y));
    if rsquared > best_rsquared
        best_k = k;
        best_rsquared = rsquared;
    end
end
fprintf('linear approximation using power law on word counts (k = %.3g, R^2 = %.3g):\n', best_k, best_rsquared);
x = sorted_word_counts.^(-1/best_k);
x_sub = x(round(linspace(1,length(x),30)));
ascii_plot_bar(x_sub, 1:length(x_sub));
