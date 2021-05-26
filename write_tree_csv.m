function write_tree_csv(Tree, filename)

fh = fopen(filename, 'w');

fprintf(fh, 'me,parent,tau_sums,lambda_sums\n');
for i=1:length(Tree)
    node = Tree(i);

    me = sprintf('%d ', node.me);
    me = me(1:end-1);

    parent = sprintf('%d ', node.parent);
    parent = parent(1:end-1);

    tau_sums = sprintf('%g', node.tau_sums);

    lambda_sums = sprintf('%g ', node.lambda_sums);
    lambda_sums = lambda_sums(1:end-1);

    fprintf(fh, '%s,%s,%s,%s\n', me, parent, tau_sums, lambda_sums);
end

fclose(fh);
