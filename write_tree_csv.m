function write_tree_csv(Tree, filename)

fh = fopen(filename, 'w');

fprintf(fh, 'me,parent,cnt,beta_cnt\n');
for i=1:length(Tree)
    node = Tree(i);

    me = sprintf('%d ', node.me);
    me = me(1:end-1);

    parent = sprintf('%d ', node.parent);
    parent = parent(1:end-1);

    cnt = sprintf('%g', node.cnt);

    beta_cnt = sprintf('%g ', node.beta_cnt);
    beta_cnt = beta_cnt(1:end-1);

    fprintf(fh, '%s,%s,%s,%s\n', me, parent, cnt, beta_cnt);
end

fclose(fh);
