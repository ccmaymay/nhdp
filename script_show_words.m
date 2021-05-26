fid = fopen(output_path, 'w');
num_words = 10;
[ElnB,ElnPtop,id_parent,id_me] = func_process_tree(Tree,beta0,5); % this only needs to be done once
for idx=1:num_topics(1)
    idx_c = find(id_parent==id_me(idx));
    fprintf(fid, 'Node %d (%f):', idx, Tree(idx).cnt);
    [a,b] = sort(Tree(idx).beta_cnt,'descend');
    for w = 1:num_words
        fprintf(fid, ' %d', b(w));
    end
    fprintf(fid, '\n');
    if ~isempty(idx_c)
        for i = 1:length(idx_c)
            fprintf(fid, 'Child %d (%d) (%f):', i, idx_c(i), Tree(idx_c(i)).cnt);
            [a,b] = sort(Tree(idx_c(i)).beta_cnt,'descend');
            for w = 1:num_words
                fprintf(fid, ' %d', b(w));
            end
            fprintf(fid, '\n');
        end
    end
    fprintf(fid, '\n');
end
fclose(fid);
