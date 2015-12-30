% display vocabulary results. the vocabulary is in a cell called vocab.
num_words = 5;
[ElnB,ElnPtop,id_parent,id_me] = func_process_tree(Tree,beta0,5); % this only needs to be done once
for idx=1:num_topics(1)
    idx_c = find(id_parent==id_me(idx));
    disp(sprintf('*** This node: %d ***', idx));
    disp(['Count ' num2str(Tree(idx).cnt)]);
    [a,b] = sort(Tree(idx).beta_cnt,'descend');
    for w = 1:num_words
        disp(['   ' vocab{b(w)}]);
    end
    if ~isempty(idx_c)
        for i = 1:length(idx_c)
            disp(['Child ' num2str(i) ' : Count ' num2str(Tree(idx_c(i)).cnt) ' : Index ' num2str(idx_c(i))]);
            [a,b] = sort(Tree(idx_c(i)).beta_cnt,'descend');
            for w = 1:num_words
                disp(['   ' vocab{b(w)}]);
            end
        end
    end
    disp(' ')
end
