% display vocabulary results. the vocabulary is in a cell called vocab.
addpath /scratch/groups/bvandur1/redis-mat
input_db = redis(input_host, input_port);
vocab = load_vocab(input_db, vocab_key);
num_words = 10;
[ElnB,ElnPtop,id_parent,id_me] = func_process_tree(Tree,beta0,5); % this only needs to be done once
for idx=1:num_topics(1)
    idx_c = find(id_parent==id_me(idx));
    fprintf('Node %d (%f)', idx, Tree(idx).cnt)
    [a,b] = sort(Tree(idx).beta_cnt,'descend');
    for w = 1:num_words
        fprintf('  %s', vocab{b(w)});
    end
    fprintf('\n');
    if ~isempty(idx_c)
        for i = 1:length(idx_c)
            fprintf('Child %d (%d) (%f)', i, idx_c(i), Tree(idx_c(i)).cnt);
            [a,b] = sort(Tree(idx_c(i)).beta_cnt,'descend');
            for w = 1:num_words
                fprintf('  %s', vocab{b(w)});
            end
            fprintf('\n');
        end
    end
    fprintf('\n');
end
