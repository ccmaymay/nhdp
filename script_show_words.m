% display vocabulary results. the vocabulary is in a cell called vocab.

[ElnB,ElnPtop,id_parent,id_me] = func_process_tree(Tree,beta0,5); % this only needs to be done once

idx = 1;    % pick a topic index to show results for
idx_p = find(id_me==id_parent(idx));
idx_c = find(id_parent==id_me(idx));
disp('*** This node ***');
disp(['Count ' num2str(Tree(idx).cnt)]);
[a,b] = sort(Tree(idx).beta_cnt,'descend');
for w = 1:10
    disp(['   ' vocab{b(w)}]);
end
disp('*** Parent node ***');
if isempty(idx_p)
    disp('No parent');
else
    [a,b] = sort(Tree(idx_p).beta_cnt,'descend');
    for w = 1:10
        disp(['   ' vocab{b(w)}]);
    end
end
if isempty(idx_c)
    disp('No children');
else
    for i = 1:length(idx_c)
        disp(['Child ' num2str(i) ' : Count ' num2str(Tree(idx_c(i)).cnt) ' : Index ' num2str(idx_c(i))]);
        [a,b] = sort(Tree(idx_c(i)).beta_cnt,'descend');
        for w = 1:10
            disp(['   ' vocab{b(w)}]);
        end
    end
end