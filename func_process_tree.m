function [ElnB,ElnPtop,id_parent,id_me] = func_process_tree(Tree,beta0,gamma1)
% process the tree for the current batch
% (put info from Tree struct into matrix and vector form)

godel = log([2 3 5 7 11 13 17 19 23 29 31 37 41 43 47]);

Voc = length(Tree(1).beta_cnt);
tot_tops = length(Tree);

id_parent = zeros(tot_tops,1); % floating point ids of parents (K x 1)
id_me = zeros(tot_tops,1); % floating point ids of topics (K x 1)
ElnB = zeros(tot_tops,Voc); % Elogtheta (K x W)
count = zeros(tot_tops,1); % count of docs in subtrees routed at these nodes (K x 1)
for i = 1:length(Tree)
    % unique floating-point id of parent of node i
    id_parent(i) = Tree(i).parent*godel(1:length(Tree(i).parent))';
    % unique floating-point id of node i
    id_me(i) = Tree(i).me*godel(1:length(Tree(i).me))';
    % fill in row of Elogtheta
    ElnB(i,:) = psi(Tree(i).beta_cnt + beta0) - psi(sum(Tree(i).beta_cnt + beta0));
    % fill in element of doc assignment counts
    count(i) = Tree(i).cnt;
end

ElnPtop = zeros(tot_tops,1); % Elogp (global Elogpi) (K x 1)
groups = unique(id_parent); % set of floating-point parent ids
for g = 1:length(groups)
    % find integer indices of this node's children
    group_idx = find(id_parent==groups(g));
    this = count(group_idx); % get doc assignment counts of children
    [group_count,sort_group_idx] = sort(this,'descend'); % sort children by doc assignment counts
    a = group_count + 1; % estimate tau1 (1 x num children)
    b = [rev_cumsum(group_count(2:end)) ; 0] + gamma1; % estimate tau2 (1 x num children)
    ElnV = psi(a) - psi(a+b); % compute ElogV
    Eln1_V = psi(b) - psi(a+b); % compute Elog(1 - V)
    vec = ElnV + [0 ; cumsum(Eln1_V(1:end-1))]; % compute Elogp
    ElnPtop(group_idx(sort_group_idx)) = vec; % store Elogp for this node's children in the global array
end
    
    
    
