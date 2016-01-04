function Tree = nHDP_step(Xid,Xcnt,Tree,scale,rho,beta0)
% NHDP_STEP performs one step of stochastic variational inference 
% for the nested hierarchical Dirichlet process.
%
% *** INPUT (mini-batch) ***
% Xid{d} : contains vector of word indexes for document d
% Xcnt{d} : contains vector of word counts corresponding to Xid{d}
% Tree : current top-level of nHDP
% rho : step size
%
% Written by John Paisley, jpaisley@berkeley.edu

Voc = length(Tree(1).beta_cnt); % W
tot_tops = length(Tree); % K
D = length(Xid); % Dt
size_subtree = zeros(1,D); % Kt per doc
hist_levels = zeros(1,10); % counts of... branch lengths?

% batch suff stats
B_up = zeros(tot_tops,Voc); % suff stats for theta (K x W)
weight_up = zeros(tot_tops,1); % suff stats for V (K x 1)

gamma1 = 5; % top-level DP concentration
gamma2 = 1; % second-level DP concentration
gamma3 = 2*(1/3); % beta stopping switches
gamma4 = 2*(2/3);

% put info from Tree struct into matrix and vector form
% ElnB: Elogtheta (K x W)
% ElnPtop: Elogp (with implicit reordering of nodes) (1 x K)
% id_parent: floating-point ids of node parents
% id_me: floating-point ids of nodes
[ElnB,ElnPtop,id_parent,id_me] = func_process_tree(Tree,beta0,gamma1);

% Family tree indicator matrix. Tree_mat(j,i) = 1 indicates that node j is 
% along the path from node i to the root node (not including node i or root)
Tree_mat = zeros(tot_tops);
for i = 1:tot_tops
    bool = 1;
    idx = i;
    % iteratively climb up tree until we hit root
    while bool
        idx = find(id_me==id_parent(idx)); % get integer id of parent of idx
        if ~isempty(idx)
            % not root, set indicator
            Tree_mat(idx,i) = 1;
        else
            % root, stop iteration
            bool = 0;
        end        
    end
end

% ELBO terms for U (Elogpi prior for just level penalty)
level_penalty = psi(gamma3) - psi(gamma3+gamma4) + sum(Tree_mat,1)'*(psi(gamma4) - psi(gamma3+gamma4));

% E-step
for d = 1:D
    ElnB_d = ElnB(:,Xid{d}); % doc-wise Elogtheta (K x Wd)
    ElnV = psi(1) - psi(1+gamma2); % local ElogV prior
    Eln1_V = psi(gamma2) - psi(1+gamma2); % local Elog(1-V) prior
    ElnP_d = zeros(tot_tops,1) - inf; % Elogpi prior (initially all nodes inactive...) (K x 1)
    ElnP_d(id_parent==log(2)) = ElnV+psi(gamma3)-psi(gamma3+gamma4); % activate children of root

    % select subtree
    bool = 1;
    idx_pick = []; % global indices of selected nodes
    Lbound = []; % scores (topic assignment and word observation ELBO components) of the subtrees corresponding to successively selected nodes
    vec_DPweights = zeros(tot_tops,1); % ElnP_d minus the level penalty---that is, Elogpi prior for just the current level's local V terms (K x 1)
    while bool
        idx_active = find(ElnP_d > -inf); % indices of active (selected and potential) nodes
        penalty = ElnB_d(idx_active,:) + repmat(ElnP_d(idx_active),1,length(Xid{d})); % Elogtheta + Elogpi for active nodes (Ka x Wd)
        C_act = penalty; % nu... see update below (Ka x Wd)
        penalty = penalty.*repmat(Xcnt{d},size(penalty,1),1); % Elogtheta + Elogpi, scaled by word counts
        ElnPtop_act = ElnPtop(idx_active); % Elogp (global pi) of active nodes
        if isempty(idx_pick)
            % selecting first node, all active nodes are candidates
            score = sum(penalty,2) + ElnPtop_act; % Elogtheta + Elogpi + Elogp for active nodes (Ka x 1)
            [temp,idx_this] = max(score); % find best candidate (highest score)
            idx_pick = idx_active(idx_this); % store global index of best candidate
            Lbound(end+1) = temp - ElnPtop_act(idx_this); % append likelihood (topic assignment and word observation components of ELBO) for best candidate to Lbound
        else
            % selecting subsequent node, some active nodes are already selected

            temp = zeros(tot_tops,1); % indices of active nodes (K x 1)
            temp(idx_active) = (1:length(idx_active))';
            idx_clps = temp(idx_pick); % indices of selected nodes
            num_act = length(idx_active); % number of active nodes
            vec = max(penalty(idx_clps,:),[],1); % word-wise max scaled Elogtheta + Elogpi (1 x Wd)

            % remove scaled Elogtheta + Elogpi for *best* node selected so far
            % from scaled Elogtheta + Elogpi for *each* node selected so far
            % and exponentiate (compute unnormalized nu, adjusting to prevent under/overflow) (Ka x Wd)
            C_act = C_act - repmat(vec,num_act,1);
            C_act = exp(C_act);

            % this part is a little tricky...
            % note below that we set score = -inf for selected nodes, so in the matrix arithmetic that leads up to that,
            % focus on the rows corresponding to active but not selected nodes.  we find that (for one of those rows)
            % numerator is Elogtheta + Elogpi for the selected nodes and the unselected node corresponding to the
            % current row, weighted by unnormalized nu (C_act) and the word counts (which appear in penalty);
            numerator = C_act.*penalty; % scaled Elogtheta + Elogpi, scaled by nu (Ka x Wd)
            numerator = numerator + repmat(sum(numerator(idx_clps,:),1),num_act,1);
            % denominator is the normalizer for nu for the selected nodes and the unselected node corresponding to
            % the given row;
            denominator = C_act + repmat(sum(C_act(idx_clps,:),1),num_act,1);
            % vec is sum of - nu' log nu' across the selected nodes;
            vec = sum(C_act(idx_clps,:).*log(eps+C_act(idx_clps,:)),1);
            % H is - nu log nu;
            H = log(denominator) - (C_act.*log(C_act+eps) + repmat(vec,num_act,1))./denominator;
            % and score is per-topic weighted sum of nu * (Elogtheta + Elogpi) (weighted by word counts)
            % + Elogp - per-topic weighted sum of nu log nu (weighted by word counts)
            score = sum(numerator./denominator,2) + ElnPtop_act + H*Xcnt{d}';
            score(idx_clps) = -inf; % set score of selected nodes to -inf (little hack)
            [temp,idx_this] = max(score); % compute best candidate (active but not selected nodes)
            idx_pick(end+1) = idx_active(idx_this); % store globl index of best candidate
            Lbound(end+1) = temp - ElnPtop_act(idx_this); % append likelihood (topic assignment and word observation components of ELBO) for best candidate to Lbound
        end

        % update candidates according to new selected node
        idx_this = find(id_parent == id_parent(idx_pick(end))); % find global indices of recently selected node's siblings
        [t1,t2] = intersect(idx_this,idx_pick); % remove nodes already selected (including recently selected node) from siblings (idx_this), leaving unselected siblings remaining
        idx_this(t2) = [];
        vec_DPweights(idx_this) = vec_DPweights(idx_this) + Eln1_V; % add local Elog(1-V) prior to unselected sibling DP Elogpi (excludes level penalty)
        ElnP_d(idx_this) = ElnP_d(idx_this) + Eln1_V; % update local Elogpi with Elog(1-V) prior for unselected siblings
        idx_add = find(id_parent == id_me(idx_pick(end))); % find global indices of recently selected node's children
        vec_DPweights(idx_add) = ElnV; % add local ElogV prior to new children DP Elogpi (excludes level penalty)
        ElnP_d(idx_add) = ElnV + level_penalty(idx_add); % add local ElogV and ElogU... prior to new children Elogpi
        % walk up tree from recently-selected node, adding DP Elogpi (excludes level penalty) to new-children Elogpi... what about new-children DP Elogpi (excludes level penalty)?
        bool2 = 1;
        idx = idx_pick(end);
        while bool2
            if ~isempty(idx) %id_me(idx) ~= log(2)
                ElnP_d(idx_add) = ElnP_d(idx_add) + vec_DPweights(idx);
                idx = find(id_me == id_parent(idx));
            else
                bool2 = 0;
            end
        end

        % stop if relative change in ELBO is less than 1e-3 or subtree has 20 nodes
        if length(Lbound) > 1
            if abs(Lbound(end)-Lbound(end-1))/abs(Lbound(end-1)) < 10^-3 || length(Lbound) == 20
                bool = 0;
            end
        end

        hist_levels(length(Tree(idx_pick(end)).me)-1) = hist_levels(length(Tree(idx_pick(end)).me)-1) + 1;
%         plot(Lbound); title(num2str(length(Tree(idx_pick(end)).me))); pause(.1);
    end
    size_subtree(d) = length(idx_pick); % store size of selected subtree

    % learn document parameters for subtree
    T = length(idx_pick); % again, size of subtree
    ElnB_d = ElnB(idx_pick,Xid{d}); % Elogtheta for given subtree, words
    ElnP_d = 0*ElnP_d(idx_pick) - 1; % Elogpi for given subtree... ignored for first iteration
    cnt_old = zeros(length(idx_pick),1);
    bool_this = 1;
    num = 0;
    while bool_this
        num = num+1;
        % estimate nu
        C_d = ElnB_d + repmat(ElnP_d,1,length(Xid{d}));
        C_d = C_d - repmat(max(C_d,[],1),T,1);
        C_d = exp(C_d);
        C_d = C_d./repmat(sum(C_d,1),T,1);
        % store nu sums (one per topic) in cnt
        cnt = C_d*Xcnt{d}';
        % estimate Elogpi
        ElnP_d = func_doc_weight_up(cnt,id_parent(idx_pick),gamma2,gamma3,gamma4,Tree_mat(idx_pick,idx_pick));
        % stop if rel change in nu sums is less than 1e-3 or 50 iters elapsed
        if sum(abs(cnt-cnt_old))/sum(cnt) < 10^-3 || num == 50
            bool_this = 0;
        end
        cnt_old = cnt;
%         stem(cnt); title(num2str(num)); pause(.1);
    end
    % update batch theta ss
    B_up(idx_pick,Xid{d}) = B_up(idx_pick,Xid{d}) + C_d.*repmat(Xcnt{d},length(idx_pick),1);
    % update batch V ss
    weight_up(idx_pick) = weight_up(idx_pick) + 1;
end
subplot(2,2,[1 3]); stem(sum(B_up,2));
subplot(2,2,2); bar(hist(size_subtree,1:20));
subplot(2,2,4); bar(hist_levels/D);

% M-step
% note scale here is as used in init, e.g. 100D/K... (?!)
% and note the D variable below is the *batch* size
for i = 1:tot_tops
    if rho == 1
        Tree(i).beta_cnt = scale*B_up(i,:)/D;
    else
        % compute avg theta ss for this topic
        vec = ones(1,size(B_up,2));
        vec = vec/sum(vec);
        vec = sum(B_up(i,:))*vec;
        % set theta ss to (1 - rho) * old + rho * ((1 - rho/10) * new + rho/10 * avg theta ss for this topic)
        Tree(i).beta_cnt = (1-rho)*Tree(i).beta_cnt + rho*((1-rho/10)*scale*B_up(i,:)/D + (rho/10)*scale*vec/D);
    end
    Tree(i).cnt = (1-rho)*Tree(i).cnt + rho*scale*weight_up(i)/D;
end
