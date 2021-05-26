function Tree = nHDP_step(X,Tree,model_params,alg_params,rho)
% NHDP_STEP performs one step of stochastic variational inference
% for the nested hierarchical Dirichlet process.
%
% *** INPUT (mini-batch) ***
% X : contains word counts for each doc (row) and word (col)
% Tree : current top-level of nHDP
% rho : step size
%
% Written by John Paisley, jpaisley@berkeley.edu

total_num_topics = length(Tree); % K
[actual_batch_size,Voc] = size(X); % batch size |Cs|, vocabulary size W
subtree_sizes = zeros(1,actual_batch_size); % Kt per doc
num_levels = 0;
for i = 1:total_num_topics
    num_levels = max(num_levels, length(Tree(i).me) - 1);
end
node_level_counts = zeros(1,num_levels);
node_level_edges = 1:num_levels;

% batch suff stats
B_up = zeros(total_num_topics,Voc); % suff stats for theta (K x W)
weight_up = zeros(total_num_topics,1); % suff stats for V (K x 1)

% put info from Tree struct into matrix and vector form
% ElnB: Elogtheta (K x W)
% ElnPtop: Elogp (with implicit reordering of nodes) (1 x K)
% id_parent: real ids of node parents
% id_me: real ids of nodes
[ElnB,ElnPtop,id_parent,id_me] = func_process_tree(Tree,model_params);

% Family tree indicator matrix. Tree_mat(j,i) = 1 indicates that node j is
% along the path from node i to the root node (not including node i or root)
Tree_mat = zeros(total_num_topics);
for i = 1:total_num_topics
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
level_penalty = psi(model_params.gamma1) - psi(model_params.gamma1+model_params.gamma2) + sum(Tree_mat,1)'*(psi(model_params.gamma2) - psi(model_params.gamma1+model_params.gamma2));

% E-step
for d = 1:actual_batch_size
    X_d = X(d,:);
    X_d_ids = reshape(find(X_d),1,[]);
    X_d_vals = reshape(nonzeros(X_d),1,[]);

    ElnB_d = ElnB(:,X_d_ids); % doc-wise Elogtheta (K x Wd)
    ElnV = psi(1) - psi(1+model_params.beta); % local ElogV prior
    Eln1_V = psi(model_params.beta) - psi(1+model_params.beta); % local Elog(1-V) prior
    ElnP_d = zeros(total_num_topics,1) - inf; % Elogpi prior (initially all nodes inactive...) (K x 1)
    ElnP_d(id_parent==log(2)) = ElnV+psi(model_params.gamma1)-psi(model_params.gamma1+model_params.gamma2); % activate children of root

    % select subtree
    bool = 1;
    idx_pick = []; % global indices of selected nodes
    Lbound = []; % scores (topic assignment and word observation ELBO components) of the subtrees corresponding to successively selected nodes
    vec_DPweights = zeros(total_num_topics,1); % ElnP_d minus the level penalty---that is, Elogpi prior for just the current level's local V terms (K x 1)
    while bool
        idx_active = find(ElnP_d > -inf); % indices of active (selected and potential) nodes
        penalty = ElnB_d(idx_active,:) + repmat(ElnP_d(idx_active),1,length(X_d_ids)); % Elogtheta + Elogpi for active nodes (Ka x Wd)
        C_act = penalty; % nu... see update below (Ka x Wd)
        penalty = penalty.*repmat(X_d_vals,size(penalty,1),1); % Elogtheta + Elogpi, scaled by word counts
        ElnPtop_act = ElnPtop(idx_active); % Elogp (global pi) of active nodes
        if isempty(idx_pick)
            % selecting first node, all active nodes are candidates
            score = sum(penalty,2) + ElnPtop_act; % Elogtheta + Elogpi + Elogp for active nodes (Ka x 1)
            [temp,idx_this] = max(score); % find best candidate (highest score)
            idx_pick = idx_active(idx_this); % store global index of best candidate
            Lbound(end+1) = temp - ElnPtop_act(idx_this); % append likelihood (topic assignment and word observation components of ELBO) for best candidate to Lbound
        else
            % selecting subsequent node, some active nodes are already selected

            temp = zeros(total_num_topics,1); % indices of active nodes (K x 1)
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
            score = sum(numerator./denominator,2) + ElnPtop_act + H*X_d_vals';
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

        % stop if relative change in ELBO is less than 1e-3
        if length(Lbound) > 1
            if abs(Lbound(end)-Lbound(end-1))/abs(Lbound(end-1)) < alg_params.subtree_sel_threshold %|| length(Lbound) == 20
                bool = 0;
            end
        end
        node_level_counts(length(Tree(idx_pick(end)).me)-1) = node_level_counts(length(Tree(idx_pick(end)).me)-1) + 1;
%         plot(Lbound); title(num2str(length(Tree(idx_pick(end)).me))); pause(.1);
    end
    subtree_sizes(d) = length(idx_pick); % store size of selected subtree

    % learn document parameters for subtree
    T = length(idx_pick); % again, size of subtree
    ElnB_d = ElnB(idx_pick,X_d_ids); % Elogtheta for given subtree, words
    ElnP_d = ElnP_d(idx_pick); % Elogpi for given subtree
    tau_sums_old = zeros(length(idx_pick),1);
    bool_this = 1;
    while bool_this
        % estimate nu
        C_d = ElnB_d + repmat(ElnP_d,1,length(X_d_ids));
        C_d = exp(C_d - max(C_d,[],1));
        C_d = C_d./sum(C_d,1);
        % store nu sums (one per topic) in tau_sums
        scaled_nu = C_d.*X_d_vals;
        tau_sums = sum(scaled_nu,2);
        % estimate Elogpi
        ElnP_d = func_doc_weight_up(tau_sums,id_parent(idx_pick),model_params,Tree_mat(idx_pick,idx_pick));
        % stop if rel change in nu sums is less than 1e-3
        if sum(abs(tau_sums-tau_sums_old))/sum(tau_sums) < alg_params.local_update_threshold %|| num == 50
            bool_this = 0;
        end
        tau_sums_old = tau_sums;
%         stem(tau_sums); title(num2str(num)); pause(.1);
    end
    % update batch theta ss
    B_up(idx_pick,X_d_ids) = B_up(idx_pick,X_d_ids) + scaled_nu;
    % update batch V ss
    weight_up(idx_pick) = weight_up(idx_pick) + 1;
end
B_up_sums = sum(B_up,2);
disp('topic expected word count distribution:');
ascii_plot_histogram(B_up_sums);
disp('subtree size distribution:');
ascii_plot_histogram(subtree_sizes);
disp('node level distribution:');
ascii_plot_bar(node_level_counts, node_level_edges);

% M-step
for i = 1:total_num_topics
    % B_up: lambda
    % weight_up: tau
    scaled_B_up = (alg_params.batch_scale/actual_batch_size) * B_up(i,:);
    if rho == 1
        Tree(i).lambda_sums = scaled_B_up
    else
        % set theta ss to (1 - rho) * old + rho * ((1 - rho/10) * new + rho/10 * avg theta ss for this topic)
        % (for smoothing?)
        Tree(i).lambda_sums = (1-rho)*Tree(i).lambda_sums + rho*( ...
            (1-rho/10)*scaled_B_up + (rho/10)*mean(scaled_B_up));
    end
    Tree(i).tau_sums = (1-rho)*Tree(i).tau_sums + rho*alg_params.batch_scale*weight_up(i)/actual_batch_size;
end
