function ElnP_d = func_doc_weight_up(tau_sums,id_parent,model_params,Tree_mat)
% update expected log probability of each topic selected for this document

T = length(tau_sums);
ElnP_d = zeros(T,1);

bin_cnt1 = tau_sums; % suff stats (nu sums) per topic
bin_cnt0 = Tree_mat*tau_sums; % suff stats (nu sums) of children of each topic
Elnbin1 = psi(bin_cnt1+model_params.gamma1) - psi(bin_cnt1+bin_cnt0+model_params.gamma1+model_params.gamma2); % Elogpi components for U
Elnbin0 = psi(bin_cnt0+model_params.gamma2) - psi(bin_cnt1+bin_cnt0+model_params.gamma1+model_params.gamma2); % Elogpi components for 1-U

% % don't re-order weights
% stick_cnt = bin_cnt1+bin_cnt0;
% partition = unique(id_parent);
% for i = 1:length(partition)
%     idx = find(id_parent==partition(i));
%     t1 = stick_cnt(idx);
%     t3 = rev_cumsum(t1);
%     if length(t3) > 1
%         t4 = [t3(2:end) ; 0];
%         t5 = [0 ; psi(t4(1:end-1)+model_params.beta) - psi(t1(1:end-1)+t4(1:end-1)+1+model_params.beta)];
%     else
%         t4 = 0;
%         t5 = 0;
%     end
%     ElnP_d(idx) =  psi(t1+1) - psi(t1+t4+1+model_params.beta) + cumsum(t5);
% end
% this = ElnP_d + Elnbin1 + Tree_mat'*(Elnbin0 + ElnP_d);
% ElnP_d = this;

% re-order weights
stick_cnt = bin_cnt1+bin_cnt0; % suff stats for subtree rooted at topic
partition = unique(id_parent); % unique parent node real ids
% first we compute level-wise components
for i = 1:length(partition)
    idx = find(id_parent==partition(i)); % find children of this parent
    t1 = stick_cnt(idx); % select subtree suff stats of children (suff stats for stopping)
    [t1,idx_sort] = sort(t1,'descend'); % sort children by subtree suff stats
    t3 = rev_cumsum(t1); % rev cumsum of sorted children subtree suff stats
    if length(t3) > 1
        t4 = [t3(2:end) ; 0]; % accumulated subtree suff stats of nodes to the right (suff stats for continuing)
        t5 = [0 ; psi(t4(1:end-1)+model_params.beta) - psi(t1(1:end-1)+t4(1:end-1)+1+model_params.beta)]; % Elogpi components for 1-V
    else
        t4 = 0;
        t5 = 0;
    end
    % Elogpi components for V + accumulated Elogpi components for 1-V
    weights = psi(t1+1) - psi(t1+t4+1+model_params.beta) + cumsum(t5);
    % store Elogpi components for V and 1-V, taking the reording into account
    ElnP_d(idx(idx_sort)) = weights;
end
% now we aggregate up the tree:
% compute Elogpi as component for stopping at a given node in the current level + components for continuing along the given nodes in the levels above
this = ElnP_d + Elnbin1 + Tree_mat'*(Elnbin0 + ElnP_d);
ElnP_d = this;
