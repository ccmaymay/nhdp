function Tree = nHDP_init(X,model_params,alg_params)
% NHDP_INIT initializes the nHDP algorithm using a tree-structured k-means algorithm.
%
% Written by John Paisley, jpaisley@berkeley.edu

L = length(model_params.num_topics); % number of levels in tree (depth)
[actual_init_size, Voc] = size(X);
X = full(X)'; % important that we make copy of X as it will be modified
X = X ./ sum(X,1);

% each natural number has a unique prime factorization prod_i[(p_i)^(n_i)], so
% each log prime factorization sum_i[n_i log p_i] produces a different real number
godel = log([2 3 5 7 11 13 17 19 23 29 31 37 41 43 47]);
cluster_assignments = zeros(L+1,actual_init_size); % matrix of vector indices (column vectors) representing document assignments
cluster_assignments(1,:) = 1;
Tree = [];

for l = 1:L % loop over levels
    fprintf('beginning initialization step %d/%d...\n', l, L);
    K = model_params.num_topics(l); % number of topics at this level
    vec = godel(1:l)*cluster_assignments(1:l,:); % compute real ids of topics at this level to documents is assigned
    S = unique(vec); % compute real ids of topics at this level to which at least one doc is assigned
    for s = 1:length(S) % loop over topics used at this level
        idx = find(vec == S(s)); % compute vector id of current topic
        X_sub = X(:,idx); % compute subset of documents assigned here
        [centroids,cluster_assignments_sub] = K_means_L1(X_sub,K,alg_params.init_num_iters);
        cluster_assignments(l+1,idx) = cluster_assignments_sub; % update assignments table, assigning current documents to children of s
        tau_sums = histc(cluster_assignments_sub,1:K); % number of topics assigned to each cluster
        % initialize tree nodes for children of s
        for i = 1:size(centroids,2)
            Tree(end+1).lambda_sums = centroids(:,i)'; % 1 x W theta ss
            Tree(end).tau_sums = tau_sums(i); % count of docs in subtree rooted here
            Tree(end).parent = cluster_assignments(1:l,idx(1))'; % vector id of parent
            Tree(end).me = [Tree(end).parent i]; % vector id
        end
        % compute "probability of what remains"
        for i = 1:length(cluster_assignments_sub)
            X(:,idx(i)) = X(:,idx(i)) - centroids(:,cluster_assignments_sub(i)); % subtract off mean
            X(X(:,idx(i))<0,idx(i)) = 0; % threshold out negative values
            X(:,idx(i)) = X(:,idx(i))/sum(X(:,idx(i))); % renormalize
        end
    end
end

disp('postprocessing initialized tree...');
for i = 1:length(Tree)
    if Tree(i).tau_sums == 0
        Tree(i).lambda_sums(:) = 0;
    end
    % generate Dirichlet rv with concentration param init_rand_scale / Voc
    randomness = gamrnd(ones(1,length(Tree(i).lambda_sums)) * alg_params.init_rand_scale / Voc,1);
    randomness = randomness / sum(randomness);
    Tree(i).lambda_sums = alg_params.init_scale*(alg_params.kappa*Tree(i).lambda_sums + (1 - alg_params.kappa)*(1/Voc + randomness));
    Tree(i).tau_sums = (alg_params.init_scale/actual_init_size) * Tree(i).tau_sums;
end

function [centroids,cluster_assignments] = K_means_L1(X,K,num_iters)
% K-Means algorithm with L1 assignment and L2 mean minimization
%
% input:
% X: num_features x num_samples data matrix
% K: number of clusters
% num_iters: number of iterations
%
% output:
% centroids: num_features x K cluster centers
% cluster_assignments: 1 x num_samples cluster assignments

num_samples = size(X,2);
if num_samples >= K % more examples than cluster centers, over-specified (good)
    % make random permutation
    % initialize cluster centers as random unique data vectors
    [~,b] = sort(rand(1,num_samples));
    centroids = X(:,b(1:K));
else % more cluster centers than examples, under-specified (bad)
    % initialize cluster centers as random vectors (i.i.d. uniform)
    centroids = rand(size(X,1),K);
    centroids = centroids./sum(centroids,1);
end

cluster_assignments = zeros(1,num_samples);
for i = 1:num_iters % main loop
    % E-step
    for d = 1:num_samples
        [~,cluster_assignments(d)] = min(sum(abs(centroids - X(:,d)),1));
    end
    % M-step
    for k = 1:K
        centroids(:,k) = mean(X(:,cluster_assignments==k),2);
    end
end

% re-index clusters so largest clusters have lowest indices
cluster_sizes = histc(cluster_assignments,1:K);
[~,cluster_sort_indices] = sort(cluster_sizes,'descend');
centroids = centroids(:,cluster_sort_indices);
new_cluster_assignments = zeros(1,length(cluster_assignments));
for i = 1:length(cluster_sort_indices)
    idx = find(cluster_assignments == cluster_sort_indices(i));
    new_cluster_assignments(idx) = i;
end
cluster_assignments = new_cluster_assignments;
