function Tree = nHDP_init(X,num_topics,scale)
% NHDP_INIT initializes the nHDP algorithm using a tree-structured k-means algorithm.
%
% Written by John Paisley, jpaisley@berkeley.edu

L = length(num_topics); % number of levels in tree (depth)
D = size(X,1); % Dt
X = full(X)';
X = X ./ sum(X,1);

num_ite = 3;
godel = log([2 3 5 7 11 13 17 19 23 29 31 37 41 43 47]);
C = zeros(L+1,D); % L+1 x Dt matrix of vector indices (column vectors) representing document assignments
C(1,:) = 1;
Tree = [];

for l = 1:L % loop over levels
    fprintf('beginning initialization step %d/%d...\n', l, L);
    K = num_topics(l); % number of topics at this level
    vec = godel(1:l)*C(1:l,:); % compute floating-point ids of topics at this level to documents is assigned
    S = unique(vec); % compute floating-point ids of topics at this level to which at least one doc is assigned
    for s = 1:length(S) % loop over topics used at this level
        idx = find(vec == S(s)); % compute vector id of current topic
        X_sub = X(:,idx); % compute subset of documents assigned here
        [B,c] = K_means_L1(X_sub,K,num_ite);
        C(l+1,idx) = c; % update assignments table, assigning current documents to children of s
        cnt = histc(c,1:num_topics(l)); % number of topics assigned to each cluster
        for i = 1:size(B,2) % loop over children
            Tree(end+1).beta_cnt = scale*B(:,i)'; % 1 x W theta ss
            Tree(end).cnt = scale*cnt(i)/D; % count of docs in subtree rooted here
            Tree(end).parent = C(1:l,idx(1))'; % vector id of parent
            Tree(end).me = [Tree(end).parent i]; % vector id
        end
        % subtract off mean
        for i = 1:length(c)
            X(:,idx(i)) = X(:,idx(i)) - B(:,c(i)); % subtract off mean
            X(X(:,idx(i))<0,idx(i)) = 0; % threshold out negative values
            X(:,idx(i)) = X(:,idx(i))/sum(X(:,idx(i))); % renormalize
        end
    end
end

function [centroids,c] = K_means_L1(X,K,maxite)
% K-Means algorithm with L1 assignment and L2 mean minimization
%
% input:
% X: W x D data matrix
% K: number of clusters
% maxite: number of iterations
%
% output:
% B: W x K cluster centers
% c: 1 x D cluster assignments

D = size(X,2);
if D >= K % more examples than cluster centers, over-specified (good)
    % make random permutation
    % initialize cluster centers as random unique data vectors
    [~,b] = sort(rand(1,D));
    centroids = X(:,b(1:K));
else % more cluster centers than examples, under-specified (bad)
    % initialize cluster centers as random vectors (i.i.d. uniform)
    centroids = rand(size(X,1),K);
    centroids = centroids./repmat(sum(centroids,1),size(centroids,1),1);
end
c = zeros(1,D);

for ite = 1:maxite % main loop
    % E-step
    for d = 1:D
        [~,c(d)] = min(sum(abs(centroids - repmat(X(:,d),1,K)),1));
    end
    % M-step
    for k = 1:K
        centroids(:,k) = mean(X(:,c==k),2);
    end
end

% compute sizes of clusters
cnt = histc(c,1:K);
[t1,t2] = sort(cnt,'descend');
% sort B and c by cluster size, descending
centroids = centroids(:,t2);
c2 = zeros(1,length(c));
for i = 1:length(t2)
    idx = find(c == t2(i));
    c2(idx) = i;
end
c = c2;
