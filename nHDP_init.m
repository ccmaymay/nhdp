function Tree = nHDP_init(X,num_topics,scale)
% NHDP_INIT initializes the nHDP algorithm using a tree-structured k-means algorithm.
%
% Written by John Paisley, jpaisley@berkeley.edu

L = length(num_topics);
D = size(X,1);
X = full(X)';
X = X ./ sum(X,1);

num_ite = 3;
godel = log([2 3 5 7 11 13 17 19 23 29 31 37 41 43 47]);
C = zeros(L+1,D);
C(1,:) = 1;
Tree = [];

for l = 1:L
    fprintf('beginning initialization step %d/%d...\n', l, L);
    K = num_topics(l);
    vec = godel(1:l)*C(1:l,:);
    S = unique(vec);
    for s = 1:length(S)
        idx = find(vec == S(s));
        X_sub = X(:,idx);
        [B,c] = K_means_L1(X_sub,K,num_ite);
        C(l+1,idx) = c;
        cnt = histc(c,1:num_topics(l));
        for i = 1:size(B,2)
            Tree(end+1).beta_cnt = scale*B(:,i)';
            Tree(end).cnt = scale*cnt(i)/D;
            Tree(end).parent = C(1:l,idx(1))';
            Tree(end).me = [Tree(end).parent i];
        end
        % subtract off mean
        for i = 1:length(c)
            X(:,idx(i)) = X(:,idx(i)) - B(:,c(i));
            X(X(:,idx(i))<0,idx(i)) = 0;
            X(:,idx(i)) = X(:,idx(i))/sum(X(:,idx(i)));
        end
    end
end

% K-Means algorithm with L1 assignment and L2 mean minimization
function [centroids,c] = K_means_L1(X,K,maxite)

D = size(X,2);
if D >= K
    [~,b] = sort(rand(1,D));
    centroids = X(:,b(1:K));
else
    centroids = rand(size(X,1),K);
    centroids = centroids./repmat(sum(centroids,1),size(centroids,1),1);
end
c = zeros(1,D);

for ite = 1:maxite
    for d = 1:D
        [~,c(d)] = min(sum(abs(centroids - repmat(X(:,d),1,K)),1));
    end
    for k = 1:K
        centroids(:,k) = mean(X(:,c==k),2);
    end
end

cnt = histc(c,1:K);
[t1,t2] = sort(cnt,'descend');
centroids = centroids(:,t2);
c2 = zeros(1,length(c));
for i = 1:length(t2)
    idx = find(c == t2(i));
    c2(idx) = i;
end
c = c2;
