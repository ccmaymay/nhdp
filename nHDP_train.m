function [Tree] = nHDP_train(X, num_topics, scale, init_size, batch_size, ...
    num_iters, beta0, rho_exp)

if ischar(X)
    fprintf('loading data from file %s...\n', X);
    X = dlmread(X);
end

if ~issparse(X)
    disp('converting data to sparse matrix...');
    X = spconvert(X);
end

if ~exist('num_topics', 'var')
    num_topics = [20 10 5];
end

if ~exist('scale', 'var')
    scale = 100000;
end

if ~exist('init_size', 'var')
    init_size = 2000;
end

if ~exist('batch_size', 'var')
    batch_size = 2000;
end

if ~exist('num_iters', 'var')
    num_iters = 1000;
end

if ~exist('beta0', 'var')
    beta0 = 0.1; % Dirichlet base distribution
end

if ~exist('rho_exp', 'var')
    rho_exp = 0.75;
end

[D, Voc] = size(X);
fprintf('corpus has %d documents spanning %d words\n', D, Voc);
init_size = min(D, init_size);
batch_size = min(D, batch_size);

disp('initializing topics with k-means algorithm...');
[~,b] = sort(rand(1,D));
Tree = nHDP_init(X(b(1:init_size),:),num_topics,scale);
disp('post-processing initialized topics...');
for i = 1:length(Tree)
    if Tree(i).cnt == 0
        Tree(i).beta_cnt(:) = 0;
    end
    vec = gamrnd(ones(1,length(Tree(i).beta_cnt)),1);
    Tree(i).beta_cnt = .95*Tree(i).beta_cnt + .05*scale*vec/sum(vec);
end

disp('estimating topics with variational inference...');
for i = 1:num_iters
    fprintf('beginning variational inference iteration %d/%d...\n', i, num_iters);
    [~,b] = sort(rand(1,D));
    rho = (1+i)^-rho_exp; % step size can also be played with
    Tree = nHDP_step(X(b(1:batch_size),:),Tree,scale,rho,beta0);
end
