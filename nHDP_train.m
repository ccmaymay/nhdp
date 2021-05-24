function [Tree] = nHDP_train(X, options)

arguments
    X (:,:) {mustBeInteger,mustBeNonnegative}
    options.num_topics (1,:) {mustBeInteger} = [20 10 5]
    options.scale (1,1) {mustBePositive} = 100000
    options.init_size (1,1) {mustBeInteger} = 2000
    options.batch_size (1,1) {mustBeInteger} = 2000
    options.num_iters (1,1) {mustBeInteger} = 1000
    options.beta0 (1,1) {mustBePositive} = 0.1
    options.rho_exp (1,1) {mustBePositive} = 0.75
    options.save_interval (1,1) {mustBeInteger,mustBeNonnegative} = 0
    options.save_path (1,:) {mustBeTextScalar} = 'nhdp.mat'
end

options.init_size = min(size(X, 1), options.init_size);
options.batch_size = min(size(X, 1), options.batch_size);

[D, Voc] = size(X);
fprintf('corpus has %d documents spanning %d words\n', D, Voc);

disp('initializing topics with k-means algorithm...');
[~,b] = sort(rand(1,D));
Tree = nHDP_init(X(b(1:options.init_size),:),options.num_topics,options.scale);
disp('post-processing initialized topics...');
for i = 1:length(Tree)
    if Tree(i).cnt == 0
        Tree(i).beta_cnt(:) = 0;
    end
    vec = gamrnd(ones(1,length(Tree(i).beta_cnt)),1);
    Tree(i).beta_cnt = .95*Tree(i).beta_cnt + .05*options.scale*vec/sum(vec);
end

disp('estimating topics with variational inference...');
for i = 1:options.num_iters
    fprintf('beginning variational inference iteration %d/%d...\n', i, options.num_iters);
    [~,b] = sort(rand(1,D));
    rho = (1+i)^-options.rho_exp; % step size can also be played with
    Tree = nHDP_step(X(b(1:options.batch_size),:),Tree,options.scale,rho,options.beta0);
    if options.save_interval > 0 && mod(i, options.save_interval) == 0
        fprintf('saving tree to %s...\n', options.save_path);
        temp_save_path = [options.save_path, '.', sprintf('%d', randi([0 9], 1, 10))];
        save(temp_save_path, 'Tree');
        movefile(temp_save_path, options.save_path);
    end
end
