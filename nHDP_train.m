function [Tree] = nHDP_train(X, model_params, alg_params)

arguments
    X (:,:) {mustBeInteger,mustBeNonnegative}
    model_params.num_topics (1,:) {mustBeInteger} = [20 10 5]
    model_params.lambda0 (1,1) {mustBePositive} = 0.1 % topic Dirichlet hyperparameter
    model_params.alpha (1,1) {mustBePositive} = 5 % top-level (global) DP concentration
    model_params.beta (1,1) {mustBePositive} = 1 % second-level (local) DP concentration
    model_params.gamma1 (1,1) {mustBePositive} = 2*(1/3) % switching DP stopping hyperparameter
    model_params.gamma2 (1,1) {mustBePositive} = 2*(2/3) % switching DP continuing hyperparameter
    alg_params.init_size (1,1) {mustBeInteger,mustBeNonnegative} = 2000
    alg_params.batch_size (1,1) {mustBeInteger,mustBePositive} = 2000
    alg_params.num_iters (1,1) {mustBeInteger,mustBePositive} = 1000
    alg_params.save_interval (1,1) {mustBeInteger,mustBeNonnegative} = 0
    alg_params.save_path (1,:) {mustBeTextScalar} = 'nhdp.csv'
    alg_params.start_iter (1,1) {mustBeInteger,mustBePositive} = 1
    alg_params.start_tree (1,:) = []
    alg_params.rho_exp (1,1) {mustBePositive} = 0.75
    alg_params.rho_base_offset (1,1) {mustBePositive} = 1
    alg_params.kappa (1,1) {mustBeNonnegative,mustBeLessThanOrEqual(alg_params.kappa,1)} = 0.5
    alg_params.init_scale (1,1) {mustBeNonnegative} = 0
    alg_params.batch_scale (1,1) {mustBeNonnegative} = 0
    alg_params.init_num_iters (1,1) {mustBeInteger,mustBePositive} = 3
    alg_params.init_rand_scale (1,1) {mustBeNonnegative} = 100
end

alg_params.init_size = min(size(X, 1), alg_params.init_size);
alg_params.batch_size = min(size(X, 1), alg_params.batch_size);

[D, Voc] = size(X);
fprintf('corpus has %d documents spanning %d words\n', D, Voc);

if alg_params.init_scale == 0
    alg_params.init_scale = D;
end

if alg_params.batch_scale == 0
    alg_params.batch_scale = D;
end

if alg_params.init_size > 0
    disp('initializing topics with k-means algorithm...');
    [~,b] = sort(rand(1,D));
    Tree = nHDP_init(X(b(1:alg_params.init_size),:),model_params,alg_params);
else
    if isempty(alg_params.start_tree)
        error('init_size is 0 but start_tree is not specified');
    end
    Tree = alg_params.start_tree;
end

disp('estimating topics with variational inference...');
for i = alg_params.start_iter:alg_params.num_iters
    fprintf('beginning variational inference iteration %d/%d...\n', i, alg_params.num_iters);
    [~,b] = sort(rand(1,D));
    rho = (alg_params.rho_base_offset+i)^-alg_params.rho_exp; % step size can also be played with
    Tree = nHDP_step(X(b(1:alg_params.batch_size),:),Tree,model_params,alg_params,rho);
    if alg_params.save_interval > 0 && mod(i, alg_params.save_interval) == 0
        temp_save_path = [alg_params.save_path, '.', sprintf('%d', randi([0 9], 1, 10))];
        if endsWith(alg_params.save_path, '.csv', 'IgnoreCase', true)
            fprintf('saving tree to CSV file %s...\n', alg_params.save_path);
            write_tree_csv(Tree, temp_save_path);
        else
            fprintf('saving tree and parameters to MAT file %s...\n', alg_params.save_path);
            save(temp_save_path, 'Tree', 'model_params', 'alg_params');
        end
        movefile(temp_save_path, alg_params.save_path);
    end
end
