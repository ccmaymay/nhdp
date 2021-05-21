% initialize / use a large subset of documents (e.g., 10,000) contained in Xid and Xcnt to initialize
num_topics = [20 10 5];
scale = 100000;
init_size = 2000;
batch_size = 2000;
num_iters = 1000;
Voc = 0;
for d = 1:length(Xid)
    Voc = max(Voc,max(Xid{d}));
end
[~,b] = sort(rand(1,length(Xid)));
Xid_batch = Xid(b(1:init_size));
Xcnt_batch = Xcnt(b(1:init_size));
Tree = nHDP_init(Xid_batch,Xcnt_batch,num_topics,scale,Voc);
for i = 1:length(Tree)
    if Tree(i).cnt == 0
        Tree(i).beta_cnt(:) = 0;
    end
    vec = gamrnd(ones(1,length(Tree(i).beta_cnt)),1);
    Tree(i).beta_cnt = .95*Tree(i).beta_cnt + .05*scale*vec/sum(vec);
end

% main loop / to modify this, at each iteration send in a new subset of docs
% contained in Xid_batch and Xcnt_batch
beta0 = .1; % this parameter is the Dirichlet base distribution and can be played with
for i = 1:num_iters
    [~,b] = sort(rand(1,length(Xid)));
    rho = (1+i)^-.75; % step size can also be played with
    Xid_batch = Xid(b(1:batch_size));
    Xcnt_batch = Xcnt(b(1:batch_size));
    Tree = nHDP_step(Xid_batch,Xcnt_batch,Tree,scale,rho,beta0);
end
