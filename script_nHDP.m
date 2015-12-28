% initialize / use a large subset of documents (e.g., 10,000) contained in Xid and Xcnt to initialize
%num_topics = [20 10 5];
name = 'giga-deps'
num_topics = [9 7 5]
level_num_topics = 1;
total_num_topics = 0;
for i=1:length(num_topics)
    level_num_topics = level_num_topics * num_topics(i);
    total_num_topics += level_num_topics;
end
total_num_topics
batch_size = 2000
addpath ../redis-matlab/src
input_host = 'compute0054'
input_port = 61236
input_db = redisConnection(input_host, input_port);
data_key = 'giga:pll:agg:10k/bow'
vocab_key = 'giga:pll:agg:10k/vocab'
D = redisScard(input_db, data_key)
W = redisLLen(input_db, vocab_key)
doc_scale = 100
scale = D * doc_scale / total_num_topics
Tree = nHDP_init(Xid,Xcnt,num_topics,scale);
for i = 1:length(Tree)
    if Tree(i).cnt == 0
        Tree(i).beta_cnt(:) = 0;
    end
    vec = gamrnd(ones(1,length(Tree(i).beta_cnt)),1);
    Tree(i).beta_cnt = .95*Tree(i).beta_cnt + .05*scale*vec/sum(vec);
end

% main loop / to modify this, at each iteration send in a new subset of docs
% contained in Xid_batch and Xcnt_batch
beta0 = .1 % this parameter is the Dirichlet base distribution and can be played with
i = 1;
while true
    disp(['iteration ' num2str(i)])
    if isinteger(log2(i))
        path = sprintf('%s-%d.mat', name, i);
        disp(['saving to ' path])
        save(path, '-v7.3', name, num_topics, scale, beta0, i, Tree)
    end
    %[a,b] = sort(rand(1,length(Xid)));
    rho = (64+i)^-.6; % step size can also be played with
    redisSRandmember(input_db, data_key, batch_size);
    %Xid_batch = Xid(b(1:2000));
    %Xcnt_batch = Xcnt(b(1:2000));
    Tree = nHDP_step(Xid_batch,Xcnt_batch,Tree,scale,rho,beta0);
    i = i + 1;
end
