% initialize / use a large subset of documents (e.g., 10,000) contained in Xid and Xcnt to initialize
%num_topics = [20 10 5];
name = 'giga-deps';
num_topics = [9 7 5];
iota = 64;
kappa = 0.6;
beta0 = .1; % this parameter is the Dirichlet base distribution and can be played with
batch_size = 2000;
init_rand_frac = 1/20;
max_init_batch_size = 10000;

addpath /scratch/groups/bvandur1/redis-mat
input_host = 'compute0054';
input_port = 61236;
input_db = redis(input_host, input_port);
data_key = 'giga:pll:agg:10k/bow';
vocab_key = 'giga:pll:agg:10k/vocab';

level_num_topics = 1;
total_num_topics = 0;
for i=1:length(num_topics)
    level_num_topics = level_num_topics * num_topics(i);
    total_num_topics = total_num_topics + level_num_topics;
end

reply = command(input_db, 'scard %s', data_key);
if reply.type ~= redisReplyType.INTEGER || reply.data == 0
    if reply.type == redisReplyType.ERROR
        disp(reply.data)
    else
        disp(reply.type)
    end
    error('unexpected reply to dataset scard')
end
D = reply.data;

reply = command(input_db, 'llen %s', vocab_key);
if reply.type ~= redisReplyType.INTEGER || reply.data == 0
    if reply.type == redisReplyType.ERROR
        disp(reply.data)
    else
        disp(reply.type)
    end
    error('unexpected reply to vocab llen')
end
W = reply.data;

init_batch_size = min(D, max_init_batch_size);

disp(sprintf('loading %d initialization documents...', init_batch_size));
reply = command(input_db, 'srandmember %s %d', data_key, init_batch_size);
if reply.type ~= redisReplyType.ARRAY
    if reply.type == redisReplyType.ERROR
        disp(reply.data)
    else
        disp(reply.type)
    end
    error('unexpected reply to dataset srandmember')
end
[Xid, Xcnt, status] = parse_redis_docs(reply);
if ~status
    error('document parse error')
end

%doc_scale = 100;
doc_scale = mean(cellfun(@sum, Xcnt));

disp('initializing...')
scale = D * doc_scale / total_num_topics;
tic
Tree = nHDP_init(Xid,Xcnt,num_topics,scale);
toc
for i = 1:length(Tree)
    if Tree(i).cnt == 0
        Tree(i).beta_cnt(:) = 0;
    end
    vec = gamrnd(ones(1,length(Tree(i).beta_cnt)),1);
    Tree(i).beta_cnt = (1 - init_rand_frac)*Tree(i).beta_cnt + init_rand_frac*scale*vec/sum(vec);
end

i = 1;
loop_nHDP;
