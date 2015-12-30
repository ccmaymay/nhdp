function run_nHDP(name, input_host, input_port, data_key, vocab_key, num_topics, batch_size, beta0, iota, kappa, init_rand_frac, max_init_batch_size)
    % initialize / use a large subset of documents (e.g., 10,000) contained in Xid and Xcnt to initialize

    if nargin < 6
        num_topics = [9 7 5];
    end

    if nargin < 7
        batch_size = 2000;
    end

    if nargin < 8
        beta0 = 0.1;
    end

    if nargin < 9
        iota = 64;
    end

    if nargin < 10
        kappa = 0.6;
    end

    if nargin < 11
        init_rand_frac = 1/20;
    end

    if nargin < 12
        max_init_batch_size = 10000;
    end

    addpath /scratch/groups/bvandur1/redis-mat
    input_db = redis(input_host, input_port);

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

    loop_nHDP(name, input_host, input_port, data_key, vocab_key, num_topics, batch_size, scale, rho, beta0, iota, kappa, Tree);
end
