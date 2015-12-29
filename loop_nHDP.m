% main loop / to modify this, at each iteration send in a new subset of docs
% contained in Xid_batch and Xcnt_batch
addpath /scratch/groups/bvandur1/redis-mat
input_db = redis(input_host, input_port);
appendCommand(input_db, 'srandmember %s %d', data_key, batch_size);
while true
    disp(['iteration ' num2str(i)])
    log2i = log2(i);
    if ceil(log2i) == log2i
        path = sprintf('%s-%d.mat', name, i);
        disp(['saving to ' path])
        save(path, '-v7.3', 'name', 'num_topics', 'scale', 'rho', 'beta0', 'iota', 'kappa', 'input_host', 'input_port', 'data_key', 'vocab_key', 'batch_size', 'i', 'Tree');
    end

    %[a,b] = sort(rand(1,length(Xid)));
    %Xid_batch = Xid(b(1:2000));
    %Xcnt_batch = Xcnt(b(1:2000));

    reply = getReply(input_db);
    if reply.type ~= redisReplyType.ARRAY
        if reply.type == redisReplyType.ERROR
            disp(reply.data)
        else
            disp(reply.type)
        end
        error('unexpected reply to dataset srandmember')
    end
    [Xid_batch, Xcnt_batch, status] = parse_redis_docs(reply);
    if ~status
        error('document parse error')
    end
    appendCommand(input_db, 'srandmember %s %d', data_key, batch_size);

    tic
    rho = (iota+i)^-kappa; % step size can also be played with
    Tree = nHDP_step(Xid_batch,Xcnt_batch,Tree,scale,rho,beta0);
    i = i + 1;
    toc
end
