function vocab = load_redis_vocab(input_db, vocab_key)
    reply = command(input_db, 'lrange %s 0 -1', vocab_key);
    if reply.type ~= redisReplyType.ARRAY
        if reply.type == redisReplyType.ERROR
            disp(reply.data)
        else
            disp(reply.type)
        end
        error('unexpected reply to vocab lrange')
    end
    vocab = cell(length(reply.data),1);
    for i=1:length(reply.data)
        vocab{i} = reply.data{i}.data;
    end
end
