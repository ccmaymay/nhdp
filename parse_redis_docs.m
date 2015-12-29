function [doc_ids, doc_counts, status] = parse_redis_docs(reply)

status = true;
doc_ids = cell(length(reply.data),1);
doc_counts = cell(length(reply.data),1);
for i=1:length(reply.data)
    [doc_ids{i}, doc_counts{i}, status] = parse_doc(reply.data{i}.data);
    if ~status
        return
    end
end

end
