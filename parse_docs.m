function [doc_ids, doc_counts, status] = parse_docs(s)

status = true;
doc_ids = cell(length(s),1);
doc_counts = cell(length(s),1);
for i=1:length(s)
    [doc_ids{i}, doc_counts{i}, status] = parse_doc(s{i});
    if ~status
        return
    end
end

end
