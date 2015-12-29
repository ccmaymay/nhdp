function [doc_ids, doc_counts, status] = parse_doc(s)

doc_ids = [];
doc_counts = [];
status = false;

if length(s) < 1
    return
end

hashes = findstr('#', s);

if length(hashes) < 1
    return
end

hash = hashes(1);

spaces = findstr(' ', s);
spaces = spaces(spaces < hash);
colons = findstr(':', s);

num_types = length(spaces);
doc_ids = zeros(1,num_types,'uint64');
doc_counts = zeros(1,num_types);

start = 1;
for i=1:num_types
    doc_ids(i) = str2num(s(start:(colons(i)-1))) + 1;
    doc_counts(i) = str2num(s((colons(i)+1):(spaces(i)-1)));
    start = spaces(i) + 1;
end

status = true;

end
