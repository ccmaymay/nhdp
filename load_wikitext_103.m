X = dlmread('/media/cjmay/Data1/wikitext-103/train_tokens.txt');
Xs = sparse(X(:,1) + 1, X(:,2) + 1, X(:,3));
Xid = cell(max(X(:,1)), 1);
Xcnt = cell(length(Xid), 1);
for i=1:length(Xid)
    Xid{i} = find(Xs(i,:));
    Xcnt{i} = nonzeros(Xs(i,:));
end
