D = 10000;
N = 100;
Xid = cell(D,1);
Xcnt = cell(D,1);
for d=1:D
    t = (rand > 0.5);
    p = rand;
    if t
        Xid{d} = [2, 3];
    else
        Xid{d} = [1, 2];
    end
    Xcnt{d} = ceil([p, 1 - p] * N);
end
