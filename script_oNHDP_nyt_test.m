cd('../nyt');
load mult_test;
cd('../oNHDP');

for i = 1:length(Xid_test)
    Xid_test{i} = Xid_test{i} + 1;
end

D = length(Xid_test);
perctest = .1;
Xid = Xid_test;
Xcnt = Xcnt_test;
Xcnt_test = cell(1,D);
Xid_test = cell(1,D);
for d = 1:D
    numW = sum(Xcnt{d});
    numTest = floor(perctest*numW);
    [a,b] = sort(rand(1,numW));
    wordVec = [];
    for i = 1:length(Xid{d})
        wordVec = [wordVec Xid{d}(i)*ones(1,Xcnt{d}(i))];
    end
    wordTestVec = wordVec(b(1:numTest));
    wordTrainVec = wordVec(b(numTest+1:end));
    Xid{d} = unique(wordTrainVec);
    Xcnt{d} = histc(wordTrainVec,Xid{d});
    Xid_test{d} = unique(wordTestVec);
    Xcnt_test{d} = histc(wordTestVec,Xid_test{d});
end
num_test = 0;
for i = 1:length(Xcnt_test)
    num_test = num_test + sum(Xcnt_test{i});
end

results_mean = zeros(1,360);
for i = 270:10:360
    tic
    cd stored_nyt
    load(['nHDP_step_nyt_' num2str(i) '.mat']);
    cd ..
    [llik_mean,C_d] = nHDP_test(Xid_test,Xcnt_test,Xid,Xcnt,Tree,.1);
    results_mean(i) = sum(llik_mean)/num_test;
    save oNHDP_nyt_test results_mean;
    disp(['Finished ' num2str(i) ' : ' num2str(toc/60)]);
end
