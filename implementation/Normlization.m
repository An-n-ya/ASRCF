function entropy=Normlization(X)
    FlattenedData = X(:)'; 
    A = mapminmax(FlattenedData, 1, 9);
    SUM = sum(A);
    A = A/SUM;
    entropy = -sum(A.*log(A));
end
