function distance = SME_distance(vec1,vec2)
    sigma = 1250;
    distance = exp(-((vec1(1) - vec2(1))^2 + (vec1(2) - vec2(2))^2)/sigma^2);
end