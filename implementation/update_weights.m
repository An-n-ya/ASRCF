function [ sample_weights ] = update_weights(sample_loss, prior_weights, frame, params)

% Update the sample weights by solving the quadratic programming problem

% compute number of existing samples
num_samples = min(frame, params.nSamples);

% Set up the QP problem
H = diag(2./(prior_weights(1:num_samples) * params.sample_reg));
sample_loss = sample_loss(1:num_samples);
constraint = -eye(num_samples);
b = zeros(num_samples,1);
Aeq = ones(1,num_samples);
Beq = 1;
options.Display = 'off';

% Do the QP optimization
sample_weights = quadprog(double(H),double(sample_loss),constraint,b,Aeq,Beq,[],[],[],options);

if frame < params.nSamples
    sample_weights = cat(1, sample_weights, zeros(params.nSamples - frame,1));
end;

end

