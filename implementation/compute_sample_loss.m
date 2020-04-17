function [ sample_loss ] = compute_sample_loss(hf_reshaped, samplesf, yf_vec_perm, dft_part_sz)

% Compute the training loss for each sample

support_sz = dft_part_sz(3);

corr_train = mtimesx(samplesf, permute(hf_reshaped,[2 3 1]), 'speed');
corr_error = bsxfun(@minus,corr_train,yf_vec_perm);
sample_loss = 1 / support_sz * (2*real(sum(corr_error .* conj(corr_error),3)) - real(sum(corr_error(:,1,1:dft_part_sz(1)) .* conj(corr_error(:,1,1:dft_part_sz(1))),3)));
