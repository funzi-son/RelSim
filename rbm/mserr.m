function mse = mserr(X,X_)
%% Compute average mean square error
mse = mean(mean((X - X_).^2,2));
end

