

function Psi = PCA_Train(X)

[N M] = size(X);

Sigma = (X * X')/M;
  
[Psi, S, V] = svd(Sigma);
