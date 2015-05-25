function [A,W,B,S] = alphaLDA(X,labels,lambdav)
% Implementation of LDA-based hybrid metric learning
% Author: Brian D. Bue (bbue@rice.edu)
% Last modified: 4/16/12
%    
% Arguments: 
%    - X: cell array of D representations of samples, each X{i} an N x d matrix
%    - labels: list of N labels for each sample in X{i}
%    
% Keyword arguments:
%    - lambdav: regularization parameter (\in [0,1])
%    
% Returns:
%    - A: $\alpha$ coefficients        
%

D = length(X);
ulab = unique(labels);
k = length(ulab);
N = length(labels);

B = zeros(D);
W = zeros(D);

for i=1:D
  Xi = X{i};
  Mi = class_means(Xi,labels);
  Mui = mean(Mi,1);
  for j=i+1:D
    Xj = X{j};
    Mj = class_means(Xj,labels);
    Muj = mean(Mj,1);
    
    Wii = 0; Wjj = 0; Wij = 0; Ni = [];
    for l=1:k
      class_mask = labels==ulab(l);
      Ni = [Ni; sum(class_mask)];
      Wdisti = distanceMatrix(Xi(class_mask,:)',Mi(l,:)');
      Wdistj = distanceMatrix(Xj(class_mask,:)',Mj(l,:)');
      Wii = Wii+sum(Wdisti.^2);
      Wjj = Wjj+sum(Wdistj.^2);
      Wij = Wij+sum(Wdisti.*Wdistj);
    end    

    
    Bdisti = distanceMatrix(Mi',Mui');
    Bdistj = distanceMatrix(Mj',Muj');
    Bii = sum(Ni.*(Bdisti.^2));
    Bjj = sum(Ni.*(Bdistj.^2));
    Bij = sum(Ni.*(Bdisti.*Bdistj));

    
    W(i,j) = Wij;    W(j,i) = Wij;
    B(i,j) = Bij;    B(j,i) = Bij;
    
    W(i,i) = Wii;    W(j,j) = Wjj;
    B(i,i) = Bii;    B(j,j) = Bjj;

  end
end

B = B ./ N;
W = W ./ N;

W = (1-lambdav)*W + (lambdav*eye(D));
[eigvecs,eigvals] = eig(inv(W)*B);
[maxv,maxi] = max(diag(eigvals));

Ao = eigvecs(:,maxi)';

A = Ao/sum(Ao);
S = (A*B*A') / (A*W*A');

function dmtx = distanceMatrix(A,B)
% squared Euclidean distance matrix between 
% A (size=[d,N]) and B (size=[d,M])
AA=sum(A.*A,1); BB=sum(B.*B,1); ab=A'*B; 
dmtx = repmat(AA',[1 size(BB,2)]) + repmat(BB,[size(AA,2) 1]) - 2*AB;
