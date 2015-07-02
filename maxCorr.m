function [val, ind1, ind2] = maxCorr(mat1, mat2)

if ~ismatrix(mat1)
    error('Input must be a matrix');
end

if ~ismatrix(mat2)
    error('Input msut be a matrix')
end

if size(mat1,1) ~= size(mat2,1)
    error('Input must have the same size')
end

innerProd = zeros(size(mat1,2), size(mat2,2));

if isequal(mat1,mat2)
    for i = 1 : size(mat1,2) - 1
        for j = i + 1 : size(mat2,2)
            innerProd(i,j) = sum(abs(mat1(:,i) .* mat2(:,j)));
        end
    end
else
    for i = 1 : size(mat1,2)
        for j = 1 : size(mat2,2)
            innerProd(i,j) = sum(abs(mat1(:,i) .* mat2(:,j)));
        end
    end
end

[val,I] = max(innerProd(:));

[ind1, ind2] = ind2sub(size(innerProd),I);