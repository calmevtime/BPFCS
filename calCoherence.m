function [s] = calCoherence(mat,K)
field1 = 'muA'; value1 = 0;
field2 = 'j';   value2 = 0;
field3 = 'k';   value3 = 0;

s = struct(field1,value1,field2,value2,field3,value3);

temp = 0;
for j = 1 : K
    for k = (j + 1) : K
        temp = abs(mat(:,j)' * mat(:,k)) / (norm(mat(:,j)) * norm(mat(:,k)));
        if(temp > s.muA)
            s.muA = temp;
            s.j = j;
            s.k = k;
        end
    end
end

