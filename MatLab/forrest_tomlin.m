function [L_up, U_up] = forrest_tomlin(Aq,q,L,U)

n = size(L,1);

%======================Update Lower Triangular (U)=============
w = L\Aq;
L_inv_B_p = U;
L_inv_B_p (:,q) = w;

%======================Cyclic Permutation======================
P_r = eye(n);
P_r = [P_r(1:q-1,:);P_r(q+1:end,:);P_r(q,:)];
P_c = P_r';

P_L_inv_B_p = P_r*L_inv_B_p*P_c;

%==============LU Calculation GE================
L_l = []; U_one=P_L_inv_B_p; L_one = eye(n); w_n = P_L_inv_B_p(n,n);
for i=2:n-1
    if(U(n,i) == 0)
        L_l = [L_l 0];
    else
        L_l = [L_l U(n,i)/U(i,i)];
        
        U_one(n,:) = U_one(n,:) - U(i,:)*U(n,i)/U(i,i);
    end
end
    
L_one(n,:) = [0 L_l 1];


L_up = L*P_c*L_one;
U_up = U_one*P_r;

end










