function [matrix_train,Wmat] = matrixtrain( train_data,train_target,Num,Conf,C,kernel_type,kernel_para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is designed to train. 
%
%   Syntax
%
%   INPUT:  train_data        - training sample features, N-by-D matrix.
%           train_target      - training sample labels, l-by-N row vector.
%           Num               - this is a number of nearest neighbors.
%           Conf              - L x L matrix of non-equilibrium label completion.
%           C                 - this is a regularization parameter.
%           kernel_type       - this is a kernel type.
%           kernel_para       - this is a kernel parpameter.
%   OUTPUT: matrix_train      - N x L matrix of nearest neighbors feature.
%           Wamt              - N x L matrix of train weight.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [num_class,num_training]=size(train_target);                           

%Computing distance between training instances
    dist_matrix=diag(realmax*ones(1,num_training));                         
    for i=1:num_training-1
        if(mod(i,100)==0)                                                   
            disp(strcat('computing distance for instance:',num2str(i)));
        end
        vector1=train_data(i,:);                                           
        for j=i+1:num_training    
            vector2 = train_data(j,:);
            dist_matrix(i,j) = sum(abs(vector1-vector2));
            dist_matrix(j,i) = dist_matrix(i,j);         
        end
    end
    Neighbors=cell(num_training,1); 
    for i=1:num_training
        [~,index]=sort(dist_matrix(i,:));    
        Neighbors{i,1}=index(1:Num);                                 
    end
    temp=zeros(num_training,num_class);
    for i=1:num_training
        neighbor_labels=[];
        for j=1:Num
            neighbor_labels=[neighbor_labels,train_target(:,Neighbors{i,1}(j))];    
        end
        new_neighbor_labels=Conf*neighbor_labels;
        for j=1:num_class
            temp(i,j)=sum(new_neighbor_labels(j,:)./Num);                          
        end      
    end
    matrix_train = temp;
    
    n = size(train_target,2);
    Omega_train = kernel_matrix(matrix_train,kernel_type,kernel_para);
    Wmat=((Omega_train+speye(n)/C)\(train_target')); 
end

