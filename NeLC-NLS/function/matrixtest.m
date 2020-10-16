function [result,Outputs,Pre_Labels] = matrixtest(train_data,train_target,test_data,test_target,Num,Wmat,Conf,kernel_type,kernel_para,matrix_train)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is designed to test. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[num_class,num_training]=size(train_target);  
[~,num_testing]=size(test_target); 
dist_matrix=diag(realmax*ones(1,num_training)); 
  
    for i=1:num_testing
        vector1 = test_data(i,:);
        for j=1:num_training
             vector2 = train_data(j,:);
             dist_matrix(i,j) = sum(abs(vector1-vector2));
        end
    end
    
    Neighbors=cell(num_testing,1); 
    for i=1:num_testing
        [~,index]=sort(dist_matrix(i,:));    
        Neighbors{i,1}=index(1:Num);                                  
    end
    
    temp=zeros(num_testing,num_class);
    for i=1:num_testing
        neighbor_labels=[];
        for j=1:Num
            neighbor_labels=[neighbor_labels,train_target(:,Neighbors{i,1}(j))];  
        end  
            new_neighbor_labels=Conf*neighbor_labels;            
        for j=1:num_class
            temp(i,j)=sum(new_neighbor_labels(j,:)./Num);                                  
        end
    end
    matrix_test = temp;
   
    Omega_test = kernel_matrix(matrix_train,kernel_type,kernel_para,matrix_test);
    Outputs=(Omega_test'*Wmat)';  %   Outputs: the actual output of the testing data
    
    %Evaluation
    Pre_Labels=zeros(num_class,num_testing);
    for i=1:num_testing
        for j=1:num_class
            if(Outputs(j,i)>0)
                Pre_Labels(j,i)=1;
            else
                Pre_Labels(j,i)=-1;
            end
        end
    end
    
    result.HammingLoss=Hamming_loss(Pre_Labels,test_target);
    result.RankingLoss=Ranking_loss(Outputs,test_target);
    result.OneError=One_error(Outputs,test_target);
    result.Coverage=coverage(Outputs,test_target);
    result.Average_Precision=Average_precision(Outputs,test_target);
end

