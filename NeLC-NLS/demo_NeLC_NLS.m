%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is an examplar file on how the NeLC-NLS [1] program could be used.
%
% [1] CHENG Yu-sheng, ZHAO Da-wei, QIAN Kun.
%     Multi-label Learning for Non-equilibrium Labels Completion in Neighborhood Labels Space. 
%     Pattern Recognition and Artificial Intelligence, 2018, 31(8):740-749. 
%
% Please feel free to contact me (zhaodwahu@163.com), if you have any problem about this programme.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc
%% load data
addpath(genpath('.'));
load('emotion.mat')
%% set parameter
Num=10;                  %Suggest set to 10. The num of nearest neighbors.
s=1;                     %Suggest set to 1. Smoothing parameter.
alpha=0.3;               %Suggest set to [0.1-0.5]. Non-equilibrium parameter.
C=1;                     %Suggest set to 1. This is a regularization parameter.
kernel_para=1.0;         %Suggest set to 1. This is a kernel parpameter.
kernel_type='RBF_kernel';%Suggest set to 'RBF_kernel'. This is a kernel type.

%% the non-equilibrium label completion matrix construction
Conf= NeLC(train_target,alpha,s);
[matrix_train,Wmat] = matrixtrain(train_data,train_target,Num,Conf',C,kernel_type,kernel_para);
[result,Outputs,Pre_Labels] = matrixtest(train_data,train_target,test_data,test_target,Num,Wmat,Conf,kernel_type,kernel_para,matrix_train);