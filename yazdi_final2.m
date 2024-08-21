clear;
clc;
close all
%% Excel
Input=xlsread('Input19.xlsx');
Target=xlsread('Target19.xlsx');
%% Input - Output
x = Input';
t = Target';
%% Training Function   'trainlm'  'Levenberg-Marquardt'
trainFcn ='backpropagation';
%% Fitting Function   tansig  purelin  logsig satlin
hiddenLayerSize=10;
TF={'tansig','purelin'};
% net=network(numInputs,numLayers,biasConnect,inputConnect,layerConnect,outputConnect)
net=newff(x,t,hiddenLayerSize,TF);
%%
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
%% Divide Data
net.divideFcn = 'dividerand'; 
net.divideMode = 'sample'; 
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
%% Performance Function
net.performFcn = 'mse'; %mae 
%% Train
[net,tr] = train(net,x,t);
%% Network   
y=net(x);
output_test= y';
error=gsubtract(t,y);
Error=error';
performance = perform(net,t,y);
%% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
perform_train=trainTargets';
valTargets = t .* tr.valMask{1};
perform_val=valTargets';
testTargets = t .* tr.testMask{1};
perform_test=testTargets';
trainPerformance=perform(net,trainTargets,y);
valPerformance=perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);
%% View
% view(net)
%% Plots Network
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, ploterrhist(e)
% figure, plotregression(t,y)
% figure, plotfit(net,x,t)
%%  output_test  error
figure
subplot(221)
plot(Target(:,1),'linewidth',1);
hold on
plot(output_test(:,1),':','linewidth',2);
grid minor
xlabel('Iteration')
ylabel('E - MPa')
legend('ABAQUS','ANN','location','northwest')
title('Younge Modulus')
%
subplot(222)
plot(Target(:,2),'linewidth',2);
hold on
plot(output_test(:,2),'linewidth',2);
grid minor
xlabel('Iteration')
ylabel('\sigma_y - MPa')
legend('ABAQUS','ANN')
title('Yield Strength')
%
subplot(223)
plot(Error(:,1),'linewidth',2);
grid minor
xlabel('Iteration')
ylabel('Amp')
title('Error')
xlim auto
ylim auto
subplot(224)
plot(Error(:,2),'linewidth',2);
grid minor
xlabel('Iteration')
ylabel('Amp')
title('Error')
xlim auto
ylim auto
%% Experimental Data
Input_exp=xlsread('InputExp.xlsx');
y_exp=net(Input_exp');
%% Assessment Network
disp('---------------------------------------------------' )
disp('   Assessment Network by Arbitarary Data  ' )
disp('---------------------------------------------------' )
disp(['  S_y / E = ',num2str(y_exp(1))])
disp(' ')
disp(['  Work Hardening = ',num2str(y_exp(2))])
disp('-----------------------------')
%%
IW1=net.IW{1};
LW21=net.LW{2,1};
b1=net.b{1};
b2=net.b{2};
ANNcoef.IW1=IW1;
ANNcoef.LW21=LW21;
ANNcoef.b1=b1;
ANNcoef.b2=b2;
%%
iw=net.IW{1}';
disp(['IW = ',mat2str(iw)])
lw=net.LW{2,1}';
disp(['LW = ',mat2str(lw)])
b1=net.b{1}';
disp(['b1 = [',num2str(b1),']'])
