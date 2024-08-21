function  BestValue=ANNcost(Network,xx)
%% Excel
Input=xlsread('Input19.xlsx');
Target=xlsread('Target19.xlsx');
%% Input - Output
x = Input';
t = Target';
Xtr=x(:,1:round(0.7*size(x,2)));
Ytr=t(:,1:round(0.7*size(t,2)));
Xts=x(:,1+round(0.7*size(x,2)):end);
Yts=t(:,1+round(0.7*size(t,2)):end);
%% 
IW=net.IW{1};
[IWx,IWy]=size(IW);
numIW=IWx*IWy;
IW=xx(1:numIW);
IW=reshape(IW,IWx,IWy);
net.IW{1}=IW;
%
LW=net.LW{2,1};
[LWx,LWy]=size(LW);
numLW=LWx*LWy;
LW=xx(numIW+1:numIW+numLW);
LW=reshape(LW,LWx,LWy);
net.LW{2,1}=LW;
%
b1=net.b{1};
[b1x,b1y]=size(b1);
numb1=b1x*b1y;
b2=xx(numIW+numLW+1:numIW+numLW+numb1);
b2=reshape(b2,b1x,b1y);
net.b{1}=b2;
%
b2=net.b{2};
[b2x,b2y]=size(b2);
numb2=b2x*b2y;
b1=xx(numIW1+numLW21+numb1+1:numIW1+numLW21+numb1+numb2);
b1=reshape(b1,b2x,b2y);
net.b{2}=b1;
%% input ANN
hiddenLayerSize=100;
TF={'tansig','purelin'};
net=newff(x,t,hiddenLayerSize,TF);
trainFcn ='backpropagation';
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};0
net.divideFcn = 'dividerand'; 
net.divideMode = 'sample'; 

net.performFcn = 'mse';
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
%% constriant
Ytr_net=sim(net,Xtr);
Yts_net=sim(net,Xts);
%
% Err_tr=Ytr_net-Ytr;
% Err_ts=Yts_net-Yts;
% MinTr=min(Err_tr);
% MinTs=min(Err_ts);
%
mse_tr=mse(Ytr_net-Ytr);
mse_ts=mse(Yts_net-Yts);

end