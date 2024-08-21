function  BestValue=ANNcost1(net,xx)
global hiddenLayerSize Input Target TF numIW  numLW numb1 NewNet
x = Input';
t = Target';
Xtr=x(:,1:round(0.7*size(x,2)));
Ytr=t(:,1:round(0.7*size(t,2)));
Xts=x(:,1+round(0.7*size(x,2)):end);
Yts=t(:,1+round(0.7*size(t,2)):end);
%% 
IW0=net.IW{1};
[IWx,IWy]=size(IW0);
numIW=IWx*IWy;
IW_1=xx(1:numIW);
IW=reshape(IW_1,IWx,IWy);
net.IW{1}=IW;
%
LW=net.LW{2,1};
[LWx,LWy]=size(LW);
numLW=LWx*LWy;
LW=xx(numIW+1:numIW+numLW);
LW=reshape(LW,LWx,LWy);
net.LW{2}=LW;
%
b1=net.b{1};
[b1x,b1y]=size(b1);
numb1=b1x*b1y;
b1=xx(numIW+numLW+1:numIW+numLW+numb1);
b1=reshape(b1,b1x,b1y);
net.b{1}=b1;
%
b2=net.b{2};
[b2x,b2y]=size(b2);
numb2=b2x*b2y;
b2=xx(numIW+numLW+numb1+1:end);
b2=reshape(b2,b2x,b2y);
net.b{2}=b2;
%% input ANN 
net=newff(x,t,hiddenLayerSize,TF);
trainFcn ='backpropagation';
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
net.divideFcn = 'dividerand'; 
net.divideMode = 'sample'; 
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.performFcn = 'mse';
%% Train
[NewNet,tr] = train(net,x,t);
%% Network   
y=NewNet(x);
output_test= y';
error=gsubtract(t,y);
Error=error';
performance = perform(NewNet,t,y);
%% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
perform_train=trainTargets';
valTargets = t .* tr.valMask{1};
perform_val=valTargets';
testTargets = t .* tr.testMask{1};
perform_test=testTargets';
trainPerformance=perform(net,trainTargets,y);
valPerformance=perform(net,valTargets,y);
testPerformance=perform(net,testTargets,y);
%% View
% view(net)
%% constriant
Ytr_net=sim(NewNet,Xtr);
Yts_net=sim(NewNet,Xts);
% type 2
Err_tr=Ytr_net-Ytr;
Err_ts=Yts_net-Yts;
MaxTr=max(max(Err_tr));
MaxTs=max(max(Err_ts));
BestValue=0.5*(MaxTr+MaxTs);
%
% mse_tr=mse(Ytr_net-Ytr);
% mse_ts=mse(Yts_net-Yts);
% BestValue=0.5*(mse_tr+mse_ts);
end