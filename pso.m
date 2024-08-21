clc;
clear;
close all;
%% Problem Definition
global hiddenLayerSize  Input Target TF numIW  numLW numb1 NewNet
%% Excel
Input=xlsread('Input19.xlsx');
Target=xlsread('Target19.xlsx');
%% Input - Output
hiddenLayerSize=10;
TF={'tansig','purelin'};
FirstNet=newff(Input',Target',hiddenLayerSize,TF);
%%
CostFunction=@(x) ANNcost1(FirstNet,x);       %Sphere(x) %YoungModule(x)% Cost Function
nVar=62;             % Number of Decision Variables
VarSize=[1 nVar];   % Size of Decision Variables Matrix
VarMin=-100;         % Lower Bound of Variables
VarMax= 100;         % Upper Bound of Variables
%% PSO Parameters
MaxIt=10;      % Maximum Number of Iterations
nPop=200;        % Population Size (Swarm Size)
% w=1;            % Inertia Weight
% wdamp=0.99;     % Inertia Weight Damping Ratio
% c1=2;           % Personal Learning Coefficient
% c2=2;           % Global Learning Coefficient
% Constriction Coefficients
phi1=2.05;
phi2=2.05;
phi=phi1+phi2;
chi=2/(phi-2+sqrt(phi^2-4*phi));
w=chi;          % Inertia Weight
wdamp=1;        % Inertia Weight Damping Ratio
c1=chi*phi1;    % Personal Learning Coefficient
c2=chi*phi2;    % Global Learning Coefficient
% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;
%% Initialization
empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
particle=repmat(empty_particle,nPop,1);
GlobalBest.Cost=inf;
for i=1:nPop
    particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    particle(i).Velocity=zeros(VarSize);
    particle(i).Cost=CostFunction(particle(i).Position);
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    if particle(i).Best.Cost<GlobalBest.Cost
        GlobalBest=particle(i).Best;        
    end    
end
BestCost=zeros(MaxIt,1);
nfe=zeros(MaxIt,1);
%% PSO Main Loop
for it=1:MaxIt   
    for i=1:nPop     
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        particle(i).Cost = CostFunction(particle(i).Position);
        if particle(i).Cost<particle(i).Best.Cost
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            if particle(i).Best.Cost<GlobalBest.Cost
                GlobalBest=particle(i).Best;
            end
        end
    end
    BestCost(it)=GlobalBest.Cost;
    disp(['Iteration ' num2str(it) ' :: ' 'Best Cost = ' num2str(BestCost(it))]);
    w=w*wdamp; 
    ANN(it).coef=particle(it).Best.Position';
    ANN(it).NetWork=NewNet;
end
%% Results
plot(BestCost,'rp');
xlabel('Iteration');
ylabel('Average MSE Train and Test');
grid minor
%% Optimal ANN   
net.IW{1}=ANN(end).coef(1:numIW);
net.LW{2}=ANN(end).coef(numIW+1:numIW+numLW);
net.b{1}=ANN(end).coef(numIW+numLW+1:numIW+numLW+numb1);
net.b{2}=ANN(end).coef(numIW+numLW+numb1+1:end);
%% Experimental Data
BestResult=ANN(end).NetWork(Input');
Input_exp=xlsread('InputExp.xlsx');
EvalExp=ANN(end).NetWork(Input_exp');
%% Evaluation Optimal ANN
figure
subplot(221)
plot(Target(:,1))
hold on
plot(BestResult(1,:))
hold on
plot(EvalExp(1),'kp')
grid minor
xlabel('Iteration')
ylabel('\sigma_y / E')
legend('ABAQUS','ANN- PSO','Exp')
title('Young Modulus')
subplot(222)
plot(Target(:,2))
hold on
plot(BestResult(2,:))
hold on
plot(EvalExp(2),'kp')
xlabel('Iteration')
ylabel('n')
grid minor
legend('ABAQUS','ANN- PSO','Exp')
title('Work Hardening')
subplot(223)
plot(Target(:,1)-BestResult(1,:)')
grid minor
xlabel('Iteration')
ylabel('Err')
subplot(224)
plot(Target(:,2)-BestResult(2,:)')
xlabel('Iteration')
ylabel('Err')
grid minor

