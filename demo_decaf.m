clear all;
% Set algorithm parameters
options.k = 100;      % subspace dimension
options.r= 1;    % control the effect of regularization term 
options.rho = 0.1;    % control the effect of intra and inter class discrimination within domain 
options.alpha = 1.0;  % control the effect of intra-domain and inter-domain discrimination
options.ker = 'linear';     % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;        % kernel bandwidth
options.beta = 1.5;   % control consisttent discrimination
T = 10;

result = [];
srcStr={'caltech_decaf.mat','caltech_decaf.mat','caltech_decaf.mat',...
    'amazon_decaf','amazon_decaf','amazon_decaf',...
    'webcam_decaf.mat','webcam_decaf.mat','webcam_decaf.mat',...
    'dslr_decaf.mat','dslr_decaf.mat','dslr_decaf.mat'};
tgtStr ={'amazon_decaf','webcam_decaf.mat','dslr_decaf.mat',...
    'caltech_decaf.mat','webcam_decaf.mat','dslr_decaf.mat',...
    'caltech_decaf.mat','amazon_decaf','dslr_decaf.mat',...
    'caltech_decaf.mat','amazon_decaf','webcam_decaf.mat'};
for iData =1:12
    
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    load(strcat('/home/wangjie/data/Office+Caltech DeCAF/',src));
    Xs = feas';   % dim*n
    Xs = Xs*diag(sparse(1./sqrt(sum(Xs.^2))));
    Ys = labels;  %n*1
    load(strcat('/home/wangjie/data/Office+Caltech DeCAF/',tgt));
    Xt = feas';
    Xt = Xt*diag(sparse(1./sqrt(sum(Xt.^2))));
    Yt = labels;
    Class_model = fitcknn(Xs',Ys);   
    Cls = Class_model.predict(Xt');
    acc = length(find(Cls==Yt))/length(Yt); fprintf('NN=%0.4f\n',acc);

    % AACD evaluation
    Cls = [];
    Ytt = [];
    Acc = []; 
    for t = 1:T
        options.can=t/T;
        fprintf('==============================Iteration [%d]==============================\n',t);
        [Z,A] = AACD(Xs,Xt,Ys,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        Class_model = fitcknn(Zs',Ys);      
        Cls = Class_model.predict(Zt');
        Cls = TSRP(Zs',Ys,Zt',Cls,0.85,3);
        acc = length(find(Cls==Yt))/length(Yt); fprintf('AACD+NN=%0.4f\n',acc);
        Acc = [Acc;acc];
    end
    result = [result;Acc(end)];
    fprintf('\n\n\n');
end
result_aver=mean(result);
Result=[result;result_aver]*100
