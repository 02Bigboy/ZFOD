function [Z,A] = AACD(Xs,Xt,Ys,Yt0,options)
% Thanks to DICD for open source 
% Li S, Song S, Huang G, et al. Domain invariant and class discriminative feature learning for visual domain adaptation[J]. IEEE transactions on image processing, 2018, 27(9): 4260-4273.
% Xs, Xt: Sample source and target domains
% Ys: labels of source domain
% Yto: pseudo labels of target domian
if nargin < 5
    error('Algorithm parameters should be set!');
end
if ~isfield(options,'k')
    options.k = 100;
end
if ~isfield(options,'alpha')
    options.alpha = 1.0;
end
if ~isfield(options,'beta')
    options.beta = 1.5;
end
if ~isfield(options,'r')
    options.r = 1;
end
if ~isfield(options,'ker')
    options.ker = 'primal';
end
if ~isfield(options,'gamma')
    options.gamma = 1;
end
if ~isfield(options,'data')
    options.data = 'default';
end
k = options.k;
alpha = options.alpha;
r = options.r;
rho=options.rho;
ker = options.ker;
gamma = options.gamma;
data = options.data;
pb = options.beta;

fprintf('AACD:  data=%s  k=%d  alpha=%f  gamma=%f r=%f beta=%f \n',data,k,alpha,gamma,r, pb);

X = [Xs,Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
[m,n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);
Xs=X(:,1:ns);
Xt=X(:,ns+1:end);
C = length(unique(Ys));

Yall=[Ys;Yt0];


% Construct MMD matrix
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e*e'*C;
% M = e*e';
M_cs = zeros(n,n);
Mcd = zeros(n,n);
if ~isempty(Yt0) && length(Yt0)==nt
    c_record = [];
    for c = reshape(unique(Ys),1,C)
        e = zeros(n,1);
        e(Ys==c) = 1/length(find(Ys==c));
        e(ns+find(Yt0==c)) = -1/length(find(Yt0==c));
        e(isinf(e)) = 0;
        M = M + e*e';                                     % M_intra
        c_record = [c_record,c];
        ecd = zeros(n,1);
        ecd(Ys==c) = 1/length(find(Ys==c));
        for c_other = setdiff(reshape(unique(Ys),1,C), c_record)
            e(Ys==c_other) = -1/length(find(Ys==c_other));
            e(ns+find(Yt0==c_other)) = 1/length(find(Yt0==c_other));
            e(isinf(e)) = 0;
            M_cs = M_cs +e*e';                            % M_inter
            e(Ys==c_other) = 0;
            e(ns+find(Yt0==c_other)) = 0;
            ecd(ns+find(Yt0==c_other)) = -1/length(find(Yt0==c_other));
            ecd(isinf(e)) = 0;
        end   
        Mcd = Mcd + ecd*ecd';                             % S consistent discrimination matrix
        Mcd(Ys==c,Ys==c) = (C-1)*Mcd(Ys==c,Ys==c);
        YtColumn=singlelbs2multilabs(Yt0,C);
        matrix = YtColumn*YtColumn'; 
        Mcd(ns+1:end,ns+1:end) = Mcd(ns+1:end,ns+1:end).*matrix;
               
    end
end
M = M + pb*M_cs/(C-1)-1/(C-1)*Mcd;
M = M/norm(M,'fro');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% construct intra-domain discrimination used DICD
BBT1=[];

for c = reshape(unique(Ys),1,C)
    SClassNum(c)=length(find(Ys==c));
    TClassNum(c)=length(find(Yt0==c));
    ClassNum(c)=length(find(Yall==c));
end

BBS1=zeros(ns,ns);
for i=1:ns
    BBS1(i,i) = ns/SClassNum(Ys(i))*(SClassNum(Ys(i)));
    for j=i+1:ns
        if Ys(i)==Ys(j)
            BBS1(i,j) = ns/SClassNum(Ys(i))*(-1);
            BBS1(j,i) = ns/SClassNum(Ys(i))*(-1);
        end
    end
end

YsColumn=singlelbs2multilabs(Ys,C);
BBS2 = YsColumn*YsColumn' - ones(ns,ns);

for i=1:ns
    BBS2(i,i) = ns - SClassNum(Ys(i));
end
BBS1=BBS1-rho*BBS2;


if ~isempty(Yt0) && length(Yt0)==nt
    BBT1=zeros(nt,nt);
    for i=1:nt
            BBT1(i,i) = nt/TClassNum(Yt0(i))*(TClassNum(Yt0(i)));
        for j=i+1:nt
            if Yt0(i)==Yt0(j)
                            BBT1(i,j) = nt/TClassNum(Yt0(i))*(-1);
                            BBT1(j,i) = nt/TClassNum(Yt0(i))*(-1);
            end
        end
    end
    BBT1 = BBT1;

    Yt0Column=singlelbs2multilabs(Yt0,C);
    BBT2 = Yt0Column*Yt0Column' - ones(nt,nt);

    for i=1:nt
        BBT2(i,i) = nt - TClassNum(Yt0(i));
    end
    BBT1=BBT1-rho*BBT2;
end

BB = blkdiag(BBS1,BBT1);
BB = BB/norm(BB,'fro');       % W_intra
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Construct centering matrix
H = eye(n)-1/(n)*ones(n,n);
if ~isempty(Yt0) && length(Yt0)==nt
    
    if strcmp(ker,'primal')
        [A,~] = eigs(X*M*X'+r*eye(m)+alpha*X*BB*X',X*H*X',k,'SM');
        Z = A'*X;
     
    else
        K = kernel(ker,X,[],gamma);
        Ks=K(:,1:ns);
        [A,~] = eigs(K*M*K'+r*eye(n)+alpha*K*BB*K',K*H*K',k,'SM');
        Z = A'*K;
    end
else
    if strcmp(ker,'primal')
        [A,~] = eigs(X*M*X'+r*eye(m)+alpha*Xs*BB*Xs',X*H*X',k,'SM');
        Z = A'*X;
    else
        K = kernel(ker,X,[],gamma);
        Ks=K(:,1:ns);
        [A,~] = eigs(K*M*K'+r*eye(n)+alpha*Ks*BB*Ks',K*H*K',k,'SM');
        Z = A'*K;
    end
end
fprintf('Algorithm AACD terminated!!!\n\n');
end

%% Convert single column labels to multi-column labels
function label=singlelbs2multilabs(y,nclass)
    L=length(y);
    label=zeros(L,nclass);
    for i=1:L
        label(i,y(i))=1;
    end
end



