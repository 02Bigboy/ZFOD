  function Yt = TSRP(Zs,Ys,Zt,Cls,r_wj,TT)
  % TSRP uses the selected target domain samples with high confidence and source domain samples
  % training a stronger classifier to help the remaining samples
  % with low confidence to remedy inaccurate pseudo labels and makes it more accurate.
         Yt = Cls;
         Ztp = Zt;
         n=TT; %挑选的级数
         n_m = 1:length(Cls);
         Z = Zs;
         Y = Ys;
         for i=1:n
             [nn] = UTSP(Ztp,Cls,r_wj);
             nn_o = n_m(nn);
             n_y = setdiff(n_m,nn_o);
             Zty =Zt(n_y,:);
             Zto = Zt(nn_o,:);
             Yty = Yt(n_y);% 
             Z = [Z;Zty];
             Y = [Y;Yty];
             knn_model_all = fitcknn (Z,Y) ;
             Yt(nn_o)=knn_model_all.predict(Zto);
             if length(nn_o)==0
                 break;
             end
       
             Ztp = Zt(nn_o,:);
             Cls = Yt(nn_o);
             n_m=nn_o;
         end   
  end