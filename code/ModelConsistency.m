function [OUT,RF] = ModelConsistency(x,y,FR)
% function [OUT, RF] = ModelConsistency(x,y,FR)
    nBin = 11; niter = 10;
    targetid = x*100 + y;

    RF = computeRF_base(x, y, FR, nBin);

    out_sh = cell(niter,2);
    for iter = 1:niter
        inds = crossvalind('kfold', targetid, 2);
        for k = 1:2
            tr = inds == k;
            [RF_, XX, YY]= computeRF_base(x(tr), y(tr), FR(tr), nBin);
            try
                out_sh{iter,k} = ModelFitting(XX, YY, RF_);
            catch e
                display('model-fitting exception')
            end
        end
    end
    OUT = get_consistency(out_sh);

    function [RF, XX, YY] = computeRF_base(x, y, FR, nBin)
        ux = linspace(-14, 14, nBin); ux_n = length(ux);
        uy = linspace(-14, 14, nBin); uy_n = length(uy);
        [XX,YY] = meshgrid(ux,uy);
        [mu,b] = grpstats(FR, [x,y], {'mean','gname'});
        xx = str2double(b(:,1));    yy = str2double(b(:,2));
        F = scatteredInterpolant(xx,yy,mu, 'natural');
        RF = F(XX(:), YY(:));
        RF = reshape(RF, ux_n, uy_n);
    end
    
    function OUT = get_consistency(in_sh)
        Nm = length(in_sh{1});
        OUT = cell(Nm,1);
        for ni = 1:Nm
            OUT{ni}.icn = [];   OUT{ni}.icm = [];
            OUT{ni}.rho1 = [];  OUT{ni}.rho2= [];
            OUT{ni}.rho1_tr = [];  OUT{ni}.rho2_tr = [];
            OUT{ni}.Z_true1 = [];  OUT{ni}.Z_true2 = [];
            OUT{ni}.Z_pred1 = [];  OUT{ni}.Z_pred2 = [];
            
            for iter_ = 1:niter
                if isempty(in_sh{iter_,1}) || isempty(in_sh{iter_,2})
                    continue;
                end
                N1 = in_sh{iter_,1}{ni}.Z_true;    N2 = in_sh{iter_,2}{ni}.Z_true;
                M1 = in_sh{iter_,1}{ni}.Z_pred;    M2 = in_sh{iter_,2}{ni}.Z_pred;
                
                OUT{ni}.icn = cat(1, OUT{ni}.icn, nancorr_rr(N1,N2));
                OUT{ni}.icm = cat(1, OUT{ni}.icm, nancorr_rr(M1,M2));
                
                OUT{ni}.rho1 = cat(1, OUT{ni}.rho1, nancorr_rr(N1,M2));
                OUT{ni}.rho2 = cat(1, OUT{ni}.rho2, nancorr_rr(N2,M1));
                
                OUT{ni}.rho1_tr = cat(1, OUT{ni}.rho1_tr, nancorr_rr(N1,M1));
                OUT{ni}.rho2_tr = cat(1, OUT{ni}.rho2_tr, nancorr_rr(N2,M2));
                
                OUT{ni}.numparam = length(in_sh{iter_,1}{ni}.b);
                
                OUT{ni}.Z_true1 = cat(1, OUT{ni}.Z_true1, N1(:)');
                OUT{ni}.Z_true2 = cat(1, OUT{ni}.Z_true2, N2(:)');
                OUT{ni}.Z_pred1 = cat(1, OUT{ni}.Z_pred1, M1(:)');
                OUT{ni}.Z_pred2 = cat(1, OUT{ni}.Z_pred2, M2(:)');
            end
            OUT{ni}.rho_n = sqrt(OUT{ni}.rho1 .* OUT{ni}.rho2) ./ sqrt(OUT{ni}.icn .* OUT{ni}.icm);
            OUT{ni}.rho_n_tr = sqrt(OUT{ni}.rho1_tr .* OUT{ni}.rho2_tr) ./ sqrt(OUT{ni}.icn .* OUT{ni}.icm);
            OUT{ni}.p_ic = ttest(OUT{ni}.icn);
            OUT{ni}.mu_ic = nanmean(OUT{ni}.icn);
            OUT{ni}.nparams = in_sh{1,1}{ni}.nparams;
        end 
    end

   
end
