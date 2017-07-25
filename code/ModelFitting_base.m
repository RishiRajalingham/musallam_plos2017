function [r,Z_pred,B] = ModelFitting_base(X, Y, Z, model, ngauss)
% function [r,Z_pred,B] = ModelFitting_base(X,Y,Z, model, ngauss)

    opts = optimset('Display','off', ...
        'MaxFunEvals', 1000000, 'UseParallel', true);
    niter = 1;
    [b0,bl,bu] = get_param_bounds(model, ngauss);
    fmodel = eval(model);

    if niter > 1        
        Bs = nan(niter,length(b0));
        rs = nan(niter,1);
        Zps = nan(niter,length(Z));
        parfor iter = 1:niter
            B_ = lsqcurvefit(fmodel, b0,[X,Y], Z, bl, bu, opts);        
            Zp_ = fmodel(B_, [X,Y]);  
            r_ = nancorr_rr(Z,Zp_);
            Bs(iter,:) = B_; 
            rs(iter) = r_;
            Zps(iter,:) = Zp_;
        end
        [~,mi] = max(rs);
        B = Bs(mi,:);
        r = rs(mi);
        Z_pred = Zps(mi,:);
    else
        B = lsqcurvefit(fmodel, b0,[X,Y], Z, bl, bu, opts);        
        Z_pred = fmodel(B, [X,Y]);  
        r = nancorr_rr(Z,Z_pred);
    end
    

    function [b0,bl,bu] = get_param_bounds(model_id, ncomp)
        switch model_id
            case {'@dirFn'} % cosine (muang)
                b0 = [0];    bl = -pi;   bu = pi;
                return;
            case {'@dirFn_2'} % gaussian (muang, -1/sig)
                b0 = [0, -0.01];    bl = [-pi,-100];   bu = [pi,0];
                return;
            case {'@dirFn_3'} % von mises (muang, 1/sig)
                b0 = [0, 1];    bl = [-pi,0];   bu = [pi,100];
                return;
            case {'@dirAmpFn'} % cosine x amp (muang, k)
                b0 = [0,1];    bl = [-pi,-100];   bu = [pi,100];
                return;
            case {'@dirAmpFn_2'} % gaussian x amp (muamp, -1/sig, k)
                b0 = [0,-0.01,1];    bl = [-pi,-100, -100];   bu = [pi,0,100];
            case {'@dirAmpFn_3'} % gaussian x amp (muamp, -1/sig, k)
                b0 = [0, 1, 1];    bl = [-pi,0, -100];   bu = [pi,100, 100];
            case {'@sum_gaussFn'}
                 [b0,bl,bu] = get_gauss_params(ncomp);
                return;
            case {'@control_fn'} % control model with large number of parameters
                [b0,bl,bu] = get_ctrl_params(ncomp);
                return
        end         
    end

    function [B0,Bl,Bu] = get_gauss_params(N)
        % N gaussian components
        maxposval = 20;
        maxspread = 50;
        minspread = 5;
        b0_ = [0,0,2*minspread,2*minspread,0];
        bl_ = [-maxposval,-maxposval,minspread,minspread,-1];
        bu_ = [maxposval,maxposval,maxspread,maxspread,1];

        w = rand(1,N);
        wl = zeros(1,N) .* eps;
        wu = ones(1,N);
        B0 = w; Bl = wl; Bu = wu;
        for i = 1:N
            B0 = [B0, b0_];
            Bl = [Bl, bl_];
            Bu = [Bu, bu_];
        end
    end   

    function [B0,Bl,Bu] = get_ctrl_params(N)
        % N cosine components
        maxfreq = 10;
        B0 = [ones(1,N), zeros(1,N), rand(1,N)*maxfreq];
        Bl = [zeros(1,N), ones(1,N)*-pi, zeros(1,N)];
        Bu = [ones(1,N), ones(1,N)*pi, rand(1,N)*maxfreq];
    end 

%% Models 
    function ZZ = dirFn(beta0, XY)
        muang = beta0(1);
        ang = cart2pol(XY(:,1), XY(:,2));
        delang = angdiff(repmat(muang,size(ang,1),1), ang);
        ZZ = midrange(cos(delang));
    end

    function ZZ = dirAmpFn(beta0, XY)
        ZZ = dirFn(beta0, XY);
        k = beta0(2);
        [~,rad] = cart2pol(XY(:,1), XY(:,2));
        ZZ = midrange(ZZ .* rad .* k);
    end

    function ZZ = dirFn_2(beta0, XY)
        muang = beta0(1);
        sigang = beta0(2);
        ang = cart2pol(XY(:,1), XY(:,2));
        delang = angdiff(repmat(muang,size(ang,1),1), ang);
        ZZ = midrange(exp(sigang .* delang.^2));
    end

    function ZZ = dirFn_3(beta0, XY)
        muang = beta0(1);
        sigang = beta0(2);
        ang = cart2pol(XY(:,1), XY(:,2));
        ZZ = midrange(circ_vmpdf(ang, muang, sigang));
        
    end

    function ZZ = dirAmpFn_2(beta0, XY)
        ZZ = dirFn_2(beta0, XY);
        k = beta0(3);
        [~,rad] = cart2pol(XY(:,1), XY(:,2));
        ZZ = midrange(ZZ .* rad .* k);
    end

    function ZZ = dirAmpFn_3(beta0, XY)
        ZZ = dirFn_3(beta0, XY);
        k = beta0(3);
        [~,rad] = cart2pol(XY(:,1), XY(:,2));
        ZZ = midrange(ZZ .* rad .* k);
    end

    function ZZ = gaussFn(beta0, XY)
        mux = beta0(1); 
        muy = beta0(2);
        sigx = beta0(3);
        sigy = beta0(4);
        rho = beta0(5);
    
        MU = [mux, muy];
        sigxy = rho * sigx * sigy;
        SIG = [sigx.^2, sigxy; sigxy sigy.^2];
        d = XY - repmat(MU, size(XY,1), 1);
        SIGinvd = (SIG)\(d');
        ZZ = exp( -(d * SIGinvd));
        ZZ = diag(ZZ);
        ZZ(~isfinite(ZZ)) = 0;
    end

    function ZZ = sum_gaussFn(beta0, XY)
        N = (length(beta0))/6;
        W = beta0(1:N);
%         W = W ./ sum(W);
        
        V = beta0(N+1:end);
        
        ZZ = zeros(size(XY,1),1);
        for i = 1:N
            w = V((i-1)*5+1:(i)*5);
            ZZ = ZZ + W(i) .* gaussFn(w, XY);
        end
        ZZ = midrange(ZZ);
    end

    function ZZ = control_fn(beta0, XY)
        ZZ = zeros(size(XY(:,1)));
        nparam_percomp = 3;
        N = length(beta0)/nparam_percomp;
        ang = cart2pol(XY(:,1), XY(:,2));
        for ni = 1:N
            amp = beta0(ni);
            muang = beta0(N+ni);
            freq = ni; %ceil(beta0(2*N+ni));
            delang = angdiff(repmat(muang,size(ang,1),1), ang);
            ZZ = ZZ + amp * cos(delang*freq);
        end
        ZZ = midrange(ZZ);
    end

end