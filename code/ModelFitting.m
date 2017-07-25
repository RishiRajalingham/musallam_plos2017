function out = ModelFitting(XX, YY, RF, plot_on)
% function out = ModelFitting(XX, YY, RF, plot_on)


%% Init 
    test_models = {
        {'@dirFn', 0} % cosine tuning
        {'@dirAmpFn', 0}
%         {'@dirFn_2', 0} %gaussian tuning
%         {'@dirAmpFn_2', 0}
        {'@dirFn_3', 0} % von mises tuning
        {'@dirAmpFn_3', 0}
        {'@sum_gaussFn', 1}
%         {'@sum_gaussFn', 2}
%         {'@sum_gaussFn', 3}
%         {'@control_fn', 2}
%         {'@control_fn', 4}
%         {'@control_fn', 5}
%         {'@control_fn', 7}
%         {'@control_fn', 9}
        };

    t = isfinite(RF(:));
    xx = XX(t);
    yy = YY(t);
    zz = midrange(RF(t));
   
    ww = warning('query', 'last');
    warning('off',ww.identifier);
    if ~exist('plot_on', 'var')
        plot_on = 0;
    end
    
%% Main 
    
    out = cell(length(test_models),1);
    parfor tmi = 1:length(test_models)
        [r,Z_pred,B] = ModelFitting_base(xx,yy,zz, test_models{tmi}{1}, test_models{tmi}{2});
        out{tmi}.test_models = test_models{tmi};
        out{tmi}.r = r;
        out{tmi}.b = B;
        out{tmi}.Z_pred = Z_pred(:);
        out{tmi}.Z_true = zz(:);
        out{tmi}.nparams = length(B) - 1*(test_models{tmi}{2} > 0);
    end
    
    if plot_on;
        figure;
        ha = tight_subplot(2, length(test_models), [.01,.01],[.01,.01],[.01,.01]);
        axes(ha(1)); sanePColor(XX(1,:),XX(1,:),RF); axis square;
        for tmi = 2:length(test_models)
            axes(ha(tmi)); axis off;
        end
        for tmi = 1:length(test_models)
            RF2 = reshape(out{tmi}.Z_pred, size(XX,1), size(XX,1));
            axes(ha(tmi+length(test_models))); sanePColor(XX(1,:),XX(1,:),RF2); 
            axis square; shading interp; axis off;
%             title(strrep(out{tmi}.test_models{1}, '@', ''));
        end
    end


end