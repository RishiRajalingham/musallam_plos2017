function RF_analysis()
    
    %% init
    addpath('lib');
    FN = {'Z_s2.mat', 'Z_s2_pmd.mat'};
    nEp = 4; nBin = 11; 
    
    %% main
    for fi = 1% 1:2
        computeRF(FN{fi});
    end

    %% functions
    function computeRF(fn)
        fprintf(1, 'Loading %s ..\n', fn);
        dat = load(fn, 'x', 'y', 'Z', 'p_task');  
        x = dat.x; y = dat.y; Z = dat.Z; p_task = dat.p_task;
        neurons_oi = find(min(p_task,[],2) < 0.05/4);
        RF = nan(length(Z), nEp, nBin, nBin);
        model_fits = cell(length(neurons_oi), nEp);
        for ni_ = 1:length(neurons_oi)
            tic;
            ni = neurons_oi(ni_);
            for ei = 1:nEp
                FR = Z{ni}.FR_eoi(:,ei);
                [OUT,RF(ni,ei,:,:)] = ModelConsistency(x,y,FR);
                model_fits{ni,ei} = OUT;
            end
            tt = toc;
            fprintf(1, 'Done %d out of %d .. in %f s \n', ni_, length(neurons_oi), tt);
        end
        outfn = ['dat/spatial_fitting_v2_', fn];
        save(outfn, 'model_fits', 'RF'); 
        fprintf(1, 'Saved to %s ..\n', outfn);
    end
        
    

end
