% run_debug_detSrs.m - interactive script version
% Open this file in MATLAB and press Run (green button) or run the script name.
%
% Top-level parameters (edit these, then Run):
%   mode: 'gen' to generate SRS and debug, 'tv' to use an existing TV HDF5 file
%   outDir: directory to save debug MAT files
%   h5file: (optional) path to HDF5 TV file when mode='tv'
%
% Example usage after opening in MATLAB editor:
%   mode = 'gen'; outDir = 'C:/tmp'; run_debug_detSrs
%
% This script will call the internal function `run_debug_detSrs_main` which
% contains the original implementation. The script sets sensible defaults so
% you can run it immediately.

%% --- User-editable parameters ---
mode = 'gen';           % 'gen' or 'tv'
outDir = fullfile(pwd,'srs_debug_out');
% If using mode 'tv', set h5file to the full path of the TV HDF5 file below
h5file = ''; % e.g. 'C:/path/to/TVnr_0001_SRS_gNB_CUPHY_... .h5'
% If you open this file in MATLAB editor and press Run, enable AUTOSET_PWD
% so the script changes MATLAB's current folder to the repo root automatically.
AUTOSET_PWD = 1;
%% -------------------------------

% If requested, change current folder to repo root (so Run in editor behaves)
if exist('AUTOSET_PWD','var') && AUTOSET_PWD
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    try
        cd(repoRoot);
        fprintf('Changed current folder to repo root: %s\n', repoRoot);
    catch
        warning('Could not change current folder to repo root.');
    end
end

if ~exist(outDir,'dir')
    mkdir(outDir);
end

fprintf('Running run_debug_detSrs (mode=%s)\n', mode);

% call main implementation
run_debug_detSrs_main(mode, outDir, h5file);

%% Local main function (implementation preserved)
function run_debug_detSrs_main(mode, outDir, h5file)
    % Implementation adapted from previous function version
    if nargin < 1
        error('Mode required: ''gen'' or ''tv''.');
    end
    if nargin < 2 || isempty(outDir)
        outDir = pwd;
    end
    if ~exist(outDir,'dir')
        mkdir(outDir);
    end

    % Ensure MATLAB path includes nr_matlab
    % Attempt to discover repo root based on file location
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    nrMatlabPath = fullfile(repoRoot, '5GModel', 'nr_matlab');
    if exist(nrMatlabPath,'dir')
        addpath(genpath(nrMatlabPath));
    else
        warning('Could not find 5GModel/nr_matlab in expected location. Ensure functions like genSrs/detSrs are on the path.');
    end

    switch lower(mode)
        case 'gen'
            fprintf('Mode: gen -> generate SRS via genSrs then run detSrs and save debug vars.\n');
            % Try to load srsTable if not in workspace by scanning nr_matlab/srs for .mat files
            if ~exist('srsTable','var')
                srsTable = struct();
                srsDir = fullfile(nrMatlabPath, 'srs');
                if exist(srsDir,'dir')
                    matFiles = dir(fullfile(srsDir, '*.mat'));
                    if ~isempty(matFiles)
                        for k = 1:numel(matFiles)
                            matPath = fullfile(matFiles(k).folder, matFiles(k).name);
                            try
                                tmp = load(matPath);
                                fns = fieldnames(tmp);
                                for fi = 1:numel(fns)
                                    % merge loaded variable into srsTable
                                    srsTable.(fns{fi}) = tmp.(fns{fi});
                                end
                                fprintf('Loaded %s into srsTable\n', matFiles(k).name);
                            catch ME
                                warning('Failed to load %s: %s', matFiles(k).name, ME.message);
                            end
                        end
                    else
                        warning('No .mat files found in %s', srsDir);
                    end
                else
                    warning('SRS folder not found: %s', srsDir);
                end

                % Try to map commonly named variables into expected srsTable fields
                if ~isfield(srsTable, 'srs_BW_table') && isfield(srsTable, 'T')
                    srsTable.srs_BW_table = srsTable.T;
                    fprintf('Mapped variable T -> srs_BW_table\n');
                end
                if ~isfield(srsTable, 'srsPrimes') && isfield(srsTable, 'p')
                    srsTable.srsPrimes = srsTable.p;
                    fprintf('Mapped variable p -> srsPrimes\n');
                end

                % Validate essential field after mapping
                if ~isfield(srsTable, 'srs_BW_table')
                    error(['srsTable not found or missing ''srs_BW_table''.\n', ...
                        'Please ensure 5GModel/nr_matlab/srs/*.mat files are present, or define ''srsTable'' in your workspace.']);
                end
            end

            % Basic carrier & pdu defaults (tune for your setup)
            carrier.N_slot_frame_mu = 10;
            carrier.N_symb_slot = 14;
            carrier.idxSlotInFrame = 0;
            carrier.idxFrame = 0;
            carrier.delta_f = 30e3; % subcarrier spacing

            % Minimal PDU structure consistent with genSrs mappings
            pdu.numAntPorts = 0; % 0->1 port
            pdu.numSymbols = 0;  % 0->1 symbol
            pdu.numRepetitions = 0;
            pdu.combSize = 0;    % 0->comb size 2
            pdu.timeStartPosition = 0;
            pdu.sequenceId = 10;
            pdu.configIndex = 0;
            pdu.bandwidthIndex = 0;
            pdu.combOffset = 0;
            pdu.cyclicShift = 0;
            pdu.frequencyPosition = 0;
            pdu.frequencyShift = 0;
            pdu.frequencyHopping = 0;
            pdu.resourceType = 0;
            pdu.Tsrs = 1;
            pdu.Toffset = 0;
            pdu.groupOrSequenceHopping = 0;
            pdu.srsPduIdx = 1; % used when saving TV names in genSrs

            % Large Xtf to avoid index overflow; genSrs will embed into it
            Nf = 20000; Nsym = carrier.N_symb_slot; Nant = 4;
            Xtf = complex(zeros(Nf, Nsym, Nant));

            % Generate SRS into Xtf using genSrs (if available)
            try
                Xtf = genSrs(pdu, srsTable, carrier, Xtf, 1);
                fprintf('genSrs ran successfully.\n');
            catch ME
                warning('genSrs failed: %s\nProceeding with partial generation heuristics.', ME.message);
            end

            % Now perform the same detection math as detSrs to save intermediate vars
            fprintf('Computing intermediate SRS debug variables...\n');
            SrsParams = struct();
            numAntPorts_mapping = [1 2 4];
            SrsParams.N_ap_SRS = numAntPorts_mapping(pdu.numAntPorts+1);
            numSymbols_mapping = [1 2 4];
            SrsParams.N_symb_SRS = numSymbols_mapping(pdu.numSymbols+1);
            numRepetitions_mapping = [1 2 4];
            SrsParams.R = numRepetitions_mapping(pdu.numRepetitions+1);
            combSize_mapping = [2 4];
            SrsParams.K_TC = combSize_mapping(pdu.combSize+1);

            SrsParams.l0 = pdu.timeStartPosition;
            SrsParams.n_ID_SRS = pdu.sequenceId;
            SrsParams.C_SRS = pdu.configIndex;
            SrsParams.B_SRS = pdu.bandwidthIndex;
            SrsParams.k_TC_bar = pdu.combOffset;
            SrsParams.n_SRS_cs = pdu.cyclicShift;
            SrsParams.n_RRC = pdu.frequencyPosition;
            SrsParams.n_shift = pdu.frequencyShift;
            SrsParams.b_hop = pdu.frequencyHopping;
            SrsParams.resourceType = pdu.resourceType;
            SrsParams.Tsrs = pdu.Tsrs;
            SrsParams.Toffset = pdu.Toffset;
            SrsParams.groupOrSequenceHopping = pdu.groupOrSequenceHopping;

            SrsParams.N_slot_frame = carrier.N_slot_frame_mu;
            SrsParams.N_symb_slot = carrier.N_symb_slot;
            SrsParams.idxSlotInFrame = mod(carrier.idxSlotInFrame, carrier.N_slot_frame_mu);
            SrsParams.idxFrame = carrier.idxFrame;
            SrsParams.delta_f = carrier.delta_f;

            % Extract table values
            srs_BW_table = srsTable.srs_BW_table;
            C_SRS = SrsParams.C_SRS; B_SRS = SrsParams.B_SRS; K_TC = SrsParams.K_TC;
            m_SRS_b = srs_BW_table(C_SRS+1,2*B_SRS+1);
            N_sc_RB = 12;
            M_sc_b_SRS = m_SRS_b*N_sc_RB/K_TC;
            if K_TC == 4
                n_SRS_cs_max = 12;
            else
                n_SRS_cs_max = 8;
            end

            % compute phase shift alpha
            alpha = zeros(1,SrsParams.N_ap_SRS);
            for p = 0:SrsParams.N_ap_SRS-1
                n_SRS_cs_i = mod(SrsParams.n_SRS_cs + (n_SRS_cs_max * p)/SrsParams.N_ap_SRS,  n_SRS_cs_max);
                alpha(p+1) = 2 * pi * n_SRS_cs_i/n_SRS_cs_max;
            end

            % compute u, v and r_bar
            c = build_Gold_sequence(SrsParams.n_ID_SRS, 10 * SrsParams.N_slot_frame * SrsParams.N_symb_slot);
            u = zeros(1,SrsParams.N_symb_SRS);
            v = zeros(1,SrsParams.N_symb_SRS);
            for l_prime = 0:SrsParams.N_symb_SRS-1
                if SrsParams.groupOrSequenceHopping == 0
                    f_gh = 0;
                    v(l_prime + 1) = 0;
                elseif SrsParams.groupOrSequenceHopping == 1
                    f_gh = 0;
                    for m = 0:7
                        idxSeq = 8 * (SrsParams.idxSlotInFrame * SrsParams.N_symb_slot + SrsParams.l0 + l_prime) + m;
                        f_gh = f_gh + c(idxSeq + 1) * 2^m;
                    end
                    f_gh = mod(f_gh, 30);
                    v(l_prime + 1) = 0;
                elseif SrsParams.groupOrSequenceHopping == 2
                    if M_sc_b_SRS >= 6 * N_sc_RB
                        idxSeq = SrsParams.idxSlotInFrame * SrsParams.N_symb_slot + SrsParams.l0 + l_prime;
                        v(l_prime + 1) = c(idxSeq + 1);
                    else
                        v(l_prime + 1) = 0;
                    end
                else
                    error('groupOrSequenceHopping not supported');
                end
                u(l_prime + 1) = mod(f_gh + SrsParams.n_ID_SRS, 30);
            end

            r_bar = zeros(SrsParams.N_symb_SRS, M_sc_b_SRS);
            for l_prime = 0:SrsParams.N_symb_SRS-1
                r_bar(l_prime+1,:) = LowPaprSeqGen(M_sc_b_SRS, u(l_prime+1), v(l_prime+1));
            end

            % compute k0
            k0 = zeros(SrsParams.N_symb_SRS, SrsParams.N_ap_SRS);
            for l_prime = 0:SrsParams.N_symb_SRS-1
                for p = 0:SrsParams.N_ap_SRS-1
                    if (SrsParams.n_SRS_cs >= n_SRS_cs_max/2) && (SrsParams.N_ap_SRS == 4) && (p == 1 || p == 3)
                        k_TC = mod(SrsParams.k_TC_bar + K_TC/2, K_TC);
                    else
                        k_TC = SrsParams.k_TC_bar;
                    end
                    k0_bar = SrsParams.n_shift * N_sc_RB + k_TC;
                    k0(l_prime+1, p+1) = k0_bar;
                    for b = 0:SrsParams.B_SRS
                        if SrsParams.b_hop >= SrsParams.B_SRS
                            Nb = srs_BW_table(C_SRS+1,2*b+2);
                            m_SRS_b = srs_BW_table(C_SRS+1,2*b+1);
                            nb = mod(floor(4*SrsParams.n_RRC/m_SRS_b), Nb);
                        else
                            Nb = srs_BW_table(C_SRS+1,2*b+2);
                            m_SRS_b = srs_BW_table(C_SRS+1,2*b+1);
                            if b <= SrsParams.b_hop
                                nb = mod(floor(4*SrsParams.n_RRC/m_SRS_b), Nb);
                            else
                                if SrsParams.resourceType == 0
                                    n_SRS = floor(l_prime/SrsParams.R);
                                else
                                    slotIdx = SrsParams.N_slot_frame * SrsParams.idxFrame + SrsParams.idxSlotInFrame - SrsParams.Toffset;
                                    if mod(slotIdx, SrsParams.Tsrs) == 0
                                        n_SRS = (slotIdx/SrsParams.Tsrs) * (SrsParams.N_symb_SRS/SrsParams.R) + floor(l_prime/SrsParams.R);
                                    else
                                        warning('Not an SRS slot');
                                        n_SRS = 0;
                                    end
                                end
                                PI_bm1 = 1;
                                for b_prime = SrsParams.b_hop+1:b-1
                                    PI_bm1 = PI_bm1 * srs_BW_table(C_SRS+1,2*b_prime+2);
                                end
                                PI_b = PI_bm1 * Nb;
                                if mod(Nb,2) == 0
                                    Fb = (Nb/2)*floor(mod(n_SRS, PI_b)/PI_bm1) + floor(mod(n_SRS, PI_b)/(2*PI_bm1));
                                else
                                    Fb = floor(Nb/2)*floor(n_SRS/PI_bm1);
                                end
                                nb = mod(Fb + floor(4*SrsParams.n_RRC/m_SRS_b), Nb);
                            end
                        end
                        M_sc_b_SRS = m_SRS_b*N_sc_RB/K_TC;
                        k0(l_prime+1, p+1) = k0(l_prime+1, p+1) + K_TC * M_sc_b_SRS * nb;
                    end
                end
            end

            % map and cross-correlate
            [~, ~, nAnt] = size(Xtf);
            xcor = zeros(M_sc_b_SRS, SrsParams.N_symb_SRS, nAnt, SrsParams.N_ap_SRS);
            for l_prime = 0:SrsParams.N_symb_SRS-1
                for p = 0:SrsParams.N_ap_SRS-1
                    freq_idx = k0(l_prime+1, p+1) + (0:K_TC:(M_sc_b_SRS-1)*K_TC);
                    sym_idx = l_prime + SrsParams.l0;
                    r = squeeze(r_bar(l_prime+1,:)).*exp(1i*(0:(M_sc_b_SRS-1))*alpha(p+1));
                    for idxAnt = 1:nAnt
                        xcor(:, l_prime+1, idxAnt, p+1) = conj(r(:)) .* squeeze(Xtf(freq_idx+1, sym_idx+1, idxAnt));
                    end
                end
            end

            % RSSI
            ant_rssi_lin = zeros(1,nAnt);
            ant_rssi_dB = zeros(1,nAnt);
            for idxAnt = 1:nAnt
                temp = xcor(:,:,idxAnt,:);
                ant_rssi_lin(idxAnt) = mean(abs(temp(:).^2));
                if ant_rssi_lin(idxAnt) == 0
                    ant_rssi_dB(idxAnt) = -100;
                else
                    ant_rssi_dB(idxAnt) = 10*log10(abs(ant_rssi_lin(idxAnt)));
                end
            end
            rssi_lin = mean(ant_rssi_lin);
            if rssi_lin == 0
                rssi_dB = -100;
            else
                rssi_dB = 10*log10(rssi_lin);
            end

            % timing offset and Hest estimation (copied from detSrs)
            Hest3 = [];
            xcor_sum = [];
            for idxSym = 1:SrsParams.N_symb_SRS
                for idxAnt = 1:nAnt
                    Hest1 = xcor(:,idxSym,idxAnt,:);
                    Hest1 = Hest1(:);
                    len_Hest1 = length(Hest1);
                    Hest1 = reshape(Hest1, [SrsParams.N_ap_SRS, len_Hest1/SrsParams.N_ap_SRS]);
                    Hest2 = mean(Hest1, 1);
                    Hest2 = reshape(Hest2, [M_sc_b_SRS/SrsParams.N_ap_SRS, SrsParams.N_ap_SRS]);
                    for idxPort = 1:SrsParams.N_ap_SRS
                        Hest_port = Hest2(:,idxPort);
                        xcor_sum(idxSym, idxAnt, idxPort) = sum(Hest_port(2:end) .* conj(Hest_port(1:end-1)));
                        Hest3(:, idxSym, idxAnt, idxPort) = Hest_port;
                    end
                end
            end
            phaRot = angle(sum(sum(sum(xcor_sum))));

            nReTotal = M_sc_b_SRS/SrsParams.N_ap_SRS;
            reDist = K_TC*SrsParams.N_ap_SRS;
            if reDist >= 16
                nRbEst = 4;
            else
                nRbEst = 2;
            end
            nRePerEst = nRbEst*N_sc_RB/K_TC/SrsParams.N_ap_SRS;
            nEst = nReTotal/nRePerEst;
            estFactor1 = 3/2;

            Hest = [];
            Ps = [];
            Pn = [];
            for idxSym = 1:SrsParams.N_symb_SRS
                for idxAnt = 1:nAnt
                    for idxPort = 1:SrsParams.N_ap_SRS
                        Hest4 = Hest3(:, idxSym, idxAnt, idxPort) .* exp(-1j*phaRot*(0:(M_sc_b_SRS/SrsParams.N_ap_SRS-1))');
                        for idxEst = 1:nEst
                            Hest5 = Hest4((idxEst-1)*nRePerEst+1:idxEst*nRePerEst);
                            Hest(idxEst, idxSym, idxAnt, idxPort) = mean(Hest5);
                            algSel = 1;
                            if algSel == 0
                                Havg = mean(Hest5);
                                Hdiff = Hest5 - Havg;
                                Ps(idxEst, idxSym, idxAnt, idxPort) = abs(Havg)^2;
                                Pn(idxEst, idxSym, idxAnt, idxPort) = mean(abs(Hdiff).^2);
                            elseif algSel == 1
                                for idxRe = 2:nRePerEst-1
                                    Havg(idxRe-1) = (Hest5(idxRe-1) + Hest5(idxRe) + Hest5(idxRe+1))/3;
                                    Hdiff(idxRe-1) = Havg(idxRe-1)-Hest5(idxRe);
                                end
                                Ps(idxEst, idxSym, idxAnt, idxPort) = mean(abs(Havg).^2);
                                Pn(idxEst, idxSym, idxAnt, idxPort) = mean(abs(Hdiff).^2) * estFactor1;
                            end
                        end
                    end
                end
            end

            Ps_wide = mean(Ps, [1,2,3,4]);
            Pn_wide = mean(Pn, [1,2,3,4]);
            Ps_rb = mean(Ps, [3,4]);
            Pn_rb = mean(Pn, [3,4]);

            SNR_wide = 10*log10(Ps_wide/Pn_wide)  - 10*log10(SrsParams.N_ap_SRS);
            SNR_rb = 10*log10(Ps_rb./Pn_rb)  - 10*log10(SrsParams.N_ap_SRS);
            SNR_rb = reshape(repmat(SNR_rb(:), [1, nRbEst]).', [m_SRS_b, SrsParams.N_symb_SRS]);

            to_est = -1/2/pi*phaRot;
            to_est_sec = to_est/(SrsParams.delta_f * SrsParams.N_ap_SRS * K_TC);

            % Pack results similar to detSrs
            SrsOutput.to_est_ms = to_est_sec*1e6;
            SrsOutput.Hest = Hest;
            SrsOutput.nRbHest = nRbEst;
            SrsOutput.wideSnr = SNR_wide;
            SrsOutput.rbSnr = SNR_rb;
            SrsOutput.rssi = rssi_dB;
            SrsOutput.ant_rssi = ant_rssi_dB;

            % Also run built-in detSrs to compare (if available)
            try
                pduList = {pdu};
                detOut = detSrs(pduList, srsTable, carrier, Xtf);
            catch
                detOut = [];
            end

            % Save debug MAT
            outMat = fullfile(outDir, ['srs_debug_gen_' datestr(now,'yyyymmdd_HHMMSS') '.mat']);
            save(outMat, 'r_bar','k0','xcor','xcor_sum','phaRot','Hest','Ps','Pn','to_est_sec','SrsOutput','detOut','-v7.3');
            fprintf('Saved debug MAT to %s\n', outMat);

        case 'tv'
            if nargin < 3 || isempty(h5file)
                error('For mode ''tv'' you must provide the path to an SRS TV HDF5 file as third argument.');
            end
            fprintf('Mode: tv -> load HDF5 TV %s and run detSrs with debug save.\n', h5file);
            % read SrsParams_0 and X_tf from file
            try
                info = h5info(h5file);
                % try common dataset names
                if any(strcmp({info.Datasets.Name}, 'X_tf'))
                    Xtf = h5read(h5file, '/X_tf');
                else
                    % try fp16 or alternate name
                    Xtf = h5read(h5file, '/X_tf_fp16');
                end
            catch ME
                error('Failed to read HDF5: %s', ME.message);
            end
            % assume SrsParams_0 exists
            try
                SrsParams0 = h5read(h5file, '/SrsParams_0');
            catch
                % try SrsParams
                try
                    SrsParams0 = h5read(h5file, '/SrsParams');
                catch
                    warning('Could not find SrsParams in HDF5. You may need to construct pdu/carrier manually.');
                    SrsParams0 = [];
                end
            end

            % If SrsParams0 has fields consistent with pdu, attempt to build pdu/carrier
            % For debugging, call detSrs directly if available
            try
                pduList = {}; % user may need to fill
                % try calling detSrs directly
                detOut = detSrs({}, srsTable, struct('N_slot_frame_mu',0,'N_symb_slot',size(Xtf,2),'idxSlotInFrame',0,'idxFrame',0,'delta_f',30e3), Xtf);
            catch ME
                warning('detSrs call failed: %s', ME.message);
                detOut = [];
            end
            % save TV-level debug
            outMat = fullfile(outDir, ['srs_debug_tv_' datestr(now,'yyyymmdd_HHMMSS') '.mat']);
            save(outMat, 'Xtf','SrsParams0','detOut','-v7.3');
            fprintf('Saved debug MAT to %s\n', outMat);

        otherwise
            error('Unknown mode %s. Use ''gen'' or ''tv''.', mode);
    end
end
