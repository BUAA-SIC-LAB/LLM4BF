
clear; clc; close all;

%% ========== dataset parameter ==========
scale_factor_W = 1e4;
scale_factor_CRB = 100;
num_samples = 57;

% Number of decimal places
save_digits = 3;
solve_fail = 0;

% Number of transmit and receive antennas
Nt = 12;
Nr = 10;

% user number K（K < N_t < N_r）
K = 7;

%% ========== Read old data ==========
filename = sprintf('ISAC_Dataset_Wk_%d_RL.mat', K);
if exist(filename, 'file')
    fprintf('An existing file was detected: %s\n', filename);
    fprintf('Loading old data...\n');
    old_data = load(filename);
    
    % 取出旧数据矩阵
    old_features = old_data.final_features;
    old_labels = old_data.final_labels;
    
    fprintf('Number of samples in the old file: %d\n', size(old_features, 1));
else
    fprintf('File not found %s，Create new file。\n', filename);
    old_features = [];
    old_labels = [];
end

%% ========== Pre-allocation ==========

% Input: [theta(1), PT_dBm(1), Gamma(1), H_real, H_imag]
feat_dim = 1 + 1 + 1 + Nt*K + Nt*K;

% Note that the user beamforming matrix W is a Hermitian complex matrix.
% Although the dimension of W is N_txN_t, only the diagonal and upper triangular elements need to be stored, 
% and the imaginary part of the diagonal is zero. 
% The upper triangular part contains N_t(N_t-1)/2 elements, which correspond to N_t(N_t-1)/2 real parts and imaginary parts.

m = Nt * (Nt - 1) / 2;

% The length of the real valued segment for each user equals the real values on the diagonal 
% plus the real values in the upper triangular part: diag_real + upper_real
real_len_per_user = Nt + m;
% The length of the imaginary valued segment for each user excludes the diagonal 
% because the diagonal entries are zero and do not require storage: upper_imag
imag_len_per_user = m;

% The total output length for each user is obtained accordingly: Nt(Nt-1)/2 * 2 + Nt = Nt^2
compact_len_per_user = real_len_per_user + imag_len_per_user;
% The total label length of W for all users is computed by summing over all users: Nt^2 * K
W_dim = compact_len_per_user * K;

% In addition, CRB is incorporated as an auxiliary prediction task.
label_dim = W_dim + 1;

data_features = zeros(num_samples, feat_dim);
data_labels = zeros(num_samples, label_dim);
valid_indices = false(num_samples, 1);

%% ========== System model ==========

% Length of DFRC frame
L = 10;

sigma2C_dBm = 0;     % Communication noise variance
sigma2R_dBm = 0;     % Radar noise variance

sigma2C = 10^((sigma2C_dBm - 30)/10);
sigma2R = 10^((sigma2R_dBm - 30)/10);

Gamma_dB = 10;
Gamma_lin = 10^(Gamma_dB/10);
Gamma_vec = Gamma_lin * ones(K,1);

% Array parameters correspond to a uniform linear array with half wavelength spacing.
lambda = 1;     % The wavelength is normalized and set to one.
d = lambda/2;   % The inter element spacing equals half of the wavelength.
k0 = 2*pi/lambda; % k0 = 2π/λ

% Generate the steering vectors for the transmit and receive arrays a(θ) and b(θ)
n_t = (-((Nt-1)/2):((Nt-1)/2)).';
n_r = (-((Nr-1)/2):((Nr-1)/2)).';

% Reflection coefficient α

%% ========== Dataset generation ==========
for loop_idx = 1:num_samples
    % A. Randomized parameter power
    PT_dBm_min = 12;
    PT_dBm_max = 13;
    PT_dBm = PT_dBm_min + (PT_dBm_max - PT_dBm_min) * rand();
    PT = 10^((PT_dBm - 30)/10);

    % B. Randomized channel noise
    % H ∈ C^{K×N_t}
    % Each row represents a channel from a user to a base station transmit array. 
    % Assuming that h_k are independent, they are generated using a Rayleigh channel simulation.
    H = (randn(K, Nt) + 1j * randn(K, Nt)) / sqrt(2);
    % Add AWGN noise to the channel matrix H
    Z_C = sqrt(sigma2C/2) * (randn(K, Nt) + 1j * randn(K, Nt));
    H = H + Z_C;

    % C. Randomized degreen
    theta_deg_min = 20;
    theta_deg_max = 30;
    theta_deg = theta_deg_min + (theta_deg_max - theta_deg_min) * rand();
    theta = deg2rad(theta_deg);
    % a(θ) ∈ C^{N_t×1}
    a_theta = exp(1j * k0 * d * n_t * sin(theta));
    % b(θ) ∈ C^{N_r×1}
    b_theta = exp(1j * k0 * d * n_r * sin(theta));
    % G = α * b(θ) * a^H(θ) ∈ C^{N_r×N_t}
    G = alpha * (b_theta * a_theta');
    A = b_theta * a_theta';         % A(θ) ≜ b(θ)a^H(θ)
    % Ȧ(θ) = ḃ(θ)a^H(θ) + b(θ)ȧ^H(θ)
    a_dot = 1j * k0 * d * n_t * cos(theta) .* a_theta;
    b_dot = 1j * k0 * d * n_r * cos(theta) .* b_theta;
    Ad = b_dot * a_theta' + b_theta * a_dot';  % Ȧ(θ) 

    % Q_k = h_k h_k^H
    Q = zeros(Nt, Nt, K);
    for k = 1:K
        hk = H(k,:).';
        Q(:,:,k) = hk * hk';
    end

    % D. Solving SDP Problems Using CVX
    % Variables: W_k (Nt×Nt，Hermitian PSD), t
    % Objective: maximize t, which is equivalently formulated as minimizing -t
    % Constraints:
    %   1) Schur complement based LMI corresponding to CRB maximization
    %   2) SINR constraint for each user
    %   3) Total power constraint,  sum_k tr(W_k) ≤ P_T
    %   4) W_k ⪰ 0

    cvx_begin quiet
        cvx_precision best 
    
        variable W(Nt, Nt, K) hermitian semidefinite
        variable t 
        
        % === R_X = sum_k W_k ===
        expression RX(Nt, Nt)
        RX = zeros(Nt, Nt);
        for k = 1:K
            RX = RX + W(:,:,k);
        end
        
        % === (2) Semi-positive definite matrix constraints  ===
        a11 = real(trace(Ad' * Ad * RX));      % tr(Ȧ^H Ȧ R_X)
        a12 = trace(Ad' * A  * RX);           % tr(Ȧ^H A R_X)
        a22 = real(trace(A'  * A  * RX));     % tr(A^H A R_X)
        
        LMI = [a11 - t,   a12;
               conj(a12), a22];
        
        LMI == hermitian_semidefinite(2); 
        
        % === (3) SINR ===
        for k = 1:K
            Gammak = Gamma_vec(k);
            Qk = Q(:,:,k);
            
            % tr(Q_k W_k)
            signal_k = trace(Qk * W(:,:,k));
            
            % Γ_k * sum_{i≠k} tr(Q_k W_i)
            interference_k = 0;
            for i = 1:K
                if i ~= k
                    interference_k = interference_k + trace(Qk * W(:,:,i));
                end
            end
            interference_k = Gammak * interference_k;
            
            % Γ_k σ_C^2
            noise_k = Gammak * sigma2C;
            
            % tr(Q_k W_k) - Γ_k * sum_{i≠k} tr(Q_k W_i) ≥ Γ_k σ_C^2
            real(signal_k - interference_k) >= noise_k;
        end
        
        % === (4) Power: sum_k tr(W_k) ≤ P_T ===
        total_power = 0;
        for k = 1:K
            total_power = total_power + trace(W(:,:,k));
        end
        real(total_power) <= PT;
        
        % === (5) Objective：minimize -t ===
        minimize(-t)
    cvx_end
    
    % E. Data Verification and Collection
    if strcmp(cvx_status, 'Solved')
        valid_indices(loop_idx) = true;

        % Find the optimal solution
        W_opt = full(W);    % Nt x Nt x K
        RX_opt = full(RX);  % Nt x Nt

        % Calculate CRB(θ)
        num_CRB = sigma2R * trace(A' * A * RX_opt);
        term1 = trace(Ad' * Ad * RX_opt);
        term2 = trace(A'  * A  * RX_opt);
        term3 = trace(Ad' * A  * RX_opt);
        
        den_CRB = 2 * abs(alpha)^2 * L * ( term1 * term2 - abs(term3)^2 );
        CRB_rad = real(num_CRB / den_CRB);
        
        CRB_deg  = CRB_rad * (180/pi)^2;
        CRB_deg2   = sqrt(CRB_deg);

        % Constructing input features: Partitioning the channel gain H by user
        % H_feat:
        %   user_1: [Re(h1_1..h1_Nt), Im(h1_1..h1_Nt)]
        %   user_2: [Re(...), Im(...)]

        H_feat = zeros(1, 2 * Nt * K);
        for k = 1:K
            base = (k-1) * 2 * Nt;
            H_feat(base + (1:Nt)) = real(H(k, :));
            H_feat(base + (Nt+1 : 2*Nt)) = imag(H(k, :));
        end
       
        % [theta, PT_dBm, Gamma_lin, H_real_flat, H_imag_flat]
        feature_vector = [theta, PT_dBm, Gamma_lin, H_feat];
        
        % Structure tag: Hermitian compressed coding of W (user-based blocking)
        W_scaled = W_opt * scale_factor_W;
        CRB_scaled = CRB_deg2 * scale_factor_CRB;

        % One block per user: [diag_real, upper_real, upper_imag]，length is Nt^2
        W_compact_all = zeros(1, compact_len_per_user * K);

        for k = 1:K
            % Retrieve the matrix for each user k
            Wk = W_scaled(:,:,k);

            % Numerically stable, forced Hermitian
            Wk = (Wk + Wk') / 2;

            % Compression coding
            [real_part, imag_part] = encode_hermitian_compact(Wk);
            user_block = [real_part, imag_part];   % 先实后虚
            
            idx = (k-1) * compact_len_per_user + (1:compact_len_per_user);
            W_compact_all(idx) = user_block;
        end

        label_vector = [W_compact_all, CRB_scaled];

        % Standardize the number of decimal places before writing to the dataset
        feature_vector = round(feature_vector, save_digits);
        label_vector   = round(label_vector, save_digits);

        data_features(loop_idx, :) = feature_vector;
        data_labels(loop_idx, :) = label_vector;

        if mod(loop_idx, 50) == 0
            fprintf('Loop %d: OK. CRB=%.4f\n', loop_idx, CRB_deg2);
        end

    else
        solve_fail = solve_fail + 1;
        valid_indices(loop_idx) = false;
        if mod(loop_idx, 10) == 0
            fprintf('Loop %d/%d: Failed (%s)\n', loop_idx, num_samples, cvx_status);
        end
    end
end

%% ========== Save as a dataset in a .mat file ==========
% Extract valid data
current_valid_features = data_features(valid_indices, :);
current_valid_labels = data_labels(valid_indices, :);

fprintf('\nThe dataset has been generated.\nNumber of valid samples:%d\nNumber of failed samples:%d\n', ...
    size(current_valid_features, 1), solve_fail);

final_features = [old_features; current_valid_features];
final_labels   = [old_labels; current_valid_labels];
fprintf('Total sample size after merging: %d\n', size(final_features, 1));

% Save all necessary information
save(filename, ...
    'final_features', 'final_labels', 'save_digits', ...
    'scale_factor_W', 'scale_factor_CRB', ...
    'Nt', 'Nr', 'K', 'PT_dBm_min', 'PT_dBm_max', ...
    'theta_deg_min', 'theta_deg_max', ...
    'real_len_per_user', 'imag_len_per_user', 'compact_len_per_user'); 

fprintf('The data has been saved as: %s\n', filename);

TEMP = load(filename) %加载数据集


%% ============================================================
% Local Functions: Hermitian Compression Coding
function [real_part, imag_part] = encode_hermitian_compact(Wk)
    % Input:
    %   Wk: Nt x Nt complex Hermitian matrix
    %
    % Output, used as the compressed block for each user:
    %   real_part = [ real(diag(Wk)), real(upper_triangle_no_diag(Wk)) ]
    %   imag_part = [ imag(upper_triangle_no_diag(Wk)) ]
    %
    % Key points:
    %   - For a Hermitian matrix, the imaginary part of the diagonal should be zero, so the imaginary diagonal entries are not stored.
    %   - The lower triangular part is fully determined by the conjugate of the upper triangular part, so the lower triangular entries are not stored.
    Nt = size(Wk, 1);

    diag_r = real(diag(Wk)).';

    mask = triu(true(Nt), 1);
    upper = Wk(mask);              % m x 1

    upper_re = real(upper).';      % 1 x m
    upper_im = imag(upper).';      % 1 x m

    real_part = [diag_r, upper_re];
    imag_part = upper_im;
end
