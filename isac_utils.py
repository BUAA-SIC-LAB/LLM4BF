import numpy as np


class ISACConfig:
    """
    存放ISAC相关的所有常数参数。
    """
    def __init__(self):
        self.Nt = 12          # 发射天线数
        self.Nr = 10          # 接收天线数
        self.L = 10            # DFRC 帧长
        self.lambda_ = 1.0
        self.d = self.lambda_ / 2  # 半波长间距
        self.alpha = 0.5

        # 噪声功率（dBm），默认都是 0 dBm
        self.sigma2C_dBm = 0.0
        self.sigma2R_dBm = 0.0

        # 缩放因子，和你 MATLAB 完全一致
        self.scale_factor_W = 100.0
        self.scale_factor_CRB = 100.0

        self.n_t = np.arange(-(self.Nt - 1) / 2.0, (self.Nt - 1) / 2.0 + 1.0, 1.0, dtype=np.float64).reshape(-1, 1)
        self.n_r = np.arange(-(self.Nr - 1) / 2.0, (self.Nr - 1) / 2.0 + 1.0, 1.0, dtype=np.float64).reshape(-1, 1)

    @property
    def k0(self):
        # 波数 k0 = 2π / λ
        return 2.0 * np.pi / self.lambda_

    @property
    def sigma2C(self):
        # dBm -> Watt: 10^((dBm - 30)/10)
        return 10.0 ** ((self.sigma2C_dBm - 30.0) / 10.0)

    @property
    def sigma2R(self):
        return 10.0 ** ((self.sigma2R_dBm - 30.0) / 10.0)

    def PT(self, P_dB):
        return 10.0 ** ((P_dB - 30.0) / 10.0)

    def Gamma(self, Gamma):
        return 10.0 ** (Gamma / 10.0)


def compact_vector_to_W(w_vec):
    """
    将一个长度为2Nt的实数向量还原成NtxNt的Hermitian复矩阵 W
    w_vec = [Re(w_1), ..., Re(w_Nt), Im(w_1), ..., Im(w_Nt)]
    """

    Nt = w_vec.size // 2
    w_real = w_vec[:Nt]
    w_imag = w_vec[Nt:]

    # 复向量 w ∈ C^{Nt}
    w = w_real + 1j * w_imag  # shape: (Nt,)

    # 构造 W = w w^H
    W = np.outer(w, np.conjugate(w))  # (Nt, Nt)

    return W


def vectors_to_W_stack(w_pred_vectors_scaled, config, K):
    """
    将模型输出的紧凑向量还原成(K, Nt, Nt)的复矩阵堆叠。
    需要注意，模型的输出W是经过缩放的，需要还原到原始的数据

    输入:
        W_pred_vectors_scaled: 经过处理后的模型原始输出，list形式，长度为2*K*Nt
        config: ISACConfig实例（提供Nt和scale_factor_W）
        K: 用户个数

    输出:
        W_stack: 形状(K, Nt, Nt)，还原后的复数Hermitian矩阵，每个W_k已经除以scale_factor_W
    """
    Nt = config.Nt
    scale_factor_W = config.scale_factor_W
    arr = np.asarray(w_pred_vectors_scaled, dtype=float)

    # 还原成真实 W
    W_stack = np.zeros((K, Nt, Nt), dtype=np.complex128)
    for k in range(K):
        base_index = k * 2 * Nt
        w_unscaled = arr[base_index: base_index + 2 * Nt] / scale_factor_W
        Wk = compact_vector_to_W(w_unscaled)
        W_stack[k] = Wk
    return W_stack


def compute_channel_H(input_obj):
    """
    从输入中解析信道矩阵H
    """
    H_real = np.asarray(input_obj["H_real"], dtype=float)
    H_imag = np.asarray(input_obj["H_imag"], dtype=float)
    H = H_real + 1j * H_imag
    return H


def compute_radar_A(config, theta):
    """
    计算CRB的雷达角矩阵A及其导数Ȧ
    """
    k0 = config.k0
    d = config.d

    n_t = config.n_t
    n_r = config.n_r


    # 阵列方向矢量
    a_theta = np.exp(1j * k0 * d * n_t * np.sin(theta))  # (Nt,1)
    b_theta = np.exp(1j * k0 * d * n_r * np.sin(theta))  # (Nr,1)

    # A(θ) = b(θ) a^H(θ)
    A = b_theta @ a_theta.conj().T

    # A点对θ的导数：Ȧ(θ) = ḃ(θ)a^H(θ) + b(θ)ȧ^H(θ)
    a_dot = 1j * k0 * d * n_t * np.cos(theta) * a_theta
    b_dot = 1j * k0 * d * n_r * np.cos(theta) * b_theta
    Ad = b_dot @ a_theta.conj().T + b_theta @ a_dot.conj().T

    return A, Ad


def compute_crb_for_sample(config, theta, W_stack):
    """
    按照MATLAB计算一条样本的CRB

    参数
    ----
    config : ISACConfig
        上面那个配置类的实例，里面存系统常数、噪声和缩放因子。

    theta : 雷达角的弧度值

    W_stack : 解析好的模型输出Wk，每个用户的波束赋形矩阵

    返回
    ----
    crb_scaled : float
        = CRB_deg2 * config.scale_factor_CRB
        可以直接和你 MATLAB 生成数据里的 'objective' 对比。
    """
    L = config.L
    alpha = config.alpha

    # ----- 噪声从分贝转换到功率 -----
    sigma2R = config.sigma2R

    # R_X = sum_k W_k
    RX = np.sum(W_stack, axis=0)

    # ----- 中心对称阵列：n_t, n_r -----
    A, Ad = compute_radar_A(config, theta)

    # ----- 按MATLAB公式算CRB -----
    # 分子
    num_CRB = sigma2R * np.trace(A.conj().T @ A @ RX)

    # 分母
    term1 = np.trace(Ad.conj().T @ Ad @ RX)
    term2 = np.trace(A.conj().T @ A @ RX)
    term3 = np.trace(Ad.conj().T @ A @ RX)
    den_CRB = 2.0 * (abs(alpha) ** 2) * L * (term1 * term2 - abs(term3) ** 2)

    num_real = float(np.real(num_CRB))
    den_real = float(np.real(den_CRB))

    # 和MATLAB一样，避免数值炸掉
    if den_real <= 0 or abs(den_real) < 1e-12:
        CRB_rad = float("inf")
    else:
        CRB_rad = num_real / den_real

    # rad^2 -> deg^2，再开方，得到 CRB_deg2（角度标准差，度）
    CRB_deg = CRB_rad * (180.0 / np.pi) ** 2
    CRB_deg2 = float(np.sqrt(max(CRB_deg, 0.0)))

    # 再乘上缩放因子，和数据集里的 objective 对齐
    crb_scaled = CRB_deg2 * config.scale_factor_CRB
    # 保留三位小数，和数据集的结果保持一致
    crb_scaled = float(np.round(crb_scaled, 3))

    return crb_scaled
