# SquareRoot Filter Based VIO 分享

> 基于平方根 EKF 的视觉惯性里程计：滤波流程、与 EKF 的等价性推导及 openvins 工程实现要点。

---

## 目录

- [1、基于滤波的 VIO 算法流程](#1基于滤波的vio算法流程)
- [2、SR-EKF 与 EKF 的等价性](#2sr-ekf与ekf的等价性)
- [3、SR-EKF VIO 的工程实现](#3sr-ekf-vio的工程实现)
- [4、总结与讨论](#4总结与讨论)

---

## 1、基于滤波的 VIO 算法流程

VIO 算法根据求解方式可分为**优化**和**滤波**两大类：优化方法构建包含所有测量数据、运动模型和约束条件的全局目标函数，通过多次迭代求解，在状态初值不佳时仍可获得较精确的估计，但计算量较大；滤波方法基于贝叶斯滤波理论（卡尔曼滤波及其衍生），递推估计系统状态，一般每观测只更新一次，计算量较低，更适合实时场景。

本文分享的 **SquareRoot EKF Based VIO** 属于滤波方法，是 EKF 的衍生算法。

基于滤波的 VIO 包含四个主要步骤：

- **IMU Propagation** — 状态与协方差递推  
- **State Augmentation And Clone** — 状态扩增与克隆  
- **Visual Measurement Update** — 视觉量测更新  
- **State Marginalization** — 状态边缘化

### 1.1 IMU Propagation

根据当前帧与上一帧之间的 IMU 测量，推算当前时刻的 IMU 状态与协方差。优化框架中也有类似步骤，但协方差通常从零开始递推，而非继承上一帧。

$$\hat{x}_{k+1|k} = \Phi_k \hat{x}_{k|k} \\ P_{k+1|k} = \Phi_k P_{k|k}\Phi_k^T + W_k$$

### 1.2 State Augmentation And Clone

将递推得到的当前帧 IMU 状态及协方差复制到滑窗状态中，供后续视觉更新使用。

$$\begin{align*}  x &= \begin{bmatrix} \hat{x}_{k+1|k} \\ \hat{x}_{C,k+1|k} \\ \hat{x}_{k|k}  \end{bmatrix} \\   P &= \begin{bmatrix} \Phi_k P_{k|k} \Phi_k^T + W_k & \Phi_k P_{k|k} \Phi_k^T + W_k & \Phi_k P_{k|k} \\  \Phi_k P_{k|k} \Phi_k^T + W_k & \Phi_k P_{k|k} \Phi_k^T + W_k & \Phi_k P_{k|k} \\  P_{k|k}\Phi_k^T &  P_{k|k}\Phi_k^T & P_{k|k} \end{bmatrix}  \end{align*}$$

### 1.3 Visual Measurement Update

将视觉观测约束用于更新系统状态与协方差。

$$\begin{align*} \hat{x}_{k|k} &= \hat{x}_{k|k-1} + K_{k}r_{k} \\ P_{k|k} &= P_{k|k-1} - P_{k|k-1}H^T_{k}S^{-1}_{k}H_{k}P_{k|k-1} \end{align*}$$

其中，

$$\begin{align*} r_{k} &= z_{k} - H_{k}r_{k} \\ S_{k} &= H_{k}P_{k|k-1} H^T_{k}+R_{k}\\ K &= P_{k|k-1}H^T_{k}S^{-1}_{k} \end{align*}$$

### 1.4 State Marginalization

将超出滑窗的位姿以及跟踪丢失的 SLAM 特征从状态向量与协方差中移除。

$$x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} , P = \begin{bmatrix} P_{11}& P_{12} &P_{13}\\ P_{21}&P_{22}&P_{22} \\ P_{31}&P_{32}&P_{33} \end{bmatrix}$$

边缘化掉 $x_3$，

$$x_R = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} , P_R = \begin{bmatrix}P_{11}& P_{12} \\ P_{21}&P_{22} \end{bmatrix}$$

边缘化掉 $x_2$，

$$x_R = \begin{bmatrix} x_1 \\ x_3 \end{bmatrix} , P_R = \begin{bmatrix}P_{11}& P_{13} \\ P_{31}&P_{33} \end{bmatrix}$$

---

## 2、SR-EKF 与 EKF 的等价性

**SR-EKF**（Square Root EKF）与 EKF 在数学上等价：SRF 将 EKF 中的矩阵求逆等操作转化为等价的 QR 运算，结果一致。SRF 的两点优势：

1. **数值稳定**：存储协方差的平方根（上三角），保证协方差正定。  
2. **计算友好**：利用观测矩阵稀疏性及 float 精度，降低更新计算量。

下面按四个主要步骤给出 EKF 与 SRF 的等价性推导。

### 2.1 状态递推过程等价性

| _EKF_ | _SRF_ |
| --- | --- |
| $\hat{x}_{k+1\mid k} = \Phi_k \hat{x}_{k\mid k} \\ P_{k+1\mid k} = \Phi_k P_{k\mid k}\Phi_k^T + W_k$ | $\begin{bmatrix} W_k^{\frac{1}{2}} \\ U_{k\mid k}\Phi_k^T \end{bmatrix} \overset{QR}{=} Q_k \begin{bmatrix} U_{k+1 \mid k} \\ 0 \end{bmatrix}$ |
| 等价性证明 |  |
| 对于SRF形式的左边，<br>$$\begin{align*} \begin{bmatrix} W_k^{\frac{T}{2}} & \Phi_k U^T_{k\mid k}  \end{bmatrix} \begin{bmatrix} W_k^{\frac{1}{2}} \\  U_{k\mid k} \Phi_k^T  \end{bmatrix}  &= \Phi_k U^T_{k\mid k} U_{k\mid k}\Phi_k^T + W_k^{\frac{T}{2}}W_k^{\frac{1}{2}} \\ &=  \Phi_k P_{k\mid k} \Phi_k^T + W_k \\ &= P_{k+1\mid k} \end{align*}$$<br>对于SRF形式的右边，<br>$$\begin{align*} \begin{bmatrix}  U^T_{k+1\mid k} & 0 \end{bmatrix}  Q^T_kQ_k \begin{bmatrix}  U^T_{k+1\mid k} \\0 \end{bmatrix}   &= U^T_{k+1\mid k}U_{k+1\mid k}  \\ &= P_{k+1\mid k} \end{align*}$$ |  |

### 2.2 状态扩增过程等价性

| _EKF_ | _SRF_ |
| --- | --- |
| $$\begin{align*}  x &= \begin{bmatrix} \hat{x}_{k+1\mid k} \\ \hat{x}_{C,k+1\mid k} \\ \hat{x}_{k\mid k}  \end{bmatrix} \\   P &= \begin{bmatrix} \Phi_k P_{k\mid k} \Phi_k^T + W_k & \Phi_k P_{k\mid k} \Phi_k^T + W_k & \Phi_k P_{k\mid k} \\  \Phi_k P_{k\mid k} \Phi_k^T + W_k & \Phi_k P_{k\mid k} \Phi_k^T + W_k & \Phi_k P_{k\mid k} \\  P_{k\mid k}\Phi_k^T &  P_{k\mid k}\Phi_k^T & P_{k\mid k} \end{bmatrix}  \end{align*}$$ | $$\begin{align*} x &= \begin{bmatrix} \hat{x}_{k+1\mid k} \\ \hat{x}_{k\mid k} \end{bmatrix} \\ \begin{bmatrix} W_k^{\frac{1}{2}} & 0 \\ U_{k\mid k}\Phi_k^T & U_{k\mid k} \end{bmatrix} &\overset{QR}{=} Q \begin{bmatrix} U_{k+1\mid k} & U_{k\mid k,1} \\ 0 &U_{k\mid k.2} \end{bmatrix}  \\ x &= \begin{bmatrix} \hat{x}_{k+1\mid k} \\ \hat{x}_{C,k+1\mid k}\\ \hat{x}_{k\mid k} \end{bmatrix} \\ U&= \begin{bmatrix} U_{k+1\mid k} & U_{k+1\mid k} & U_{k\mid k,1} \\ 0&0&U_{k\mid k,2} \\ 0&0&0 \end{bmatrix} \end{align*}$$ |
| 等价性证明 |  |
| 由于，<br>$$\begin{align*} \begin{bmatrix} W_k^{\frac{T}{2}} & \Phi_k U^T_{k\mid k} \\ 0 & U^T_{k\mid k} \end{bmatrix} \begin{bmatrix} W_k^{\frac{1}{2}} & 0 \\  U_{k\mid k}\Phi_k^T  & U_{k\mid k} \end{bmatrix} &= \begin{bmatrix} U^T_{k+1\mid k} & 0 \\ U^T_{k\mid k,1}  &U^T_{k\mid k,2} \end{bmatrix} Q^TQ \begin{bmatrix} U_{k+1\mid k} & U_{k\mid k,1} \\ 0 &U_{k\mid k,2} \end{bmatrix}  \\ \begin{bmatrix} \Phi_k P_{k\mid k}\Phi^T_k & \Phi_k P_{k\mid k} \\ P_{k\mid k}\Phi_k^T & P_{k\mid k} \end{bmatrix} &= \begin{bmatrix} U^T_{k+1\mid k}U_{k+1\mid k} & U^T_{k+1\mid k} U_{k\mid k,1} \\U^T_{k\mid k}U_{k+1\mid k,1} & U^T_{k\mid k,1}U_{k\mid k,1}+U^T_{k\mid k,2}U_{k\mid k,2} \end{bmatrix} \end{align*}$$<br>所以，<br>$$\begin{align*} U^TU &= \begin{bmatrix} U^T_{k+1\mid k} & 0&0 \\ U^T_{k+1\mid k} &0 &0 \\U^T_{k\mid k,1} & U^T_{k\mid k,2} & 0 \end{bmatrix}  \begin{bmatrix} U_{k+1\mid k} & U_{k+1\mid k} & U_{k\mid k,1} \\ 0&0&U_{k\mid k,2} \\ 0&0&0 \end{bmatrix}\\ &= \begin{bmatrix} U^T_{k+1\mid k}U_{k+1\mid k} &  U^T_{k+1\mid k}U_{k+1\mid k} & U^T_{k+1\mid k}U_{k\| k,1} \\ U^T_{k+1\mid k}U_{k+1\mid k} &  U^T_{k+1\mid k}U_{k+1\mid k} & U^T_{k+1\mid k}U_{k\| k,1} \\ U^T_{k\mid k,1}U_{k+1\mid k} &  U^T_{k\mid k,1}U_{k+1\mid k} & U^T_{k\mid k,1}U_{k\| k,1}+U^T_{k\mid k,2}U_{k\mid k,2} \end{bmatrix}  \\ &= \begin{bmatrix} \Phi_k P_{k\mid k} \Phi_k^T + W_k & \Phi_k P_{k\mid k} \Phi_k^T + W_k & \Phi_k P_{k\mid k} \\  \Phi_k P_{k\mid k} \Phi_k^T + W_k & \Phi_k P_{k\mid k} \Phi_k^T + W_k & \Phi_k P_{k\mid k} \\  P_{k\mid k}\Phi_k^T &  P_{k\mid k}\Phi_k^T & P_{k\mid k} \end{bmatrix}  \\  &= P \end{align*}$$ |  |

### 2.3 状态更新过程等价性

| _EKF_ | _SRF_ |
| --- | --- |
| $$\begin{align*} \hat{x}_{k\mid k} &= \hat{x}_{k\mid k-1} + K_{k}r_{k} \\ P_{k\mid k} &= P_{k\mid k-1} - P_{k\mid k-1}H^T_{k}S^{-1}_{k}H_{k}P_{k\mid k-1} \end{align*}$$<br>其中，<br>$$\begin{align*} r_{k} &= z_{k} - H_{k}r_{k} \\ S_{k} &= H_{k}P_{k\mid k-1} H^T_{k}+R_{k}\\ K &= P_{k\mid k-1}H^T_{k}S^{-1}_{k} \end{align*}$$<br>对于增益$K$的计算公式，还有另一种写法：<br>$K = P_{k\mid k}H^T_{k}R_{k}^{-1}$ | $$\begin{align*} U_{k\mid k} &= F^{-T}_kU_{k\mid k-1} \\ \hat{x}_{k\mid k} &= \hat{x}_{k\mid k-1} + U^T_{k\mid k}U_{k\mid k}H^T_kR^{-1}_kr_k  \end{align*}$$<br>其中下三角矩阵$F_k$的计算过程是：<br>$$\begin{align*} \begin{bmatrix} R_k^{-\frac{1}{2}}H_kU^T_{k\mid k-1}\\ I \end{bmatrix} \overset{PQR}{=} Q\begin{bmatrix} 0\\F_k \end{bmatrix} \end{align*}$$ |
| 等价性证明 |  |
| $$\begin{align*} P_{k\mid k} &= P_{k\mid k-1} - P_{k\mid k-1}H^T_kS^{-1}_kH_kP_{k\mid k-1}\\ &= U^T_{k\mid k-1}U_{k\mid k-1} - U^T_{k\mid k-1}U_{k\mid k-1}H^T_{k}(H_{k}U^T_{k\mid k-1}U_{k\mid k-1}H^T_k + R_k)^{-1}H_kU^T_{k\mid k-1}U_{k\mid k-1}  \\ &= U^T_{k\mid k-1}(I-U_{k\mid k-1}H^T_k(H_kU^T_{k\mid k-1}U_{k\mid k-1}H^T_k + R_k)^{-1}H_kU^T_{k\mid k-1})U_{k\mid k-1}\\ \end{align*}$$<br>按照矩阵求逆引理，<br>$(A^{-1} + BD^{-1}C)^{-1} = A-AB(D+CAB)^{-1}CA$<br>令，<br>$$\begin{align*} A &= I \\ B &= U_{k\mid k-1}H^T_k \\ C &= H_kU^T_{k\mid k-1} \\ D &= R_{k} \end{align*}$$<br>则有，<br>$(I+U_{k\mid k-1}H^T_kR_k^{-1}H_kU^T_{k\mid k-1})^{-1} = I-U_{k\mid k-1}H^T_k(H_kU^T_{k\mid k-1}U_{k\mid k-1}H^T_k + R_k)^{-1}H_kU^T_{k\mid k-1}$<br>又，<br>$I+U_{k\mid k-1}H^T_kR_k^{-1}H_kU^T_{k\mid k-1} =  \begin{bmatrix} R_k^{-\frac{1}{2}}H_kU^T_{k\mid k-1} \\ I \end{bmatrix}^T \begin{bmatrix} R_k^{-\frac{1}{2}}H_kU^T_{k\mid k-1} \\ I \end{bmatrix}  = F_k^T F_k$<br>所以，<br>$$\begin{align*} P_{k\mid k} &= U^T_{k\mid k-1}(F^T_kF_k)^{-1}U_{k\mid k-1}  \\ &= U^T_{k\mid k-1}F^{-1}_kF^{-T}_kU_{k\mid k-1}\\ &= U^T_{k\mid k}U_{k\mid k} \end{align*}$$ |  |

### 2.4 状态边缘化过程等价性

| _EKF_ | _SRF_ |
| --- | --- |
| $x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} , P = \begin{bmatrix} P_{11}& P_{12} &P_{13}\\ P_{21}&P_{22}&P_{22} \\ P_{31}&P_{32}&P_{33} \end{bmatrix}$<br>边缘化掉$x_3$,<br>$x_R = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} , P_R = \begin{bmatrix}P_{11}& P_{12} \\ P_{21}&P_{22} \end{bmatrix}$<br>边缘化掉$x_2$,<br>$x_R = \begin{bmatrix} x_1 \\ x_3 \end{bmatrix} , P_R = \begin{bmatrix}P_{11}& P_{13} \\ P_{31}&P_{33} \end{bmatrix}$ | $x = \begin{bmatrix}x_1\\x_2\\x_3 \end{bmatrix}, U =\begin{bmatrix}U_{11}&U_{12}&U_{13}\\0&U_{22}&U_{23}\\0&0&U_{33} \end{bmatrix}$<br>边缘化掉$x_3$,<br>$x_R = \begin{bmatrix} x_1 \\x_2 \end{bmatrix}, U_R = \begin{bmatrix} U_{11}&U_{12} \\0&U_{22} \end{bmatrix}$<br>边缘化掉$x_2$,<br>$x_R = \begin{bmatrix} x_1 \\x_3 \end{bmatrix},  \begin{bmatrix} U_{11}&U_{13} \\0&U_{23}\\0&U_{33}  \end{bmatrix}\overset{QR}{=} Q\begin{bmatrix} U_{11}&U'_{13} \\0&U'_{23}\\0&0 \end{bmatrix}\\  U_R= \begin{bmatrix} U_{11}&U'_{13} \\0&U'_{23} \end{bmatrix}$ |
| 等价性证明 |  |
| 对于边缘化掉$x_2$,<br>由，<br>$$\begin{align*} U^TU &= \begin{bmatrix} U^T_{11}U_{11} & U^T_{11}U_{12}& U^T_{11}U_{13}  \\ U^T_{12}U_{11} & U^T_{12}U_{12}+ U^T_{22}U_{22}  & U^T_{12}U_{13}+U^T_{22}U_{23}  \\ U^T_{13}U_{11} & U^T_{13}U_{12}+U^T_{23}U_{22}  & U^T_{13}U_{13}+U^T_{23}U_{23}+U^T_{33}U_{33}  \end{bmatrix}\\ &=  \begin{bmatrix} P_{11}&P_{12}&P_{13}\\ P_{21}&P_{22}&P_{23}\\ P_{31}&P_{32}&P_{33}\\ \end{bmatrix} \\ &= P \end{align*}$$<br>有，<br>$$\begin{align*} U_R^TU_R &= \begin{bmatrix} U^T_{11}&0 \\ U'^T_{13}&U'^T_{23} \end{bmatrix} \begin{bmatrix} U_{11}&U'_{13}  \\ 0&U'_{23} \end{bmatrix}  \\ &=\begin{bmatrix} U^T_{11}&0&0 \\ U'^T_{13}&U'^T_{23}&0 \end{bmatrix} \begin{bmatrix} U_{11}&U'_{13}  \\ 0&U'_{23}  \\ 0&0 \end{bmatrix}\\ &=  \begin{bmatrix} U^T_{11}&0&0 \\ U^T_{13}&U^T_{23}&U^T_{33} \end{bmatrix} \begin{bmatrix} U_{11}&U_{13}  \\ 0&U_{23}  \\ 0&U_{33} \end{bmatrix}\\ &=  \begin{bmatrix} U^T_{11}U_{11} & U^T_{11}U_{13}  \\ U^T_{13}U_{11} & U^T_{13}U_{13}+U^T_{23}U_{23}+U^T_{33}U_{33} \end{bmatrix}\\ &=  \begin{bmatrix} P_{11}&P_{13}  \\ P_{31}&P_{33} \end{bmatrix}\\ &= P_R \end{align*}$$ |  |

---

## 3、SR-EKF VIO 的工程实现

SR-EKF VIO 基于 **openvins** 实现，整体流程如下。

![SR-EKF VIO 整体流程](doc/assets/01-sr-vio-flow.png)

与 openvins 一致，同属 **Hybrid-VIO**：同时包含 MSCKF 特征更新与 SLAM 特征更新。

### 3.1 状态向量定义

状态向量从前到后依次为：IMU 状态、标定参数、滑窗位姿、SLAM 特征。

$$\begin{align*} x &= \begin{bmatrix} x^T_{I_k} & x^T_{calib} & x^T_C & x^T_f  \end{bmatrix}^T \\ x_{I_k} &= \begin{bmatrix} ^{I_k}_G \bar{q}^T & ^Gp^T_{I_k} & ^Gv^T_{I_k}& b^T_g & b^T_a \end{bmatrix}^T \\ x_{calib} &= \begin{bmatrix} t_d & ^I_C\bar{q}^T & ^Cp^T_I & \xi^T \end{bmatrix}  \\ x_C &=  \begin{bmatrix} x^T_{T_k}& \ldots  & x^T_{T_{k-c}} \end{bmatrix}^T \\ x_f &= \begin{bmatrix} f^T_1& \ldots  & f^T_g \end{bmatrix}^T \end{align*}$$

在 EKF 的分块运算中状态顺序对计算量影响不大，openvins 中滑窗与 SLAM 特征实为混序；而在 SR-EKF 的 QR 中，被频繁边缘化的状态越靠后，计算量越小，因此 $x_f$ 置于队尾，滑窗状态按逆序排列。

### 3.2 IMU Propagation And State Augmentation

IMU 递推仅涉及 $x_{I_k}$，参与 QR 的只是全状态的一小部分。

$$\begin{bmatrix} W_k^{\frac{1}{2}} & 0 \\ U_{k|k} T_0 \Phi_k^T &  U_{k|k} T_1 \end{bmatrix}, \quad T_0 = \begin{bmatrix}1&0 & 0 \\ 0&...&0\\  0&0&0\end{bmatrix} ,\quad T_1 = \begin{bmatrix}0&0 & 0 \\ 0&...&0\\  0&0&1\end{bmatrix},\quad \text{T0 取前 k 列，T1 取剩余列}$$
$$\begin{bmatrix} W_k^{\frac{T}{2}} &\Phi_k T_0^TU_{k|k}^T  \\ 0 &  T_1^T U_{k|k}^T  \end{bmatrix}  \begin{bmatrix} W_k^{\frac{1}{2}} & 0 \\ U_{k|k} T_0 \Phi_k^T &  U_{k|k} T_1 \end{bmatrix} =  \begin{bmatrix}  \Phi_kT_0^T P_{k|k}T_0\Phi_k^T + W_k^{\frac{1}{2}}& \Phi_kT_0^TP_{k|k}T_1 \\ T_1^TP_{k|k}T_0\Phi_k^T&  T_1^TP_{k|k} T_1 \end{bmatrix}$$

**示意：**

![IMU Propagation](doc/assets/02-imu-prop.png)

> **说明**：该过程中矩阵 $U$ 的所有元素都会更新，计算量明显大于 EKF 形式。

状态 clone 与扩增可与上述 IMU propagation 合并处理。

**示意：**

![State Augmentation](doc/assets/03-state-aug.png)

### 3.3 视觉特征更新

#### 3.3.1 视觉特征预处理

按跟踪结果将特征分为三类：

| 类型 | 条件 |
|------|------|
| **MSCKF Feat** | 当前帧跟踪失败且可三角化 |
| **SLAM Feat** | 当前帧跟踪成功且已三角化 |
| **SLAM Delay Init Feat** | 在最老帧被观测且尚未三角化 |

![视觉特征分类](doc/assets/04-feat-class.png)

每种特征仅参与一次更新，不重复三角化。
    

#### 3.3.2 视觉特征观测矩阵

各类视觉特征的观测方程可统一写成：

$$\begin{align*} z_{m,k} &= h(x_k) + n_k \\ &= h_d(z_{n,k}, \xi) + n_k \\ &= h_d(h_p(^{C_k}p_f), \xi) + n_k \\ &= h_d(h_p(h_t(^Gp_f, ^{c_k}_GR, ^Gp_{C_k})), \xi) \\ &= h_d(h_p(h_t(h_r(\lambda, \ldots), ^{c_k}_GR, ^Gp_{C_k})),\xi)+n_k \end{align*}$$

其中：

- $z_k = h_d(z_{n,k}, \xi)$ — 畸变
- $z_{n,k} = h_p(^{C_k}p_f)$ — 投影
- $^{C_k}p_f = h_t(^Gp_f, ^{C_k}_GR,^Gp_{C_k})$ — 相机系到世界系
- $^Gp_f = h_r(\lambda, \ldots)$ — 地图点参数化

观测矩阵由链式求导得到：

$\frac{\partial{z_k}}{\partial{x}} = \frac{\partial h_d(\cdot)}{\partial z_{n,k}}  \frac{\partial h_p(\cdot)}{\partial ^{C_k}p_{f}}  \frac{\partial h_t(\cdot)}{\partial x} +   \frac{\partial h_d(\cdot)}{\partial z_{n,k}}  \frac{\partial h_p(\cdot)}{\partial ^{C_k}p_{f}}  \frac{\partial h_t(\cdot)}{\partial ^Gp_f} \frac{\partial h_r(\cdot)}{\partial x}$

求导细节见 [openvins 文档](https://docs.openvins.com/update-feat.html)。

#### 3.3.3 MSCKF 特征的特殊处理

MSCKF 特征的观测中不显含 3D 点参数，需用零空间投影对特征位置做边缘化。

$$\widetilde{z}_{m,k} \simeq H_x \widetilde{x}_k + H_f ^G\widetilde{p}_f + n_k, \quad H_f = \begin{bmatrix} Q_1 & Q_2 \end{bmatrix}  \begin{bmatrix} R_1 \\ 0 \end{bmatrix} = Q_1R_1$$

$$\widetilde{z}_{m,k}  \simeq H_x \widetilde{x}_k + Q_1R_1^G\widetilde{p}_f + n_k \\ \Rightarrow Q_2^T\widetilde{z}_m \simeq Q_2^TH_x\widetilde{x}_k + Q_2^TQ_1R_1^G\widetilde{p}_f + Q_2^Tn_k \\ \Rightarrow Q_2^T\widetilde{z}_m  \simeq Q_2^TH_x \widetilde{x}_k + Q_2^Tn_k \\ \Rightarrow \widetilde{z}_{0,k} \simeq H_{0,k}\widetilde{x}_k + n_{o,k}$$

MSCKF 更新维度较大，可再用 QR 压缩观测维度。

$$H_{o,k} = \begin{bmatrix} Q_1 & Q_2 \end{bmatrix} \begin{bmatrix} R_1 \\ 0 \end{bmatrix} = Q_1R_1$$

$$\widetilde{z}_{o,k} \simeq H_{o,k} \widetilde{x}_k + n_o \\ \widetilde{z}_{o,k} \simeq Q_1R_1 \widetilde{x}_k + n_o \\ Q_1^T\widetilde{z}_{o,k} \simeq Q_1^TQ_1R_1 \widetilde{x}_k + Q_1^Tn_o \\ Q_1^T \widetilde{z}_{o,k}  \simeq R_1  \widetilde{x}_k + Q_1^Tn_o \\ \Rightarrow \widetilde{z}_{n,k}  \simeq H_{n,k}  \widetilde{x}_k + n_n$$

#### 3.3.4 Permulate QR 过程

下三角矩阵 $F_k$ 由下式（PQR）得到：

$$\begin{align*} \begin{bmatrix} R_k^{-\frac{1}{2}}H_kU^T_{k|k-1}\\ I \end{bmatrix} \overset{PQR}{=} Q\begin{bmatrix} 0\\F_k \end{bmatrix} \end{align*}$$

做法：将所有视觉观测（SLAM / MSCKF / delay init）矩阵堆叠后做更新；先列变换再 QR，再对结果做行、列变换。为利用稀疏性，QR 前会对矩阵做行变换使其更接近上三角。

![Permulate QR](doc/assets/05-permulate-qr.png)

常用 QR 实现有两种：**Householder** 与 **Givens**。稠密矩阵下 Givens 与 Householder 计算量约 3 : 2，而 VIO 观测矩阵常仅约 20% 非零，实测中 Givens 在稀疏情形下更高效。

**Givens 示意：**

![Givens](doc/assets/06-givens.png)

```cpp
void FilterUtils::performQRGivens(Eigen::Ref<MatrixX> tempQR, int start_col, int num_cols)
{
  constexpr number_t tol_thre = std::is_same_v<number_t, float> ? 5e-8 : 1e-13;
  const size_t total_cols = tempQR.cols();
  if (num_cols == -1)
    num_cols = total_cols - start_col;
  Eigen::JacobiRotation<number_t> GR;
  for (int col = start_col; col < start_col + num_cols; ++col)
  {
    const size_t remaining_cols = total_cols - col;
    for (int row = (int)tempQR.rows() - 1; row > col; --row)
    {
      if (__builtin_expect(std::abs(tempQR(row, col)) < tol_thre, 0))
      {
        tempQR(row, col) = 0;
        continue;
      }
      GR.makeGivens(tempQR(row - 1, col), tempQR(row, col));
      tempQR.block(row - 1, col, 2, remaining_cols).applyOnTheLeft(0, 1, GR.adjoint());
    }
  }
  return;
}
```

**Householder 示意：**

![Householder](doc/assets/07-householder.png)

```cpp
void FilterUtils::performQRHouseholder(Eigen::Ref<MatrixX> tempQR, int start_col, int num_cols)
{
  const size_t total_cols = tempQR.cols();
  const size_t total_rows = tempQR.rows();
  if (num_cols == -1)
  {
    num_cols = tempQR.cols() - start_col;
  }
  size_t total_rank = start_col;
  VectorX tempVector(total_cols + 1);
  number_t *tempData = tempVector.data();
  for (size_t k = start_col; k < start_col + num_cols && total_rank < total_rows; ++k)
  {
    size_t remainingRows = total_rows - total_rank;
    size_t remainingCols = total_cols - k - 1;
    number_t beta;
    number_t hCoeff;
    tempQR.col(k).tail(remainingRows).makeHouseholderInPlace(hCoeff, beta);
    if (std::abs(beta) > 1e-6)
    {
      tempQR.coeffRef(total_rank, k) = beta;
      tempQR.bottomRightCorner(remainingRows, remainingCols).applyHouseholderOnTheLeft(tempQR.col(k).tail(remainingRows - 1), hCoeff, tempData + k + 1);
      total_rank += 1;
    }
    else
    {
      tempQR.coeffRef(total_rank, k) = 0;
    }
    tempQR.col(k).tail(remainingRows - 1).setZero();
  }
  return;
}
```

### 3.4 零速更新（ZUPT）

在静止段用伪零速观测约束状态，抑制 IMU 积分漂移。零速观测包括：零速度、零加速度、零角速度。

误差方程：

$$\widetilde{z} = \begin{bmatrix} a - (a_m -b_a -  ^{I_k}_GR^Gg-n_a) \\ w - (w_m-b_g-n_g) \\ 0-v \end{bmatrix}$$

观测矩阵：

$$\begin{align*} \frac{\partial{\widetilde{z}}}{\partial{^{I_k}_G\widetilde{R}}^G} &= -\lfloor{^{I_k}_GR^Gg\times}\rfloor \\ \frac{\partial{\widetilde{z}}}{\partial{b_a}} &= \frac{\partial{\widetilde{z}}}{\partial{b_g}} = I  \\ \frac{\partial{\widetilde{z}}}{\partial{v}} &= -I \end{align*}$$

ZUPT 观测维度小，$F$ 矩阵也可通过反向 QR 得到。

![ZUPT](doc/assets/08-zupt.png)

### 3.5 SLAM Feat Anchor Change

anchor 在最老帧上的 SLAM 特征，在最老帧被边缘化后需重新 anchor 到最新帧，其协方差也需做相应变换。

特征点在最老帧和最新帧下的表示：

$^Gp_f = ^{A_1}_GR^T {^{A_1}}p_f + ^Gp_{A_1} = ^{A_2}_GR^T {^{A_2}}p_f + ^Gp_{A_2}$

对上式求导，

$^G\widetilde{p}_f = H_{old}\widetilde{x}_{old} + H_{f_{old}}^{A_1}\widetilde{p}_{f} = H_{new}\widetilde{x}_{new} + H_{f_{new}}^{A_2}\widetilde{p}_{f}$

所以，

$^{A_2}\widetilde{p}_f = H^{-1}_{f_{new}}H_{f_{old}}^{A_1}\widetilde{p}_f + H^{-1}_{f_{new}}H_{old}\widetilde{x}_{old} - H^{-1}_{f_{new}}H_{new}\widetilde{x}_{new}$

anchor change 前后特征协方差相当于一次线性变换（与 IMU propagation 类似但无噪声项）。为减少 QR 次数，可将多个 SLAM 特征的 anchor change 合并处理。

![Anchor Change](doc/assets/09-anchor-change.png)

### 3.6 SLAM Feat Delay Init

满滑窗观测三角化后加入 SLAM 特征时，需给定其初始协方差。由

$$\begin{align*} r &= H_x \widetilde{x} + H_f \widetilde{f} + n \\ \begin{bmatrix} r_1 \\ r_2 \end{bmatrix} &= \begin{bmatrix} H_{x1} \\ H_{x2} \end{bmatrix}\widetilde{x} +   \begin{bmatrix} H_{f1} \\ 0 \end{bmatrix}\widetilde{f} +  \begin{bmatrix} n_1 \\ n_2 \end{bmatrix}  \\ \widetilde{f} &= H^{-1}_{f1}(r_1-n_1-H_x\widetilde{x}) \end{align*}, \quad \Rightarrow \mathbb{E}[\widetilde{f}] = H^{-1}_{f1}(r_1)$$

有

$$\begin{align*} P_{ff} &= \mathbb{E}[(\widetilde{f} -  \mathbb{E}[\widetilde{f}])(\widetilde{f} -  \mathbb{E}[\widetilde{f}])^T]  \\&= \mathbb{E}[(H_{f1}^{-1}(-n_1-H_{x1}\widetilde{x}))(H_{f1}^{-1}(-n_1-H_{x1}\widetilde{x}))^T]  \\ &= H^{-1}_{f1}(H_{x1}P_{xx}H^T_{x1} + R_1)H_{f1}^{-T} \end{align*}$$

$$\begin{align*} P_{xf} &= \mathbb{E}[(\widetilde{x})(\widetilde{f}-\mathbb{E}[\widetilde{f}])^T]  \\ &= \mathbb{E}[\widetilde{x}(H^{-1}_{f1}(-n_1-H_{x1}\widetilde{x}))^T]  \\ &= -P_{xx}H^T_{x1}H^{-T}_{f1} \end{align*}$$

$$P_{aug} = \begin{bmatrix} P_{xx} &P_{xf}  \\ P_{fx}&P_{ff} \end{bmatrix}$$

对应有

$$U_{aug} = \begin{bmatrix} U & -UH_{x1}^{-T}H_{f1}^{-T}  \\ 0 & R^{\frac{1}{2}} H_{f2}^{-T}\end{bmatrix}$$

![Delay Init](doc/assets/10-delay-init.png)

---

## 4、总结与讨论

- SquareRoot EKF 在数值稳定性与稀疏 QR 利用上仍有优化空间，可与工程实现结合进一步探索。

---

### 参考论文

1. **Wu, K., Ahmed, A., Georgiou, G., & Roumeliotis, S.** (2016). *A Square Root Inverse Filter for Efficient Vision-Aided Inertial Navigation on Mobile Devices.* Robotics: Science and Systems XI. [doi:10.15607/rss.2015.xi.008](https://doi.org/10.15607/rss.2015.xi.008)

2. **Peng, Y., Chen, C., & Huang, G.** (2024). *Ultrafast Square-Root Filter-based VINS.* IEEE ICRA, Yokohama, pp. 6966–6972. [doi:10.1109/ICRA57147.2024.10610916](https://doi.org/10.1109/ICRA57147.2024.10610916)
