# openvins-srf

> 基于 openvins 的 square root filter based VIO 实现，参考 ultra fast VINS。

---

## 📋 目录

- [依赖要求](#-依赖要求)
- [编译步骤](#-编译步骤)
- [运行方法](#-运行方法)
- [SRF 滤波器编译与测试](#-srf-滤波器编译与测试)
- [滤波器对比](#-滤波器对比)

---

## 📦 依赖要求

| 依赖项 | 版本要求 |
|--------|---------|
| ROS | Noetic 或其他版本 |
| CMake | >= 3.10 |
| C++ 编译器 | C++17 支持 |
| OpenCV | >= 4.0 |
| Eigen3 | - |
| Boost | - |
| Google glog | libgoogle-glog-dev |

---

## 🔨 编译步骤

### 1. 设置 ROS 环境

```bash
source /opt/ros/noetic/setup.bash
```

### 2. 创建构建目录并编译

#### 默认编译（使用 EKF 滤波器）

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

#### 使用 SRF 滤波器编译

```bash
mkdir -p build && cd build
cmake -DVIO_FILTER=SRF ..
make -j$(nproc)
```

#### 使用 SRF 滤波器并启用浮点数（推荐，性能更好）

```bash
mkdir -p build && cd build
cmake -DVIO_FILTER=SRF -DVIO_USE_FLOAT=ON ..
make -j$(nproc)
```

> 💡 **提示**: 编译完成后，可执行文件位于 `build/application/vio_sim/mono_vio`

#### 可用的滤波器选项

| 选项 | 说明 | 状态 |
|------|------|------|
| `EKF` | 扩展卡尔曼滤波器（默认） | ✅ 推荐 |
| `SRF` | 平方根滤波器（Square Root Filter） | ✅ 推荐 |
| `SRIF` | 平方根信息滤波器（Square Root Information Filter） | ⚠️ 存在精度问题 |

> ⚠️ **注意**: 当前 SRIF 实现存在精度问题，不推荐用于生产环境

#### 其他编译选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `VIO_USE_FLOAT` | 使用浮点数而非双精度 | `ON` (SRF 推荐开启，SRIF 不推荐使用) |
| `VIO_BUILD_TEST` | 编译单元测试 | `OFF` |
| `VIO_ENABLE_VISUALIZATION` | 启用可视化 | `OFF` |

---

## 🚀 运行方法

`mono_vio` 是样例程序，用于处理 ROS bag 文件中的 IMU 和相机数据，并输出轨迹估计结果。

### 基本用法

```bash
./build/application/vio_sim/mono_vio -c <配置文件> -d <bag文件路径> [选项]
```

### 参数说明

| 参数 | 长参数 | 说明 | 必需 |
|------|--------|------|------|
| `-c` | `--config` | 配置文件路径（YAML 格式） | ✅ |
| `-d` | `--dataset` | ROS bag 文件路径 | ✅ |
| `-t` | `--trajectory` | 保存轨迹文件的路径（TUM 格式） | ❌ |
| `-l` | `--log` | 保存日志文件的目录路径 | ❌ |

### 配置文件说明

配置文件需要包含以下内容：

#### 1. 主配置文件 (`estimator_config.yaml`)

VIO 系统的主要参数配置：

- 特征跟踪参数（KLT、特征点数量等）
- 滤波器参数（滑动窗口大小、SLAM 特征数量等）
- 初始化参数
- 相机和 IMU 的相对配置文件路径

#### 2. IMU 配置文件 (`kalibr_imu_chain.yaml`)

IMU 传感器参数：

- IMU 噪声参数
- ROS 话题名称
- 更新频率

#### 3. 相机-IMU 配置文件 (`kalibr_imucam_chain.yaml`)

相机标定参数：

- 相机内参（焦距、主点、畸变系数）
- 相机到 IMU 的外参（旋转和平移）
- ROS 话题名称
- 相机分辨率

> 📁 配置文件示例位于 `config/euroc_mav/` 目录下

### 运行示例

#### 基本运行（仅处理数据，不保存轨迹）

```bash
./build/application/vio_sim/mono_vio \
  -c ./config/euroc_mav/estimator_config.yaml \
  -d /path/to/your/dataset.bag
```

#### 保存轨迹文件

```bash
./build/application/vio_sim/mono_vio \
  -c ./config/euroc_mav/estimator_config.yaml \
  -d /path/to/your/dataset.bag \
  -t ./trajectory_output.txt
```

#### 保存轨迹和日志

```bash
./build/application/vio_sim/mono_vio \
  -c ./config/euroc_mav/estimator_config.yaml \
  -d /path/to/your/dataset.bag \
  -t ./trajectory_output.txt \
  -l ./log_output/
```

### 输出说明

#### 轨迹文件 (`-t` 参数指定)

TUM 格式的轨迹文件，包含时间戳、位置和四元数姿态：

```
timestamp tx ty tz qx qy qz qw
```

#### 日志文件 (`-l` 参数指定目录)

包含系统运行日志，包括：

- VIO 位置、速度、姿态
- IMU 偏置
- 真值位置（如果 bag 文件中包含 `/gt_pose` 话题）

### 注意事项

1. **ROS 环境**: 虽然 `mono_vio` 从 bag 文件读取数据，但需要 ROS 环境来解析 bag 文件格式

2. **配置文件路径**: 配置文件中的相对路径（如 `relative_config_imu` 和 `relative_config_imucam`）是相对于主配置文件所在目录的

3. **真值对齐**: 如果 bag 文件中包含 `/gt_pose` 话题，程序会自动对齐 VIO 估计和真值轨迹

4. **性能优化**: 
   - 可以通过配置文件调整特征点数量 (`num_pts`)
   - 调整跟踪频率 (`track_frequency`) 可以平衡精度和计算量
   - 使用多线程可以加速处理（在配置文件中设置 `num_opencv_threads`）

### 故障排查

| 问题 | 解决方案 |
|------|---------|
| **配置文件解析错误** | 检查 YAML 文件格式是否正确，路径是否正确 |
| **Bag 文件读取错误** | 确认 bag 文件路径正确，且包含所需的 IMU 和相机话题 |
| **话题名称不匹配** | 检查配置文件中的 `rostopic` 是否与 bag 文件中的话题名称一致 |

---

## 🔬 SRF 滤波器编译与测试

### SRF 滤波器简介

**SRF (Square Root Filter)** 是项目支持的三种滤波器之一，使用平方根形式的协方差矩阵，具有更好的数值稳定性和计算效率。

> 💡 当启用 `VIO_USE_FLOAT=ON` 时，SRF 使用单精度浮点数，可以进一步提升性能。

### 编译 SRF 版本

#### 基本编译

```bash
mkdir -p build && cd build
cmake -DVIO_FILTER=SRF ..
make -j$(nproc)
```

#### 启用浮点数优化（推荐）

```bash
mkdir -p build && cd build
cmake -DVIO_FILTER=SRF -DVIO_USE_FLOAT=ON ..
make -j$(nproc)
```

#### 验证编译配置

编译时会显示以下信息确认 SRF 已启用：

```
-- VIO: USE SRF enabled
-- VIO: USE FLOAT enabled  (如果启用了 VIO_USE_FLOAT)
```

### 运行 SRF 版本

SRF 版本的 `mono_vio` 使用方法与标准版本完全相同：

```bash
./build/application/vio_sim/mono_vio \
  -c ./config/euroc_mav/estimator_config.yaml \
  -d /path/to/your/dataset.bag \
  -t ./trajectory_output.txt
```

### SRF 单元测试

项目包含 SRF 滤波器的单元测试，用于验证滤波器实现的正确性。

#### 编译测试

```bash
mkdir -p build && cd build
cmake -DVIO_FILTER=SRF -DVIO_BUILD_TEST=ON ..
make -j$(nproc)
```

#### 运行 SRF 测试

```bash
./build/msckf/test_filters_srf
```

#### 测试内容

测试会验证以下功能：

- ✅ 状态更新 (Update)
- ✅ 协方差初始化 (Initialize)
- ✅ 状态传播 (Propagation)
- ✅ 边缘化 (Marginalize)
- ✅ 状态克隆 (Clone)
- ✅ 克隆增强 (AugmentClone)
- ✅ 传播和克隆组合 (PropagationAndClone)
- ✅ 边缘协方差提取 (GetMarginalCovariance)

#### 预期输出

所有测试通过时会显示类似以下信息：

```
[==========] Running X tests from 1 test case.
[----------] Global test environment set-up.
[----------] X tests from FilterSRFTest
[ RUN      ] FilterSRFTest.Update
[       OK ] FilterSRFTest.Update
...
[==========] X tests from 1 test case ran.
[  PASSED  ] X tests.
```

---

## 📊 滤波器对比

| 特性 | EKF | SRF | SRIF |
|------|-----|-----|------|
| **数值稳定性** | 标准 | 更好（平方根形式） | 较好（信息矩阵形式） |
| **计算效率** | 标准 | 更高（特别是启用浮点数时） | 中等 |
| **内存使用** | 标准 | 可能更少（浮点数模式） | 标准 |
| **精度** | 双精度 | 单精度（浮点数模式）或双精度 | ⚠️ **存在精度问题** |
| **生产就绪** | ✅ 是 | ✅ 是 | ❌ **否（存在已知问题）** |

### 建议使用场景

- **SRF + 浮点数**: 适用于对性能要求高、对精度要求适中的实时应用 ⭐ **推荐**
- **EKF**: 适用于对精度要求极高的应用场景 ⭐ **推荐**
- **SRIF**: ⚠️ **当前实现存在精度问题，仅用于研究和测试，不推荐用于生产环境**

---

## 🧹 清理构建

如果需要切换滤波器类型，建议清理构建目录：

```bash
rm -rf build
mkdir build && cd build
cmake -DVIO_FILTER=SRF ..  # 或其他滤波器选项
make -j$(nproc)
```

---
