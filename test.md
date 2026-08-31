# fast-pointcloud 优化实验记录

> 更新时间：2026-08-31  
> 平台：RK3588S / AArch64 / ROS2 Humble / C++17 / PCL / OpenMP / NEON  
> 目的：记录 `cloud_workspace_node` 与 `pcl_obstacle_node` 从 V0 到 V7 的代码演进、性能指标、失败方案和最终结论。  
> 注意：不同表格可能使用不同场景、stride 或运行隔离条件。只有同一张表内、且测试条件一致的数据才可直接计算加速比。

## 1. 测试对象与数据链路

主要性能链路：

```text
depth image
  -> 深度有效性检查
  -> stride采样
  -> 深度反投影
  -> TF坐标变换
  -> workspace ROI过滤
  -> VoxelGrid
  -> PointCloud2发布
  -> 地面分割
  -> 障碍物点云
  -> BEV
```

主要节点：

- `cloud_workspace_node`：反投影、TF、ROI、VoxelGrid和点云发布。
- `pcl_obstacle_node`：PointCloud2转换、近地候选、RANSAC、地面提取、可选聚类。
- `bev_node`：障碍物点云转局部占用栅格。

固定回归包：

```text
/home/cat/bags/perf_regression_heavy_2s
```

播放方式：

```bash
cd /home/cat/bags/perf_regression_heavy_2s
ros2 bag play . -l
```

不要使用 `--start-offset 0.5`：该参数会跳过包开头的静态TF，导致持续出现 `camera_color_optical_frame does not exist`，测试无法进入有效计算。

## 2. 指标定义

- `core_p95`：反投影、坐标变换和ROI过滤的p95耗时。
- `voxel_p95`：VoxelGrid阶段p95耗时。
- `compute_total_p95`：Core与Voxel完整计算链的逐帧总耗时p95。不能用两个独立p95直接相加替代。
- `queue_wait_p95`：深度帧进入latest-frame队列后等待计算线程的p95时间。
- `callback_p95`：帧到达节点后，包含排队、计算、转换和发布调用在内的节点级p95耗时。
- `dropped_pending`：latest-frame槽中尚未处理的帧被新帧覆盖的累计数量。
- `affinity_ok`：计算线程和OpenMP线程是否实际运行在目标CPU上。
- `samples=1000`：正式窗口包含1000个有效样本；少于1000的停止窗口只能作为辅助证据。

默认使用p95而不是平均值，因为机器人避障更关心尾延迟。

## 3. 版本代码概述

### V0：原始基线

- 原始多遍点云处理。
- 深度反投影、逐点TF、PassThrough等阶段相互独立。
- 存在多次完整点云遍历和中间点云写入。

### V1：射线缓存

- 按相机内参和stride预计算 `ray_x`、`ray_y`。
- 每帧不再重复计算 `(u-cx)/fx`、`(v-cy)/fy`。
- 其他多遍处理基本保持不变。

### V2：标量遍历融合

- 将深度检查、反投影、TF和ROI融合为一次标量遍历。
- 删除中间相机坐标点云和多次PassThrough遍历。
- Voxel仍调用原始 `pcl::VoxelGrid<PointXYZ>`。
- 作为后续OpenMP、NEON和Voxel重构的标量基线。

### V3：OpenMP反投影

- 按连续行区间切分深度图。
- 每个OpenMP线程拥有一个线程局部点云。
- 线程局部结果按线程ID计算最终offset，并并行复制到最终row-major点云。
- OpenMP线程使用 `sched_setaffinity()`逐线程手动绑定。
- 不再使用全局 `OMP_PLACES + OMP_PROC_BIND=CLOSE`污染整个ROS进程的线程亲和性。
- Voxel仍为串行PCL VoxelGrid。

### V4：单大核NEON

- 四个像素一组进行float32 NEON反投影、Transform和ROI判断。
- 使用持久计算线程，ROS回调只提交latest-frame任务。
- 计算线程可单独绑定大核，ROS/FastDDS线程保持CPU0-7调度能力。
- Voxel仍为串行PCL VoxelGrid。

### V5：PCL等价并行Voxel

曾测试两种实现。

#### V5早期哈希融合版（已否决）

- 反投影后直接计算体素key并插入线程局部哈希表。
- 合并线程局部哈希表后输出体素质心。
- 避免生成约16万点的完整ROI点云。
- 性能提高，但哈希桶/线程合并顺序改变了输出点序，导致后续随机RANSAC在相同bag上出现明显耗时变化。
- 不符合“尽量保持PCL输出一致”的目标，因此不作为最终实现。

#### V5稳定Radix版（保留）

- 保留与标量实现一致的row-major ROI点序。
- 使用与PCL相同的线性体素索引语义。
- 使用稳定OpenMP radix sort，体素key相同时保持原始点顺序。
- 按key建立连续group，OpenMP并行计算质心。
- 输出顺序保持PCL的线性体素索引顺序。
- radix从8 bit/4 pass调为11 bit/3 pass，减少一次全数组扫描。
- 尝试过NEON体素索引，实测约 `0.435 ms`，慢于标量索引约 `0.324 ms`，因此最终关闭该负优化。

### V6：stride=1专用NEON

- 基于单大核NEON路径继续优化stride=1。
- `uint16`深度直接使用 `vld1_u16 + vmovl_u16`。
- float深度直接使用 `vld1q_f32`。
- 不再构造四元素连续临时数组。
- 四点全部有效时，一次扩容并使用 `vst4q_f32`交错写入四个 `PointXYZ`。
- 混合有效组保持逐点压缩，点序不变。
- stride大于1时保留通用gather回退。
- Voxel仍使用原始PCL VoxelGrid，因此V6用于单独衡量NEON上限。

### V7：V6 NEON + OpenMP + PCL等价并行Voxel

- Core使用V6的stride=1连续NEON加载和批量写出。
- 四个大核CPU4-7按连续行区间并行处理。
- 接入V5稳定Radix版并行Voxel。
- 对靠近ROI边界和体素边界的点执行double精确复查，保证体素成员关系不变。
- 输出体素数量、体素key和体素顺序与标量反投影+原始PCL VoxelGrid一致。

## 4. 历史场景：V0～V3原始演进

以下为早期高教园场景，主要用于证明多遍标量融合和OpenMP的方向；不能与后面的固定bag直接比较。

| 版本 | Core p95 | Voxel p95 | Callback p95 | 说明 |
|---|---:|---:|---:|---|
| V0 | 9.385 ms | 5.695 ms | 15.200 ms | 原始多遍基线 |
| V1 | 8.877 ms | 5.515 ms | 14.286 ms | 射线缓存 |
| V2 | 4.272 ms | 6.408 ms | 10.859 ms | 标量遍历融合 |
| V3 1线程 | 4.937 ms | 5.755 ms | 10.876 ms | OpenMP单线程开销 |
| V3 4线程 | 1.566 ms | 6.260 ms | 7.917 ms | Core并行，Voxel未优化 |

早期结论：

- V0到V2的主要收益来自数据流重构，而不是SIMD。
- V3可以显著压缩Core，但串行Voxel逐渐成为主要瓶颈。

## 5. 固定bag早期线程数与大小核实验

这一轮仍处于全局OpenMP绑定和ROS线程亲和性问题排查阶段，Callback存在明显污染。

| 配置 | Core p95 | Voxel p95 | Callback p95 | 观察 |
|---|---:|---:|---:|---|
| V2 Scalar | 3.219 ms | 5.025 ms | 8.522 ms | 固定bag标量参考 |
| V3 1小核 | 4.839 ms | 5.170 ms | 10.238 ms | 小核单线程明显较慢 |
| V3 1大核 | 1.323 ms | 2.188 ms | 18.171 ms | Core快，但Callback被亲和性污染 |
| V3 2大核 | 0.884 ms | 2.143 ms | 17.802 ms | Core继续加速 |
| V3 4大核 | 0.636 ms | 2.221 ms | 17.565 ms | Core最好，Callback仍异常 |
| V3 6混合核，一轮 | 14.249 ms | 2.173 ms | 16.344 ms | 小核成为barrier拖尾 |
| V3 8全核 | 5.790 ms | 6.773 ms | 25.485 ms | 全核并非最优 |

另一轮6混合核曾得到：

```text
core_p95=1.504 ms voxel_p95=7.123 ms callback_p95=23.416 ms
```

手动亲和性版本的6核 `{4},{5},{6},{7},{2},{3}` 还得到：

```text
core_p95=14.358 ms voxel_p95=2.224 ms callback_p95=16.427 ms
```

结论：

- OpenMP并不是线程越多越快。
- 混入小核后，OpenMP隐式barrier必须等待最慢线程。
- 线程工作量按数量平均并不等于按性能平均。
- RK3588S该算子最终采用四个大核CPU4-7。

## 6. OpenMP亲和性问题与修复实验

### 6.1 失败方式：全局OMP绑定

曾使用：

```text
OMP_PLACES={0},{1},{2},{3},{4},{5},{6},{7}
OMP_PROC_BIND=CLOSE
```

OpenMP的tid0就是调用并行区的线程，不是额外创建的第九个线程，因此计算调用线程会被绑定到第一个place。ROS/FastDDS线程如果在绑定后创建，还会继承受限亲和性。

观测方式：

```bash
taskset -pc <pid>
ps -L -p <pid> -o pid,tid,psr,pcpu,stat,wchan,comm
for task_status in /proc/<pid>/task/*/status; do
  awk '/^Pid:|^Cpus_allowed_list:/' "$task_status"
done
```

### 6.2 最终方式：持久计算线程 + 每个OMP线程手动绑定

```text
ROS Executor / FastDDS线程：CPU0-7
持久计算线程 / OMP tid0：目标大核
OMP worker：其余目标大核
```

- 回调只把最新深度帧放入有界latest-frame槽。
- 持久计算线程取帧并执行Core、Voxel和发布。
- 每个OMP线程调用 `sched_setaffinity(0, ...)`，只改变自身。
- 使用 `thread_local`记住当前线程已绑定CPU，目标不变时不重复系统调用。
- 禁止全局 `OMP_PROC_BIND`影响ROS进程其他线程。

修复后的代表数据：

| 版本 | Queue p95 | Core p95 | Voxel p95 | Callback p95 | 丢帧 |
|---|---:|---:|---:|---:|---:|
| V2 Scalar | 0.141 ms | 0.853 ms | 2.012 ms | 3.148 ms | 0 |
| V3 4大核 | 0.132 ms | 0.662 ms | 2.239 ms | 3.339 ms | 0 |
| V4 NEON | 0.130 ms | 0.583 ms | 1.991 ms | 2.880 ms | 0 |

该组场景点较少，只用于验证线程架构和Callback恢复正常，不用于最终加速比。

## 7. 固定重负载bag，stride=2、voxel=0.05

### 7.1 V2标量与V5 PCL等价Voxel

| 版本 | Core p95 | Voxel p95 | Compute p95 | Callback p95 |
|---|---:|---:|---:|---:|
| V2 PCL基线 | 2.958 ms | 8.472 ms | — | 13.085 ms |
| V5 纯OpenMP | 2.528 ms | 2.835 ms | 5.563 ms | 6.282 ms |
| V5 OpenMP+NEON | 2.289 ms | 2.906 ms | 5.441 ms | 5.935 ms |

相对V2：

- Core：`2.958 -> 2.289 ms`，约1.29倍。
- Voxel：`8.472 -> 2.835 ms`，纯OpenMP约2.99倍。
- Callback：`13.085 -> 5.935 ms`，约2.20倍。

V5阶段明细，纯OpenMP四大核：

```text
minmax_p95=0.383 ms
index_p95=0.344 ms
sort_p95=1.018 ms
group_p95=0.279 ms
centroid_p95=0.904 ms
voxel_p95=2.835 ms
```

11-bit radix相对早期8-bit radix：

```text
sort约 1.17 -> 1.05 ms
voxel约 3.013 -> 2.906 ms
```

### 7.2 V5早期哈希融合实验（否决）

```text
accumulate_p95=3.061 ms
merge_p95=2.670 ms
emit_p95=0.222 ms
compute_total_p95=6.183 ms
callback_p95=7.162 ms
```

虽然比V2快，但哈希输出顺序与PCL不同。同样点集进入随机RANSAC后，RANSAC p95一度从约3 ms升到约16 ms。最终改为稳定radix实现，不采用哈希输出版。

## 8. 点云密度参数实验

### 8.1 stride=2、voxel=0.05

```text
cloud_workspace:
core_p95=2.958 ms
voxel_p95=8.472 ms
callback_p95=13.085 ms

pcl_obstacle:
input_pts_p95=8877
ransac_p95=3.463 ms
ground_total_p95=3.650 ms
callback_p95=4.036 ms
```

该配置在点云细节和30 FPS预算之间较均衡。

### 8.2 stride=2、voxel=0.03

```text
cloud_workspace:
core_p95=2.966 ms
voxel_p95=9.687 ms
callback_p95=15.653 ms

pcl_obstacle:
input_pts_p95=21505
ransac_p95=8.082 ms
ground_total_p95=8.288 ms
callback_p95=8.724 ms
```

相对0.05 m体素：

- 输出点数由约0.9万增加到约2.15万。
- Voxel和RANSAC耗时都明显增长。
- 可保留更多细障碍物细节，但后续地面分割也需要重构。

阶段性决定：默认先采用 `stride=2、voxel=0.05` 完成优化，再用 `voxel=0.03` 作为高精度压力场景。

### 8.3 stride=1、voxel=0.05

stride从2降到1后：

- 采样像素从约230400增加到约921600，约4倍。
- ROI有效点p95从约163283增加到约653925，约4倍。
- 原始PCL VoxelGrid由约8.5 ms上升到约29～32 ms。
- 更能放大OpenMP和并行Voxel的价值。

### 8.4 不同stride的Voxel前点云数

固定重负载bag、相同ROI和深度范围、`voxel_leaf_m=0.05` 下，VoxelGrid的输入是完成深度有效性过滤、反投影、Transform和ROI过滤后的 `cloud_roi`。V5/V7日志中的 `input_pts_p95` 就是Voxel前点云数，`output_voxels_p95` 是Voxel后点云数。

| depth_stride | 每帧采样像素上限 | Voxel前点云数p95 | Voxel后点云数p95 | 数据性质 |
|---:|---:|---:|---:|---|
| 1 | 921600 | 653925 | 9823 | V5/V7日志直接记录 |
| 2 | 230400 | 163283 | 8877～8917 | V5及早期融合Voxel日志直接记录，不同1000帧窗口略有波动 |
| 4 | 57600 | 未直接记录；按同场景比例约4.08万 | 7082～7105 | Voxel后点数来自早期 `pcl_obstacle_node input_pts_p95`；Voxel前点数仅作规模估计 |

说明：

- 图像分辨率为1280×720，因此stride为1、2、4时的理论采样上限分别为921600、230400、57600。
- Voxel前点数小于采样像素数，是因为无效深度、深度范围、Transform后ROI等条件会继续过滤像素。
- stride=1和stride=2的实测比例约为4.00倍，符合二维图像两个方向采样间隔同时减半所带来的4倍采样密度。
- stride=4实验发生在增加 `input_pts_p95` 日志之前，旧日志只能看到Voxel后的点数和耗时，无法还原严格的Voxel前p95。约4.08万来自同场景stride=1/2结果的比例估算，不能与前两项同等引用。
- `pcl_obstacle_node` 的 `input_pts_p95` 是 `/perception/cloud_workspace` 已完成Voxel后的输入点数。例如stride=2、voxel=0.05时约8877点，它不是Voxel前点数。
- stride从4降到1后，Voxel前点数约从4.08万增至65.39万，约16倍；Voxel后点数却只从约0.71万增至0.98万，约1.38倍。这说明固定5 cm叶尺寸下，新增的密集采样大多落入已有体素，主要增加了Voxel聚合计算量，而不是同比增加最终点云规模。

## 9. stride=1下V2～V5统一测试

每项主结果均为1000样本，亲和性检查通过。

| 版本 | CPU配置 | Queue p95 | Core p95 | Voxel p95 | Compute p95 | Callback p95 | dropped_pending |
|---|---|---:|---:|---:|---:|---:|---:|
| V2 Scalar | CPU7 | 32.298 ms | 10.709 ms | 29.270 ms | 39.872 ms | 70.599 ms | 138 |
| V3 OMP | 2大核CPU6-7 | 30.684 ms | 12.436 ms | 31.674 ms | 42.636 ms | 68.170 ms | 102 |
| V3 OMP | 4大核CPU4-7 | 16.936 ms | 8.385 ms | 29.240 ms | 36.534 ms | 53.180 ms | 46 |
| V4 NEON | CPU7 | 27.542 ms | 6.945 ms | 29.975 ms | 36.883 ms | 62.332 ms | 39 |
| V5 纯OMP | 4大核CPU4-7 | 0.105 ms | 5.846 ms | 7.317 ms | 13.270 ms | 18.122 ms | 2 |
| V5 OMP+NEON | 4大核CPU4-7 | 0.346 ms | 4.948 ms | 7.571 ms | 12.858 ms | 15.782 ms | 3 |

结论：

- V3两大核比V2慢，说明旧V3的并行区、线程局部点云和结果合并开销超过两核收益。
- V3四大核可以压缩Core，但无法加速原始PCL VoxelGrid。
- V4将Core降到6.945 ms，但仍被约30 ms的Voxel限制。
- V5并行Voxel是stride=1下的主要收益来源。
- V5 OMP+NEON相对V2的Compute约3.10倍，Callback约4.47倍；Callback加速包含消除队列积压的非线性收益。

V5 OMP+NEON曾出现一个异常窗口：

```text
core_p95=7.914 ms
voxel_p95=12.594 ms
compute_total_p95=20.734 ms
callback_p95=22.449 ms
```

同一进程下一窗口906样本恢复为：

```text
core_p95=4.799 ms
voxel_p95=7.440 ms
compute_total_p95=12.349 ms
callback_p95=15.396 ms
```

重新启动后的完整1000样本为表中结果，因此异常窗口不作为主结果，但保留为稳定性记录。

## 10. V6、V7最终开发与测试

### 10.1 完整感知链首个干净窗口

| 版本 | 配置 | Core p95 | Voxel p95 | Compute p95 | Callback p95 |
|---|---|---:|---:|---:|---:|
| V6 | 单大核NEON，原始PCL Voxel | 6.901 ms | 29.769 ms | 36.525 ms | 61.464 ms |
| V7 | 四大核NEON，并行PCL等价Voxel | 4.661 ms | 7.602 ms | 13.436 ms | 14.855 ms |

V6到V7：

- Core约1.48倍。
- Voxel约3.92倍。
- Compute约2.72倍。
- Callback约4.14倍；包含队列积压消失带来的收益。

V6相对旧V4的Core只从6.945 ms降到6.901 ms，约0.6%。说明Release编译器已基本消除旧四元素临时数组，stride=1显式连续加载不是主要瓶颈。单核NEON最终受Transform计算、ROI压缩写出和内存流量限制。

### 10.2 算子隔离测试

完整链测试发现 `pcl_obstacle_node` 的RANSAC会与CPU4-7上的Core/Voxel争用共享缓存和内存带宽。为得到算子本身的稳定值，临时执行：

```text
bag、pcl_obstacle_node、bev_node -> CPU0-3
V6计算线程 -> CPU7
V7 OpenMP -> CPU4-7
```

最终又停止下游PCL和BEV，仅保留TF与 `cloud_workspace_node` 收集隔离窗口。

| 版本 | Core p95 | Voxel p95 | Compute p95 | Callback p95 |
|---|---:|---:|---:|---:|
| V6 NEON | 6.861 ms | 29.203 ms | 35.951 ms | 60.406 ms |
| V7 NEON+OMP | 4.164 ms | 7.545 ms | 11.688 ms | 13.606 ms |

V6到V7隔离加速：

- Core：约1.65倍。
- Voxel：约3.87倍。
- Compute：约3.08倍。
- Callback：约4.44倍。

V6连续完整窗口：

```text
窗口1 core=6.868 voxel=29.252 compute=36.001 ms
窗口2 core=6.862 voxel=29.199 compute=35.947 ms
窗口3 core=6.861 voxel=29.203 compute=35.951 ms
```

V7下游退出后的完整窗口：

```text
core=4.164 ms
minmax=0.590 ms
index=0.681 ms
sort=3.379 ms
group=0.729 ms
centroid=2.085 ms
voxel=7.545 ms
compute_total=11.688 ms
```

说明：算子隔离结果用于评估代码能力；完整链路结果用于评估实际系统表现。简历或报告中必须标明采用哪一种测试条件。

## 11. V7输出一致性验证

验证参考：

```text
标量double反投影 + 原始 pcl::VoxelGrid
        对比
V7 NEON/OpenMP反投影 + 稳定Radix并行Voxel
```

验证项：

- 输出体素数量完全一致。
- 每个位置的体素key完全一致。
- 体素输出顺序完全一致。
- ROI/体素边界点使用double精确复查。
- 最大观察质心坐标误差约 `0.0267 mm`。
- 验证门限为 `0.05 mm`。

stride=1时单个体素可能包含stride=2约四倍的点。NEON float变换与标量double参考会产生少量末位误差，密集体素质心会累积该误差；0.05 mm门限仍远低于RGB-D相机毫米级深度噪声。验证门限不能替代体素拓扑检查，数量、key和顺序仍要求严格一致。

验证模式只用于正确性，不得用于性能统计：

```bash
projection_fused_voxel_verify:=true
```

## 12. pcl_obstacle_node分阶段Profiling

### 12.1 开启欧式聚类

关闭RViz后的代表1000样本：

| 阶段 | p95 |
|---|---:|
| PointCloud2 -> PCL | 0.110 ms |
| 近地候选扫描 | 0.008 ms |
| RANSAC | 2.859 ms |
| 地面提取 | 0.107 ms |
| KdTree | 1.391 ms |
| Euclidean Cluster提取 | 28.748 ms |
| Cluster总计 | 30.190 ms |
| PCL -> PointCloud2 | 0.045 ms |
| publish调用 | 0.111 ms |

工作量：

```text
input_pts_p95=7105
ground_pts_p95=1245
obstacle_pts_p95=6926
published_clusters_p95=5
```

结论：欧式聚类是该节点最大瓶颈，但BEV/Nav2几何避障主链不要求物体实例聚类，因此增加参数将聚类默认关闭。

### 12.2 关闭聚类

stride=2、voxel=0.05时：

```text
pcl callback_p95=4.036 ms
ransac_p95=3.463 ms
kdtree_n=0
cluster_extract_n=0
```

关闭聚类后，PCL节点回到个位数毫秒，地面分割成为主要阶段。

## 13. RViz、发布和rosbag稳定性实验

### 13.1 RViz导致的假性发布耗时

曾观察到：

```text
cloud publish p95约211 ms
callback p95约241 ms
```

关闭RViz后：

```text
cloud_to_ros_p95=0.045 ms
cloud_publish_call_p95=0.111 ms
```

结论：211 ms不是点云序列化算法本身，而是可视化订阅、DDS队列和系统背压共同造成。正式Profiling应关闭RViz，或将可视化与算法测试分开。

### 13.2 rosbag循环播放频率不稳定

曾观测深度输入约26～27 Hz，点云输出在不同窗口从个位数恢复到约26～30 Hz，并出现0.3～1.5秒甚至更长的间隔。

原因包括：

- 2秒短包循环边界。
- rosbag重新定位和TF/static消息时序。
- `ros2 topic hz`自身订阅与统计窗口。
- 下游PCL、RViz和DDS竞争。

因此：

- 算子耗时以节点内部steady clock分阶段p95为主。
- `ros2 topic hz`用于检查系统现象，不作为唯一性能结论。
- 实机相机30 FPS需要单独验证，不能直接由短bag循环频率推断。

## 14. 启动命令

### V2 Scalar

```bash
ros2 launch rgbd_bringup perception.launch.py \
  use_orbbec:=false \
  enable_yolo:=false \
  projection_enable_openmp:=false \
  projection_openmp_threads:=1 \
  projection_enable_neon:=false \
  projection_fuse_voxel:=false \
  projection_fused_voxel_openmp:=false
```

### V3 四大核

```bash
ros2 launch rgbd_bringup perception.launch.py \
  use_orbbec:=false \
  enable_yolo:=false \
  projection_enable_openmp:=true \
  projection_openmp_threads:=4 \
  projection_enable_neon:=false \
  projection_omp_places:='{4},{5},{6},{7}' \
  projection_fuse_voxel:=false
```

### V6 单大核NEON

```bash
ros2 launch rgbd_bringup perception.launch.py \
  use_orbbec:=false \
  enable_yolo:=false \
  projection_enable_openmp:=false \
  projection_openmp_threads:=1 \
  projection_enable_neon:=true \
  projection_compute_cpu:=7 \
  projection_fuse_voxel:=false \
  projection_fused_voxel_openmp:=false
```

### V7 四大核NEON+OpenMP

```bash
ros2 launch rgbd_bringup perception.launch.py \
  use_orbbec:=false \
  enable_yolo:=false \
  projection_enable_openmp:=false \
  projection_openmp_threads:=4 \
  projection_enable_neon:=true \
  projection_omp_places:='{4},{5},{6},{7}' \
  projection_fuse_voxel:=true \
  projection_fused_voxel_openmp:=true \
  projection_fused_voxel_verify:=false
```

V7的 `projection_enable_openmp=false` 是正确的：它关闭旧V3反投影路径；V7的OpenMP由 `projection_fused_voxel_openmp=true`控制。

### V7正确性验证

```bash
ros2 launch rgbd_bringup perception.launch.py \
  use_orbbec:=false \
  enable_yolo:=false \
  projection_enable_openmp:=false \
  projection_openmp_threads:=4 \
  projection_enable_neon:=true \
  projection_omp_places:='{4},{5},{6},{7}' \
  projection_fuse_voxel:=true \
  projection_fused_voxel_openmp:=true \
  projection_fused_voxel_verify:=true
```

## 15. 编译与检查

```bash
cd /home/cat/fast-pointcloud
colcon build \
  --packages-select rgbd_pcl rgbd_bringup \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

最终状态：

- Release编译通过。
- `cloud_workspace_node.cpp`差异检查通过。
- V7亲和性检查通过，OpenMP线程实际位于CPU4、5、6、7。
- 测试结束后感知节点已停止。
- 固定bag继续循环播放，bag进程亲和性已恢复CPU0-7。

## 16. 最终结论

1. 最大的第一阶段收益来自V0到V2的数据流重构：融合遍历、射线缓存和减少中间点云。
2. 单核NEON在stride=1下将Core压到约6.86 ms，但显式连续加载相对旧V4只有约1%收益，说明主要瓶颈已转向压缩写出和内存流量。
3. 四大核OpenMP对大点数场景有效，但两核、混合大小核和全核都不一定更快；RK3588S最终选择四个大核。
4. 原始PCL VoxelGrid在stride=1下约29 ms，是完整链最大瓶颈。
5. 稳定radix并行Voxel将Voxel降到约7.55 ms，同时保持PCL体素拓扑与输出顺序。
6. V7隔离Compute p95为11.688 ms，相对V6的35.951 ms约3.08倍。
7. 完整链路还会受到RANSAC、DDS、RViz和共享内存带宽影响，因此必须同时报告算子隔离结果和系统结果。
8. 最终可用于简历的技术点不是简单“使用OpenMP/NEON”，而是：固定bag回归、分阶段Profiling、大小核实验、线程亲和性隔离、稳定并行Radix Voxel、NEON连续加载、输出一致性验证和尾延迟治理。
