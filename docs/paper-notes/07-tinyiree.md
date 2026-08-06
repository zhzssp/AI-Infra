# TinyIREE：把 IREE 缩到嵌入式规模的一组部署选项（简版笔记）

> **导航**：[笔记索引](README.md) · [自学枢纽](../README.md)（阶段 3） · [横切概念](../ai-compiler-foundations.md) §8（设备抽象机）  
> **配套阅读**：本篇**只建立第一印象**，刻意不按七节骨架写；正式材料是 [`iree-learning-guide.md`](../iree-learning-guide.md)，最小必要集用它的 §7。

> 论文元信息
> - 标题：*TinyIREE: An ML Execution Environment for Embedded Systems from Compilation to Deployment*
> - 作者/机构：Hsin-I Cindy Liu、Mahesh Ravishankar、Nicolas Vasilache、Ben Vanik、Stella Laurenzo（Google）+ Marius Brehler（Fraunhofer IML）
> - 出处：IEEE Micro，2022 年 7/8 月刊，共 9 页
> - arXiv：https://arxiv.org/abs/2205.14479
> - 开源：https://github.com/iree-org/iree

> **这篇笔记为什么短**
> 这不是老师指定的必读论文，是用来建立"IREE 大概长什么样"的第一印象的。
> 论文只有 9 页、发表于 2022 年，对 IREE 的核心机制（尤其是 HAL）讲得很浅，
> 且很多细节已经过时（例如当时的 `--iree-llvm-target-*` 系列 flag 现已更名）。
> **真正的学习材料见 [`docs/iree-learning-guide.md`](../iree-learning-guide.md)**，
> 那份文档基于 IREE 官方文档与源码，重点讲 HAL。这里只留一个骨架印象。

---

## 1. 一句话总结

IREE 是一套基于 MLIR 的端到端 ML 编译器 + 运行时；**TinyIREE 不是一个独立项目，而是 IREE 里一组"面向裸机/嵌入式"的部署选项的统称**——同一条编译流水线不变，只是换掉运行时的组装方式（换 HAL driver、换 VM 承载形式、换 executable 打包形式），把产物压到几十 KB 量级。

## 2. 它解决什么问题

论文的出发点是一个失衡：产业投入大量集中在"向上扩展"（数据中心、大加速卡），而低功耗与嵌入式设备缺乏好的支持。当时嵌入式侧的两条路线各有短板：

- **TFLM（TensorFlow Lite for Microcontrollers）**：运行时库很小（Cortex-M3 上约 16 kB），但它是 **kernel-based、逐算子解释执行**的——只支持 TensorFlow 算子的一个子集，每个算子都要针对硬件手写 kernel，图优化仍停留在算子层。换句话说，它靠"砍功能"换小体积。
- **Glow / TVM 这类编译器路线**：确实是编译式的，但 Glow 只有两级 IR、TVM 需要额外的 MicroTVM 扩展来补上裸机环境下缺失的调度与内存管理。

IREE 的主张是：**把 ML 模型当作一个普通程序来编译**，用 MLIR 的多级 dialect 渐进式降低，一路生成目标架构的二进制。这样"向下缩到微控制器"和"向上扩到数据中心"用的是**同一套编译流水线**，只是末端的部署选项不同——泛化能力是经典的两阶段发射式编译器（如 Glow）难以做到的，更是 TFLM 这类执行器完全做不到的。

## 3. 编译流水线（论文视角，很粗）

论文只描述了"设备侧代码生成"这一条线，没有展开 IREE 真正的核心分层（flow / stream / hal）：

```
TOSA / MHLO          张量算子层（论文当时的两个输入 dialect）
      │              例：tosa.matmul + tosa.add
      ▼
Linalg               完美嵌套循环 + 计算负载（iterator_types + indexing_maps）
      │              在这一层做融合、tiling、循环交换
      │              ── tiling 之后，每个 tile 被封装进一个 dispatch region
      ▼
（分叉）  host 侧：VM commands —— 决定 dispatch region 的执行顺序
          device 侧：dispatch region 内部继续优化
      ▼
Vector               可重定向的向量指令（如 Arm sdot）
      ▼
LLVM dialect ──▶ LLVM IR ──▶ 目标架构二进制
```

论文对 Linalg 这一层讲得最清楚，值得记住两点：

1. **`linalg.generic` 用 `iterator_types`（parallel / reduction）+ `indexing_maps`（仿射映射）来描述计算**，不描述"算的是什么"。于是**生产者-消费者关系的两个算子可以只凭这两组元信息就融合**，完全不需要理解算子语义。若在 TOSA/MHLO 那一层做同样的融合，就会陷入"算子两两组合"的排列爆炸。这是 Linalg 存在的核心理由。
2. **tiling 同样只靠这两组元信息**。切出来的 tile 被封装成 **dispatch region**——即"必须在设备上原子执行的一段代码"，这是后面 flow/stream/hal 三层的最小调度单位。

## 4. TinyIREE 的四个部署开关

这是论文真正的贡献，也是唯一需要记住的部分。**编译流水线完全不变，变的只是产物怎么组装**：

| 开关 | 两种选择 | 差别 |
|------|---------|------|
| **VM 控制流的承载形式** | ① 序列化成 **bytecode**（FlatBuffer，`.vmfb`），运行时用解释器执行<br>② 经 **EmitC** dialect 翻译成 **C 源码**，直接调用 VM API | 选 ②，字节码解释器就不用链进产物——论文实测在 Armv7E-M 上省掉约 **15 kB** |
| **ML workload 的库形式** | ① **静态库**（`.h`/`.o` 直接链进应用）<br>② **动态库**（IREE 自带固定 ABI 的轻量加载器，不依赖 OS 的 dlopen） | 静态库能参与链接期优化，体积最小；动态库可以为不同架构编多份、**运行时再决定用哪一份** |
| **HAL driver 的调度器** | ① **异步任务调度器**（多线程，按 DAG 乱序 + 流水执行）<br>② **同步调度器**（顺序发射） | 裸机无线程支持时用同步版。工作负载按 GPU 风格的 **3D grid** 派发（`worker.cnt.{x,y,z}` 三重循环调 `dispatch_ptr`） |
| **要不要 IREE runtime** | 甚至可以**完全绕过** IREE runtime，直接部署一个 workload | 极限场景 |

另外两个与内存有关的机制值得记一笔：

- **Stream execution**：异步执行期间所需的内存从一个"感知流式行为"的池子里预留，**只为当前可并发执行的那部分工作分配内存**。因为一次推理里绝大部分内存要么是常量、要么是只在单次调用内存活的临时量，这让一个已加载程序的**静止内存占用降到几 KB**（常量另算，且常量常可映射为可丢弃内存）。于是多个程序可以同时加载、交错执行，而峰值内存只等于其中最大的那一次调用。
- **Buffer 可见性与权限控制**：host 侧和 device 侧分别有各自的 allocator；一块内存可以在一端分配、而**选择性地**对另一端开放某种访问权限（如输入缓冲在 host 分配并写入初值，对 device 只开放读权限）。论文提到这与 enclave computing 结合可提供更安全的执行环境。

## 5. 实验结论（规模很小，注意局限）

模型只有一个：**MobileNet V2 SSD**，IREE 快照版本 20211203.686。

- **产物体积**：Armv7E-M 目标下，workload 从 Debug-dylib 的 136.55 kB 依次降到 Dylib 102.98 kB、Embedded 88.84 kB。后两者即 TinyIREE 模式。嵌入式目标的 workload 明显小于 x86-64（203.80 kB），主要来自 LLVM 针对该目标的优化。
- **运行时对比 TFLite**（x86-64）：峰值内存 **20.05 MB → 5.93 MB**，host 库 **2971 kB → 96 kB**。
- **诚实的注脚**：论文自己指出 TFLM 也能做到相近的库体积（约 80 kB / 4.91 MB 峰值内存），但 TFLM 只支持算子子集——它后来去掉了无符号算子支持，恰好就跑不了这个模型了。所以这组对比真正说明的是"**编译式路线在拿到相近体积的同时保住了通用性**"，而不是"IREE 内存管理碾压 TFLite"。
- 没有任何加速器（GPU/ASIC/DSP）上的数据，也没有延迟/吞吐数据。

## 6. 读完这篇应该带走的三件事

1. **IREE 的定位**：不是 kernel 库、不是解释器，而是"把模型当程序编译"，**调度逻辑与执行逻辑一起编译**成产物。同一条流水线覆盖从 MCU 到数据中心。
2. **Dispatch region 是核心调度单位**：Linalg 层 tiling 之后封装出来的"必须在设备上原子执行的一段代码"。后续 flow/stream/hal 三层都是围绕它做文章。
3. **"缩小"靠的是替换运行时的组装方式，而不是砍编译能力**：EmitC 替掉字节码解释器、静态库替掉动态库、同步调度器替掉任务调度器、精简的 HAL driver 替掉完整 HAL。**这四个开关全都落在 HAL 与 VM 这两层上**——这也正说明为什么接下来该重点读 HAL。

## 7. 这篇论文没讲、但必须另外补的

论文对下面这些只字未提或一带而过，而它们才是 IREE 的骨架（**全部在 [`docs/iree-learning-guide.md`](../iree-learning-guide.md) 里展开**）：

- **flow / stream / hal 三层 dialect 的分工**——论文只说"dispatch region + VM commands"，完全跳过了 Stream 这一层的异步调度与资源生命周期建模。
- **HAL 的完整对象模型**——论文只出现了 "HAL driver = workload loader + scheduler" 这一句话式的描述。真实的 HAL 有 driver / device / allocator / buffer / buffer_view / command_buffer / executable / executable_cache / semaphore / fence / event / channel 等十余类对象。
- **timeline semaphore 同步模型**——论文完全没提，而这是 IREE 异步执行模型的基石。
- **executable 的多目标 variant 机制**——即"一个 executable 里放多份不同目标的实现，运行时按条件选"，这是"低成本切换后端"的核心机制，论文里只在讲动态库时侧面提了一句。
- **集合通信（`iree_hal_channel_t`）与多设备拓扑**——2022 年时还没有，现在是 HAL 的一等公民，对算力网分布式场景直接相关。
