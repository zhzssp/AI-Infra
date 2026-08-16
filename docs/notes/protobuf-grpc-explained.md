# Protobuf 与 gRPC 核心概念速记

## 为什么 ONNX 使用 Protobuf

**ONNX 的 `.onnx` 文件是 Protobuf 二进制序列化格式**。

Protobuf（Protocol Buffers）由谷歌设计，是通用的**数据序列化格式**。相比 JSON/XML：

| 优点 | 说明 |
|------|------|
| **紧凑高效** | 二进制格式，体积小（GB 级模型文件显著节省），编解码快 |
| **向前/向后兼容** | 新字段可无缝添加，旧版本解析器自动忽略新字段 |
| **强类型约束** | 字段类型明确，序列化时自动检查，适合 RPC 参数传递 |
| **跨语言支持** | C++/Python/Go/Java/Rust 等都有官方支持 |

**为什么AI模型用它**：
- 模型文件巨大（GB）→ 二进制高效
- 版本演进频繁（新 opset、新算子）→ 兼容性机制避免破裂
- 框架通用（PyTorch/TensorFlow/ONNX Runtime）→ 强类型+跨语言不可或缺

---

## Protobuf 与 gRPC 的关系（核心）

### 两个身份

**Protobuf**：通用数据序列化工具，**两种用法**：

1. **序列化模式**（如 ONNX）
   ```protobuf
   message Tensor {
     string name = 1;
     repeated int64 shape = 2;
     bytes data = 3;
   }
   ```
   用途：定义数据结构，序列化到文件/网络

2. **RPC 契约模式**（如 gRPC）
   ```protobuf
   service InferenceService {
     rpc Predict(InferenceRequest) returns (InferenceResponse);
   }
   ```
   用途：定义服务接口和通信协议

### 从属关系（不是平行关系）

```
Protobuf（独立存在）
   ↑
   │ gRPC 依赖它
   │
gRPC（RPC 框架）
```

| 层级 | 职责 |
|------|------|
| **Protobuf** | 定义消息结构 + 二进制序列化 |
| **gRPC** | 定义服务接口 + HTTP/2 通信框架 |

**关键点**：
- gRPC 的所有 RPC 输入/输出都**必须是 Protobuf message**
- gRPC 代码生成工具从 `.proto` 文件生成**服务接口代码**（调用 Protobuf 序列化库）
- Protobuf **不依赖** gRPC，可独立用于文件存储、消息队列等场景

---

## 工作流对比：两种用法

### 场景 A：Protobuf 单独使用（ONNX 模型存储）
```
.proto 定义 → protoc 编译 → 生成序列化代码
                              ↓
                        手工创建对象
                        obj.SerializeToString() → model.onnx
```

### 场景 B：gRPC + Protobuf（微服务通信）
```
.proto 定义服务 + 消息
        ↓
protoc --grpc_python_out 编译
        ↓
自动生成服务基类 + 序列化代码
        ↓
服务端实现接口 ←→ 客户端调用存根（stub）
        ↓
gRPC 框架自动处理序列化/反序列化/HTTP/2 通信
```

---

## 代码示例对比

### 只用 Protobuf（需手工网络处理）
```python
from inference_pb2 import Request, Response
import socket

req = Request(query="what is 2+2?")
data = req.SerializeToString()  # 手工序列化

# 需要自己管理 socket、重试、超时等
sock = socket.socket()
sock.send(data)
response_data = sock.recv(4096)
resp = Response.FromString(response_data)  # 手工反序列化
```

### gRPC + Protobuf（自动处理）
```python
# server.py
from inference_pb2_grpc import CalculatorServicer
from inference_pb2 import Request, Response

class CalculatorImpl(CalculatorServicer):
    def Compute(self, request: Request, context) -> Response:
        # gRPC 已自动反序列化 request
        return Response(result=str(eval(request.query)))

server = grpc.server(...)
add_CalculatorServicer_to_server(CalculatorImpl(), server)
server.add_insecure_port("[::]:50051")
server.start()

# client.py
stub = CalculatorStub(channel)
response = stub.Compute(Request(query="2+2"))  # 一行代码！
```

---

## gRPC 增强能力（Protobuf 做不到）

| 能力 | Protobuf | gRPC |
|------|----------|------|
| 消息序列化 | ✅ | ✅ |
| 网络通信框架 | ❌ | ✅ |
| HTTP/2 流式传输 | ❌ | ✅ |
| 四种通信模式（单向/服务端流/客户端流/双向） | ❌ | ✅ |
| 超时/重试/拦截器 | ❌ | ✅ |
| 连接池管理 | ❌ | ✅ |

四种通信模式定义示例：
```protobuf
service Service {
  rpc UnaryCall(Request) returns (Response);                    // 1. 一元
  rpc ServerStream(Request) returns (stream Response);          // 2. 服务端流
  rpc ClientStream(stream Request) returns (Response);          // 3. 客户端流
  rpc BiStream(stream Request) returns (stream Response);       // 4. 双向流
}
```

---

## 为什么这样设计

**Protobuf 的哲学**："我只关心数据怎么存、怎么序列化；网络传输由你自己处理"→ 复用性强

**gRPC 的哲学**："我需要高效 RPC 框架，消息结构用业界标准 Protobuf"→ 整合最佳实践

**结果**：
- Protobuf = 通用工具（ONNX、REST API、消息队列都能用）
- gRPC = 专为 RPC 优化的"套装"（集成 Protobuf）
- ONNX = 选择用 Protobuf 作存储格式，但不用 gRPC

---

## 回顾：企业 RPC 场景

你在企业中遇到的 gRPC + Protobuf 组合：
```
微服务 A                 微服务 B
  │                       │
  └──→ gRPC 调用 ────────→│
      (Protobuf 参数       (自动反序列化)
       自动序列化)
       
      HTTP/2 socket 通道
```

gRPC 的核心价值：
- 定义 `.proto` → 自动生成客户端/服务端代码
- 参数**自动序列化/反序列化**（程序员无感知）
- 版本演进兼容（新字段服务端自动忽略）
- 原生支持流式、双向通信

---

**一句话总结**：Protobuf 和 gRPC 是**从属关系**。gRPC 说"我需要 Protobuf 来序列化"；Protobuf 说"我独立存在，谁都能用"。类似 HTTP 和 JSON 的关系。
