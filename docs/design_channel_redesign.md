# Channel 模块重构设计

> 状态:设计定稿,待实现
> 背景:对 PR #1997 `feat(channel)` 的架构评审与重新设计
> 部署前提:**面向分布式**(N 个对等节点 + Redis 承载 Storage 与 MessageBus),任何设计不得依赖单节点假设。

---

## 1. 目标与设计原则

### 1.1 模块定位

让 AgentScope 中配置好的 agent **以机器人身份入驻 IM 平台**(首个实现:飞书)。模块是 "IM 平台 ↔ agent 服务" 的双向翻译层:平台差异收敛在 adapter,编排收敛在 gateway,agent 服务完全不感知 IM 的存在——**channel 只是 agent 的另一种前端**,消费与 Web 端同一条内部事件流。

### 1.2 设计原则

1. **锚定 IM 的自然单位**,不发明系统自己的单位。会话以 `chat`(聊天/群)为中心,而非虚构的用户聚合体。
2. **删除可疑抽象**,而不是为其寻找更好的表达。本次重构删掉了两个 PR 中的核心抽象(SessionMapper、跨 chat 的 per-user 语义),因为它们的**语义本身**站不住,而非机制不好。
3. **真相唯一在存储**。运行时实例是存储记录的投影;广播/通知只是加速器,不承载正确性;丢失可由对账自愈。
4. **每次处理 = 一个短命协程**。不常驻订阅、不进程内跨消息状态、不手工编排多任务。
5. **最大化复用主分支通路**。输入、执行、输出、插话四条链路全部走已有基础设施。
6. **安全默认**。不支持确认 UI 的平台 → 自动拒绝;凭据加密存储、出口脱敏。

### 1.3 两条评审教训(记录以备后续设计参照)

- **SessionMapper 的删除**:PR 用一张 `(channel_id, key) → session_id` 的持久映射表记录分配结果,由此衍生分布式锁、竞态删除、O(N) 孤儿扫描。根因是 session_id 随机生成、必须"记住"。改为**确定性派生**(uuid5)后,映射变成纯函数,整个抽象连同其全部并发问题一起消失。
- **per-user scope 的删除**:PR 默认让同一用户的私聊 + 各群共享一份记忆。此语义有隐私缺陷(私聊上下文可能出现在群可见的回复里),且与主流 IM bot 预期不符。删除后 scope 从 4 值降为 2 值。

---

## 2. 架构总览

### 2.1 组件与职责

| 组件 | 层 | 职责 | 状态 |
|---|---|---|---|
| `ChannelBase` + 平台实现 | 适配 | 维持长连接、平台报文 ↔ 内部结构体、把回复/确认 UI 发回平台 | 平台连接(节点内) |
| `ChannelGateway` | 编排(数据面) | `process(event, channel)`:解析路由/会话 → 入 inbox + wake → 收集回复 → 呈现 | **无状态** |
| `ChannelService` | 服务 | channel 的 CRUD、校验、bot 查重、写记录、发 lifecycle 通知 | 无状态 |
| `ChannelRunRegistry` | 运行时 | 进程内实例表 `{channel_id → (adapter, listener_task, version)}` | 哑容器(节点内) |
| `ChannelLifecycleDispatcher` | 运行时 | 对账 storage↔实例;启动加载;定时兜底;status 心跳 | 常驻任务(每节点一个) |
| `ChannelTypeRegistry` | 注册 | 类型元数据 + 凭据 JSON Schema + 工厂 | 无状态 |

依赖方向严格单向,无环:

```
Router → ChannelService → Storage/Bus
ChannelLifecycleDispatcher → Storage → 造实例 → ChannelRunRegistry
ChannelRunRegistry ──(adapter.bind)──▶ adapter ──process(event, self)──▶ ChannelGateway
ChannelGateway ──(仅调用期借用)──▶ adapter.{send_response, present_confirm, update_confirm}
```

- `ChannelGateway` **不持有任何 channel**(无注册表),只处理"送上门的 event + 随行的 channel 引用"。
- `ChannelRunRegistry` 是唯一持有实例的地方;`ChannelLifecycleDispatcher` 是唯一增删实例的角色(参照 `ChatRunRegistry` / `CancelDispatcher` 的分工:一个存、一个动)。

### 2.2 与主分支基础设施的复用

channel 不新建任何执行/事件通路,四条链路全部复用:

| 链路 | 复用的主分支设施 |
|---|---|
| 输入 | `inbox_push(HintBlock)` + `enqueue_run_trigger(kind=wake)` |
| 执行 | `WakeupDispatcher`(唯一 spawn 点,结构性避免并发 spawn 竞态) |
| 输出 | 订阅 `MessageBusKeys.session_events(session_id)` |
| 插话 | `InboxMiddleware`(每个 reasoning step 前 drain inbox 注入 HintBlock) |

channel 特有的只剩三样:**平台翻译、收集权租约、呈现**。

---

## 3. 会话路由:一条规则,两个输出

### 3.1 核心问题

拿着入站消息的 `(channel_id, chat_id, user_id, metadata)`,决定 `(agent_id, session_id)`。

约束:`chat_id` / `user_id` 是**运行时才出现、无界**的——任何用户第一次私聊都会带来一个从未见过的键。因此路由不能是枚举映射表,必须是"匹配 + 兜底"的规则,且对未见键有确定行为。

### 3.2 路由模型

路由是**唯一一个概念**:有序的规则列表。每条规则 = 匹配条件 → 两个输出(交给谁 + 会话怎么归组)。

```python
class SessionScope(str, Enum):
    PER_CHAT = "per_chat"            # 默认:一个聊天一个 session
                                     #   私聊天然如此;群聊 = 全群共享上下文
    PER_CHAT_USER = "per_chat_user"  # 仅群聊有意义:群内按人隔离

class ChannelBinding(BaseModel):
    """一条路由规则:匹配到的消息 → 交给哪个 agent + 按什么粒度归组会话。"""
    match_key: str = "chat_id"       # 匹配 event 的哪个字段(chat_id/user_id/metadata 键)
    match_value: str = "*"           # 精确值,或 "*" 通配(兜底)
    agent_id: str
    session_scope: SessionScope = SessionScope.PER_CHAT

class RoutingConfig(BaseModel):
    bindings: list[ChannelBinding]   # 顺序即优先级;首条命中即停
    # 校验:必有且仅有最后一条为 "*";无重复的 (match_key, match_value)
```

- **兜底规则**(`match_value="*"`,必有)承载"默认交给谁、默认怎么归组"——前端简单模式只渲染这一条。
- **例外规则**(0 条起步)覆盖特定群/用户。
- 无 `default_agent_id` / `default_dm_scope` 顶层字段——兜底规则即默认。

### 3.3 解析:纯函数,零存储

```python
_NS = uuid.UUID("...")  # 固定命名空间

def resolve(event, record) -> tuple[str, str]:
    b = first_match(record.routing.bindings, event)     # 顺序扫描,首条命中
    agent_id = b.agent_id
    if b.session_scope is SessionScope.PER_CHAT:
        scope_key = event.chat_id
    else:  # PER_CHAT_USER
        scope_key = f"{event.chat_id}:{event.channel_user_id}"
    session_id = str(uuid5(_NS, f"{record.id}:{agent_id}:{scope_key}"))
    return agent_id, session_id
```

派生键含 `agent_id`,因此**不同 agent 永不共享 session**;scope 的共享/隔离都发生在"已路由到同一 agent"的前提下。

### 3.4 为什么不落表

规则 + 落表 = 规则 + 一层不必要的物化缓存,而缓存在分布式下有真实代价:

- **首次插行竞态**:同一用户在两个群同时发言,两条消息落到两个节点,都发现表里没有该键 → 都新建 session → 需要分布式锁 + 竞态删除。这正是被删掉的 SessionMapper。uuid5 现算两节点零协调得同一 id,`get_or_create` 幂等,竞态在数学上不存在。
- **规则变更语义**:scope 改动后存量行如何处理?现算方案无此问题——下一条消息自然按新规则归组,行为向前、确定。
- 该表是系统按规则自动生成的**派生状态**,不是配置;派生便宜到一次哈希,缓存无收益。

### 3.5 overlap

规则允许交叠,由**顺序消解**(first-match-wins),同防火墙/nginx location。目标不是消灭交叠(不同维度的匹配天然交叠,且"谁优先"是业务决策),而是让结果**确定**。保存时静态校验三种病态:完全重复、`*` 不在末尾、缺 `*`。UI 提供"规则测试器":输入一条消息特征,即时显示命中哪条、路由到哪。

---

## 4. 接口定义

### 4.1 ChannelBase(平台适配)

```python
class ChannelBase(ABC):
    capabilities: ChannelCapability

    @property
    @abstractmethod
    def channel_id(self) -> str: ...

    @abstractmethod
    async def start_listening(self) -> None:
        """建立长连接,循环接收;规范化为 event 后 await self._emit(event)。含自动重连。"""

    @abstractmethod
    async def send_response(
        self, event: ChannelEvent, content: list[TextBlock | DataBlock],
    ) -> None:
        """把回复发回平台。content 与入站对称,支持多模态;
        平台按 capability.image/file 决定发送或降级为占位。超长按
        capability.max_message_length 分段。"""

    async def present_confirm(
        self, event: ChannelEvent, prompt: ConfirmPrompt,
    ) -> str | None:
        """呈现确认请求(飞书=交互卡片,纯文本平台=一句提示)。
        返回呈现句柄(如卡片 message_id)供后续更新;None=此平台无法呈现
        → gateway 立即按拒绝处理。默认实现返回 None。"""
        return None

    async def update_confirm(
        self, ref: str, outcome: Literal["approved", "denied"],
    ) -> None:
        """更新呈现为最终态(卡片定格 / 文本回执 / no-op)。默认 no-op。"""

    # 轻交互(可选,默认 no-op)
    async def add_reaction(self, event, emoji_type) -> str | None: ...
    async def remove_reaction(self, event, reaction_id) -> None: ...

    # 生命周期
    async def on_start(self) -> None: ...
    async def on_stop(self) -> None: ...
    def bind(self, emit: Callable[[ChannelEvent], Awaitable[None]]) -> None:
        self._emit = emit   # 装配时注入 gateway.process 的偏函数;adapter 不认识 Gateway

    # 管理 UI 辅助(可选)
    async def list_bot_chats(self) -> list[dict]: ...
```

对比 PR:HITL 从 6 个卡片方法(`build_approval_card` / `build_resolved_card` / `send_interactive_card` / `update_card` / `register_approval` / `resolve_approval`)+ 进程内 Future 表,收敛为 `present_confirm` / `update_confirm` 两个通用方法,接口层无 "card" 概念。审批编排(超时、resume、轮数)全在 gateway,呈现全在 adapter。

### 4.2 入站事件

```python
class ChannelEvent(BaseModel):
    channel_id: str
    channel_user_id: str
    chat_id: str
    channel_message_id: str | None = None
    content: list[TextBlock | DataBlock] = []
    metadata: dict[str, Any] = {}          # 平台字段:chat_type 等,供路由匹配
    received_at: str

class ConfirmDecisionEvent(BaseModel):     # 卡片点击 / 文本确认,与消息走同一入口
    channel_id: str
    request_id: str                        # 不透明 token,原样从呈现处带回
    approved: bool
    actor: str                             # 操作者(审计)
```

### 4.3 Gateway

```python
class ChannelGateway:
    def __init__(self, storage, message_bus): ...   # 无 channel 注册表,无自有状态

    async def process(
        self, event: ChannelEvent | ConfirmDecisionEvent, channel: ChannelBase,
    ) -> None:
        """单一入口。消息 → 路由+收集;确认 → 取 pending+resume+续流。"""
```

### 4.4 ChannelRecord(持久化,唯一事实源)

```python
class RoutingConfig(BaseModel):
    bindings: list[ChannelBinding]

class SessionSettings(BaseModel):
    chat_model_config: dict[str, Any]                     # 必填,无全局兜底
    fallback_chat_model_config: dict[str, Any] | None = None
    permission_mode: PermissionMode = PermissionMode.DEFAULT

class ReplyPresentation(BaseModel):
    show_tool_process: bool = False        # 正向命名(原 filter_tool_messages 取反)
    show_thinking: bool = False

class ChannelRecord(BaseModel):
    # 身份与属主
    id: str                                # 服务端生成 UUID
    channel_type: str                      # "feishu" | ...
    user_id: str                           # AgentScope 侧属主(非平台数据)
    enabled: bool = True
    # 平台接入
    credentials: dict[str, Any]            # 机密;仅 schema format:password 字段加密
    platform_config: dict[str, Any] = {}   # 平台私有(如 only_at_reply)
    # 三组业务配置
    routing: RoutingConfig
    session: SessionSettings
    presentation: ReplyPresentation = ReplyPresentation()
    # 版本(dispatcher 对账依据)
    created_at: str
    updated_at: str                        # 每次写入刷新
```

- `platform_bot_id` **不入库**:写入时由 `registry.extract_bot_id(type, credentials)` 现场提取(app_id 明文),用于唯一性索引;展示处同样现场提取。
- `credentials` **部分加密**:存储层只加密标 `format:password` 的字段(如 app_secret),密钥复用应用级 KMS/env,非 channel 模块职责。

---

## 5. 存储清单

### A. 权威数据(ChannelRecord)

索引三个(PR 为 6 个):

| Key | 类型 | 用途 |
|---|---|---|
| `channel:{id}` | 记录 | 主记录(id 全局唯一,无需 user 前缀 + 反向 lookup) |
| `user:{user_id}:channels` | SET | 管理页列表 |
| `channel_botid:{bot_id}` | 值 | 接入查重(防同一 bot 被接入两次) |

### B. 临时中间态(Redis,带 TTL,替代 PR 的进程内状态)

| 数据 | Key → Value | 生命周期 |
|---|---|---|
| pending 确认 | `request_id → {session_id, agent_id, reply_id, tool_calls, 回复路由, 呈现句柄}` | 决定到达时原子取删;宽松 GC TTL |
| 媒体缓冲索引 | `(channel_id, chat_id, user_id) → [{blob_uri, mime, name, ts}]` | 下一条文本消费即清;TTL ~5 分钟 |
| seen chat_ids | `channel:{id}:seen_chats`(SET) | 长期保留,给管理 UI 选群配规则 |
| status 心跳 | `channel:liveness:{id}` hash,field=node_id → `{status, error?, since}` | `ttl_secs=30`,节点崩溃自动过期 |

### C. 字节(复用 RAG 的 `BlobStoreBase`,提升到 `app/blob_store/`)

| 数据 | key | 清理 |
|---|---|---|
| 缓冲媒体文件 | `channel/{channel_id}/{block_id}` | 消费后尽力删 + sweeper 兜底(参照 `_index_sweeper`) |

### 不再存储(对比 PR)

- SessionMappingRecord 整张表(纯函数派生)
- `channel_lookup` 反向索引、全局 channel 双重索引(UUID 化后多余)
- 进程内:审批 Future 表、媒体内存缓冲(全部进 B 类)

Channel 建的 session 走既有 session 存储,来源标记新增 `SessionSource.CHANNEL`(避免污染 Web 会话列表)。

---

## 6. 运行时序

### 6.1 消息处理(核心场景)

```
平台把消息投给某节点 A 的 WS → adapter 规范化 → await emit(event) → gateway.process(event, adapter)

process(MessageEvent):
  ① resolve(event, record) → (agent_id, session_id)          # 纯函数
  ② record_chat_id(seen_chats)
  ③ 抢 per-session 收集权租约(SETNX + TTL):
       抢不到 → inbox_push(HintBlock) + wake → 返回           # 让位给在跑的收集者
       抢到   → 继续
  ④ 订阅 session_events(session_id)                          # 订阅开始(先订阅后触发)
  ⑤ inbox_push(把本条消息作为 HintBlock) + enqueue wake
       (WakeupDispatcher 是唯一 spawn 点;session 忙则 inbox 被在跑的 run 吸收)
  ⑥ 收集,逐事件拼装(按 presentation 过滤 tool/thinking),直到:
       REPLY_END           → 复查 inbox 空且 session 未再跑 → 发回复
       REQUIRE_USER_CONFIRM→ 进 §6.2
       finished_reason=error → 发错误提示
       response_timeout    → 发超时提示
  ⑦ 取消订阅、释放租约、process 返回                          # 订阅结束
```

**碰撞域 = 派生 session_id**(非 chat_id):同 session 的并发消息由收集权租约串行化,后到者不重复订阅、只入 inbox,由在跑的收集者一并消化。**订阅生命周期 = 单次收集窗口**(几秒~60s),其余时间零订阅——与 Web SSE(页面打开即订阅)本质不同。

### 6.2 确认(HITL)——两段式,事件驱动,无阻塞等待

```
[第一段] process 收集中遇到 REQUIRE_USER_CONFIRM:
  · 持久化 pending{request_id → session, agent_id, reply_id, tool_calls, 回复路由}
  · ref = await channel.present_confirm(event, prompt)   # 唯一跨界调用;塞 request_id 进按钮
  · ref is None(平台无法呈现) → 按拒绝走第二段逻辑
  · 已收集文本先发出 → 释放租约 → 返回                    # 不等待

[用户点击,可能隔数分钟,回调可能到任意节点 B]

[第二段] adapter 收到点击 → 规范化为 ConfirmDecisionEvent → gateway.process:
  · 原子取删 pending(取不到=已处理/GC → update_confirm 回执"已失效",结束)
  · 组装 UserConfirmResultEvent(confirmed × tool_calls) → enqueue resume
  · channel.update_confirm(ref, approved|denied)
  · 抢租约 → 订阅 → 收集续流 → 发回复
```

**无超时机制**:service 语义是"需要确认时 run park,resume 到达即唤醒",不在乎间隔。pending 仅有宽松 GC TTL 防堆积,无 expired 状态。`tool_calls` 等业务数据全程在 gateway 持久化,channel 只往返不透明的 `request_id`——channel 不懂 tool call 概念。分布式天然正确:pending 在共享存储,哪个节点收到点击哪个节点处理。

### 6.3 媒体聚合

```
图片消息 → adapter 下载 → 规范化为"仅含 DataBlock 的 MessageEvent" → process
  process 识别纯媒体 → 写 blob + 缓冲索引(共享存储)→ 返回(不触发 agent)
文本消息 → process → 取出并清空该 (channel,chat,user) 的缓冲 → 并入 content → 正常处理
```

聚合策略上收到 gateway(通用 IM 行为,非飞书特有),adapter 只负责下载 + 规范化。任何节点都能取到缓冲——分布式正确。大文件存 blob、索引存引用,不入 Redis。

---

## 7. 控制面:CRUD 与集群对账

### 7.1 对账模型(替代 PR 的双轨广播)

```
ChannelService(无状态):校验 → 写 ChannelRecord → bus 发 lifecycle 通知("channel X 变了",不带数据)

ChannelLifecycleDispatcher(每节点常驻):唯一动作 = 对账
  期望 = storage 中 enabled 的记录集合
  实际 = 本节点 RunRegistry 的实例集合
  差集:该跑没跑 → 造实例+on_start+起 listener+登记
        不该跑还在跑 → 停 listener+on_stop+注销
        在跑但 version 变了 → 重启
  触发:启动时 / 收到通知 / 定时兜底(如 60s)
```

**通知只是加速器,正确性来自对账**:通知丢失最多慢一个周期,定时对账拉齐,系统自愈。所有节点(含发起修改的节点)走同一条代码路径——PR 的"本地直接调用 + 远端广播 handler"双轨消失,"广播丢失无对账"问题一并解决。类比 K8s controller loop:storage 是 spec,RunRegistry 是 status。

### 7.2 多节点 status

每 dispatcher 每 ~10s 心跳 `registry_set("channel:liveness:{id}", node_id, {status}, ttl_secs=30)`;`GET status` 做 `registry_getall` 聚合各节点视图;崩溃节点条目 TTL 过期消失。同一 channel 在 N 节点各一条 WS,该聚合视图如实反映"每个节点分别什么状态"。

此心跳/租约机制也是**重复投递的兜底**:若某平台把消息广播给所有 WS 连接,重复消息派生同一 session_id,收集权租约只放行一个。

---

## 8. 与 PR #1997 的逐项对照

| 维度 | PR #1997 | 本设计 | 理由 |
|---|---|---|---|
| 会话映射 | SessionMapper(持久表 + 锁 + 竞态删 + O(N) 找回) | uuid5 纯函数派生 | 删整个抽象与其并发问题 |
| 路由概念 | routing_rules + dm_scope 两个平级字段 | 单一 bindings 列表(规则含 agent + scope) | 概念从并列变从属 |
| scope 取值 | MAIN / PER_PEER / PER_CHAT / PER_CHANNEL_PEER | PER_CHAT / PER_CHAT_USER | 删 MAIN(边缘)与 PER_USER(隐私缺陷) |
| Channel↔Gateway | 双向对象互持 + gateway 内 channel 注册表 | 单向:process(event, channel);gateway 无注册表 | 依赖无环;实例表归 RunRegistry |
| HITL 接口 | 6 卡片方法 + 进程内 Future | present_confirm/update_confirm 2 方法 | 编排归 gateway、呈现归 adapter |
| HITL 流程 | 阻塞 await Future(120s),进程内 | 两段式事件驱动,pending 入共享存储 | 分布式正确;无长挂协程;无超时 |
| run 触发 | gateway 直接 spawn + 409 重试 + 盯 run task + 10s 宽限 | inbox + wake,复用 WakeupDispatcher | 每次 process 一个协程;删多任务编排 |
| 失败感知 | 盯 run_task 异常 | 事件流 finished_reason/error | 复用主分支能力 |
| 媒体缓冲 | 进程内存 dict + 锁 | BlobStore + Redis 索引,聚合上收 gateway | 分布式正确;adapter 减负 |
| 集群同步 | fire-and-forget 广播,无对账 | 通知 + 对账,storage 为真相 | 自愈;单一代码路径 |
| status | 本节点视图 | 心跳聚合多节点 | 如实反映集群 |
| ChannelManager | 一类混装 CRUD + 实例 + 同步(393 行) | 拆 Service / RunRegistry / LifecycleDispatcher | 单一职责,贴合 app 分层 |
| ChannelRecord | 14 字段平铺,filter_* 双否定,credentials 明文,platform_bot_id 冗余存储 | 三组子配置,正向命名,部分加密,bot_id 现场提取 | 各归其位;安全 |
| Bindings API | 3 个子资源端点(读改写,并发丢更新) | 随 PATCH channel 整体提交 | 删冗余通路 |

---

## 9. 分阶段迁移清单

### 阶段一:核心通路(可独立验证)

1. 数据模型:`ChannelRecord` / `ChannelBinding` / `RoutingConfig` 等;Storage 三索引 + 部分加密;`SessionSource.CHANNEL`。
2. `resolve()` 纯函数 + 单元测试(覆盖 per_chat / per_chat_user / 兜底 / 顺序命中)。
3. `ChannelGateway.process` 消息路径:租约 + 订阅 + inbox/wake + 收集。
4. **修补 WakeupDispatcher**:对已删除 session 的 trigger 当前静默丢弃 → 改为发 `ReplyEndEvent(finished_reason=error)`,使收集者不必等超时。
5. `ChannelBase` 新接口 + `bind` 装配。

### 阶段二:控制面

6. `ChannelService`(CRUD + 校验 + 通知)。
7. `ChannelRunRegistry` + `ChannelLifecycleDispatcher`(对账 + 心跳)。
8. Router 收敛:删 3 个 binding 端点 + `test` 占位;`status` 改聚合;保留 chat_ids。

### 阶段三:确认与媒体

9. 两段式确认:pending 存储 + `present_confirm` / `update_confirm` + `ConfirmDecisionEvent` 入口。
10. 媒体聚合上收 gateway;`BlobStoreBase` 提升到 `app/blob_store/` + sweeper。

### 阶段四:Feishu adapter 重写

11. 拆 `_connection` / `_client` / `_normalize` / `_present`。
12. **消除 SDK 私有 API hack**(`ws_module.loop` monkeypatch、`_connect()`/`_ping_loop()`):优先公开 `client.start()` + `run_coroutine_threadsafe` 单向桥(方案一);不行退版本锁定隔离(方案二)。实连 SDK 时定夺。
13. 删 `_pending_attachments`(聚合已上收)。

### 阶段五:前端

14. 两个 dialog 合并为共享表单组件。
15. routing 做成可拖拽规则列表 + 规则测试器。
16. channel 类型从 `/types` API 取,删硬编码;删 `DM_SCOPES` 常量。

---

## 10. 已知待实现时定夺的点

- **Feishu SDK 连接方案**(§8 阶段四):公开入口 vs 私有隔离,取决于 SDK 重连能力是否够用,实连时定。
- **凭据加密密钥来源**:复用应用级 KMS/env 的具体 hook 形式,与运维确认。
- **收集权租约 TTL / response_timeout 具体值**:压测后定。
