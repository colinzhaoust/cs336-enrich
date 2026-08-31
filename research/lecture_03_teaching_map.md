# CS336 Spring 2026 Lecture 3 teaching map

> 目标：保留 Tatsu 实际的授课主线，在原 lecture 的时间轴和 slide 顺序上增加解释，而不是把内容重组为一组彼此平行的知识卡。
>
> 本文只包含字幕的时间化转述，不保存或复刻完整 transcript。时间点来自公开英文人工字幕，可能有数秒误差。

## 0. 结论先行

Lecture 3 不是“29 个 Transformer 知识点”。它是一场以经验调查为方法的 architecture survey：

1. 从 original Transformer 和 Assignment 1 的 modern baseline 建立问题。
2. 对每一组设计选择，先定义差异，再看跨模型采用情况和论文证据。
3. 用系统效率、优化稳定性或推理成本解释为什么某个选择成为默认值。
4. 主动给出反例、异常模型和证据边界。
5. 用 recap 收回到“哪些已形成共识，哪些仍值得试验”。

当前网站保留了大致主题顺序，但把这套 `question -> comparison -> evidence -> caveat -> recap` 反复切成独立 segment cards。结果是每张卡都重复 claim、三步 storyboard、take-away、视频说明和 notes；共享视频又会在多个 row 中重复出现。用户读到的是我们的信息架构，而不是 Tatsu 的 lecture。

正确方向是：以原视频和 67 页 deck 为连续主轴，29 个 segment ID 只保留为深链锚点；界面上合并成约 10 个连续 lecture runs。增强材料应在具体公式、证据图和讲者停顿处以内联 compare/table/link/demo 出现。

## 1. 已核实的 source boundary

| 项目 | 核实结果 |
| --- | --- |
| 课程 | Stanford CS336: Language Modeling from Scratch, Spring 2026 |
| Schedule | Mon April 6, `Architectures, hyperparameters [Tatsu]` |
| 讲者 | Tatsunori Hashimoto（schedule 中简称 Tatsu） |
| 官方 deck | [`lecture_03.pdf`](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)，67 页；本地 PDF metadata 的 CreationDate 为 2026-04-06 |
| 官方录像 | [Stanford Online - Spring 2026 Lecture 3: Architectures](https://www.youtube.com/watch?v=lVynu4bo1rY) |
| 录像边界 | 约 `00:05` 开始 lecture 内容，`1:29:10` 左右结束；不是 playlist 中的模糊范围 |
| 字幕 | 公开 `English (United States)` 人工字幕和 `English (auto-generated)` 均可用；本分析优先使用人工字幕 |
| 下一讲边界 | SSM、linear attention、MoE/attention alternatives 明确留到 Lecture 4；不要为了“完整”把这些提前塞进 L3 |

当前 `data/lectures.js` 的 L3 `videoUrl` 指向整个 playlist，应改为上面的准确单集 URL，并让每个 lecture run 有 `startSeconds/endSeconds`。

官方入口：

- [Spring 2026 course schedule](https://cs336.stanford.edu/#schedule)
- [Spring 2026 playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [Official lectures repository](https://github.com/stanford-cs336/lectures)

## 2. Tatsu 实际采用的教学语法

### 2.1 不是“定义大全”，而是“从别人的训练经验中归纳”

开头先说明：亲手训练、比较 architecture 最有效，但课程不可能穷举设计空间，因此这一讲采用 second-best 方法——横看大量模型与技术报告，找出稳定共识和仍可变的轴。这个 epistemic framing 很重要；没有它，后面的 model matrix 会看起来像一张无解释的 feature catalog。

### 2.2 每一个大单元都有固定节奏

典型节奏是：

1. 从 original Transformer 或 A1 baseline 指出一个设计轴。
2. 画公式或 topology，使选择可比较。
3. 列出采用该选择的 model families。
4. 给论文曲线、table、systems accounting 或 stability failure 作为 evidence。
5. 给 exception，说明不是数学定理。
6. recap 后才进入下一个轴。

站点应让 augmented material 填补步骤 2-4 中的理解成本，而不是在步骤 1 后另开一套独立叙事。

### 2.3 三个反复出现的解释 lens

- **表达能力 / generalization**：这个变体是否损害模型容量或 quality？
- **系统效率**：FLOPs、arithmetic intensity、data movement、parallelism、KV-cache traffic。
- **训练稳定性**：gradient propagation、softmax 数值、昂贵训练 run 是否会 spike 或 blow up。

这三条 lens 把 Lecture 2 的 resource accounting 接到了 Lecture 3。最好的 augmentation 是时间化 cross-link 到 L2 的 arithmetic-intensity/roofline 解释，而不是在 L3 右栏重复写一遍 FLOPs 定义。

### 2.4 讲者的 caveat 是内容，不是脚注

Tatsu 经常用异常模型、后续版本的回退、缺少 controlled ablation、specific-paper result 来限制自己的结论。例如 parallel blocks、T5 的 64x FFN、soft-capping、MQA quality hit 都不是一句“好/坏”能总结的。网站目前的短 claim 容易把这些压平为 prescription；新版本必须把 caveat 与 evidence 放在同一视觉单元内。

## 3. Timecoded teaching flow

下表的时间点按公开人工字幕和 slide 内容的自然切换对齐。每个时间链接都打开同一官方录像。

| 时间 | Slides | 授课动作与主线 | 合适的 augmentation | 不应在此打断的关系 |
| --- | ---: | --- | --- | --- |
| [00:05-01:51](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5s) | 1-2 | 定义 survey 方法：先自己训练；做不到穷举时，就从许多模型的经验中归纳哪些选择固定、哪些可变。 | 一张很轻的“本讲如何读”注释：`try yourself -> compare reports -> infer conservative defaults`；链接到 assignment 1。 | 不要一开场就出现 29-card 目录；会把方法论降格成 glossary。 |
| [01:52-03:21](https://www.youtube.com/watch?v=lVynu4bo1rY&t=112s) | 3-4 | 原始 Transformer：sin/cos、ReLU、post-LayerNorm。随后与 A1 的 modern baseline 对照：pre-norm、RoPE、SwiGLU、no bias。问题从“是什么”变成“为什么选这些、你该选什么”。 | 四行 before/after table；点击某一行只高亮对应 block 部位。 | original -> A1 modern baseline 是全讲的 setup，不能拆成互不相干的四个特性卡。 |
| [03:22-05:12](https://www.youtube.com/watch?v=lVynu4bo1rY&t=202s) | 5-8 | 讲者展示 2024-2026 大量 dense/MoE releases，说明样本足够丰富，可以观察 norm、activation、position、attention、vocab 等趋势；随后给全讲 outline。 | 可筛选 architecture matrix，但默认必须按年份/讲者顺序；在 matrix 上标“common / variable / next lecture”。 | slides 5-8 是 survey 动机和 roadmap，不能直接跳到 matrix filter 而省略问题。 |
| [05:13-07:40](https://www.youtube.com/watch?v=lVynu4bo1rY&t=313s) | 8-9 | 提出 architecture 同时要 learn well、run efficiently、not blow up。历史概括：早期探索 -> LLaMA-like 收敛 -> stability tweaks -> long-context hybrids。 | 三列 lens 表，以及一条很克制的历史 timeline；每项链接后续 run。 | 这段是后面所有 trade-off 的坐标系，应保持连续。 |
| [07:41-13:04](https://www.youtube.com/watch?v=lVynu4bo1rY&t=461s) | 10-12 | 先给最强共识：不要让 norm 截断主 residual route。随后依次给 warmup/convergence data、clean residual path 的 gradient 解释、gradient-spike evidence。 | 慢速 topology demo；同时保留 slide 11-12 的 evidence thumbnails 与“不是 final-quality 保证”的 caveat。 | `claim -> data -> gradient explanation` 是一个原子论证；不要把动画插在 claim 后并把 evidence 推到下一卡。 |
| [13:05-14:14](https://www.youtube.com/watch?v=lVynu4bo1rY&t=785s) | 13 | 在“norm 不应进入 residual stream”的原则上提出例外形态：把 post norm 放在 branch 外侧，或 double norm；Grok/Gemma/OLMo 是经验例子。 | topology switcher：pre、non-residual post、double；明确哪条 identity path 未被改写。 | 这是上一段原则的 extension，不是新的 normalization 课程。 |
| [14:15-20:14](https://www.youtube.com/watch?v=lVynu4bo1rY&t=855s) | 14-19 | LayerNorm 与 RMSNorm 公式 -> “更快且通常同样好”的现代解释 -> FLOPs 不等于 runtime/data movement -> 论文 runtime/perf table -> 同一 systems rationale 推广到 no-bias -> recap。中间学生提问帮助澄清 normalization 为什么 memory-heavy。 | 公式差异表；read/reduce/write 的小型 data-movement demo；Ivanov/Narang evidence 读图；no-bias 作为同一段的最后一行。 | 这是完整的六页论证。当前 10 秒视频把三个结论硬切换，既来不及读，也破坏 `formula -> systems reason -> evidence -> generalization`。 |
| [20:15-27:13](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1215s) | 20-26 | 从 activation zoo 缩小到真正重要的轴：gating。先定义普通 FFN，再增加独立投影并逐元素相乘；说明 ReGLU/GeGLU/SwiGLU 命名；提前埋下 parameter matching 的 2/3 width；随后用 Shazeer 与 Narang 的 controlled results 验证 consistent small gain，最后强调 GLU 不是必要条件。 | ReLU/GELU/Swish 公式对比表；一支只解释 content branch × gate branch 的慢 demo；parameter matched 的两列计数；直接在原 evidence table 旁做“how to read”。 | slides 22-25 的 `construction -> parameter fairness -> evidence` 必须连读。当前 segments 完全遗漏 slides 24-26 的 evidence/recap。 |
| [27:14-31:10](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1634s) | 27-29 | serial block 与 parallel block 对比。parallel 有共享 LN、fused matmul 的潜在 systems win；但后续采用下降，可能存在 expressiveness/depth trade-off，而且讲者坦率说明缺少好的 controlled ablation。最后用 architecture summary matrix 收束。 | serial/parallel critical-path diagram；一张 `claimed win / evidence gap / current adoption` table。 | 不应把它并入 GLU mechanics；也不能只动画“并行更快”，遗漏 adoption retreat 与 evidence uncertainty。 |
| [31:11-39:09](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1871s) | 30-35 | 先比较 sine、learned absolute、relative bias、RoPE 的注入位置；再定义 RoPE 想满足的 relative-inner-product 条件；通过共同旋转保持相对角度；推广到成对坐标/多频率；最后落到 sin/cos matrix 和 Q/K code path。 | 一张 position-family compare table；一支 45-60 秒、可暂停的 RoPE demo：同一相对距离 -> coupled rotation -> coordinate pairs -> code highlight。 | 目标公式、2D geometry、高维 pairing、代码实现是一条推导，不能拆成四张相互重复的 cards。 |
| [39:10-43:39](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2350s) | 35 -> 36 | RoPE 后自然 Q&A：高维旋转、如何从论文提炼经验、parallel layer 证据不一致、p-RoPE、relative bias 的 factorization 限制。讲者再次强调广读和小规模亲测。 | 可折叠 Q&A notes 和背景链接；这是放 extended reading 的安全位置。 | 不要把 Q&A 中的不确定回答升级为课程主张。 |
| [43:40-49:52](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2620s) | 36-41 | 实例化模型时才真正遇到 hyperparameter 空间。先讲 `d_ff/d_model`：4x 默认、GLU parameter matching 的约 8/3、LLaMA 的偏移、T5 64x systems experiment；Kaplan sweep 显示 1-10 的宽 basin；T5 v1.1 回归常规值形成结论。 | exact parameter-count calculator；`4x / 8/3 / LLaMA / T5` table；Kaplan curve 上的可拖动 marker，但曲线和结论不能脱离原图。 | `default -> exceptions -> empirical basin -> later retreat` 是一个完整故事。`L03-GLU-DIMENSION` 的完整解释应在这里，而不是提前独立成 activation card。 |
| [49:53-55:13](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2993s) | 42-46 | 先讲常见但非必须的 `n_heads * d_head ≈ d_model`；再讲 depth/width aspect ratio 约 100 的经验带。系统理由：太深增加 latency、pipeline parallelism 难；宽更易 tensor parallel。Kaplan/Tay evidence 表明 FLOPs 往往比精确 aspect ratio 更主导。 | 直接在官方 model table 上排序/高亮 exception；固定参数量的 depth-width calculator；链接后续 parallelism lecture。 | 不应把“约 100”显示成硬规律；必须与 broad basin 和 systems constraints 同屏。 |
| [55:14-59:20](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3314s) | 47 | vocab 观察：早期 English-centric models 常在 30-50K，多语/production systems 常在 100-250K；model scale 也是 confounder。随后 Q&A 扩展到 multimodal vocab 和 tokenizer 比较口径。 | 原表旁增加 language coverage、embedding/output parameter cost、token sequence length 三列；cross-link L1 tokenization。 | 不能把区间写成 universal recommendation；它是所列模型的观察值。 |
| [59:21-65:09](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3561s) | 48-51 | 先陈述单遍海量数据下“不需要 classical regularization”的直觉；再看实际模型仍使用 weight decay；论文 evidence 表明其可能通过 optimizer/LR schedule 改善 optimization，而非缓解 train/val gap。最后 recap 所有 hyperparameters。 | `intuition / observed practice / mechanism caveat` 三列表；将 weight-decay curve 与 cosine schedule 联动。 | slides 48-50 的反直觉转折不可拆；slide 51 summary 也不可遗漏。 |
| [65:10-74:05](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3910s) | 52-56 | 训练越昂贵，stability 的价值越高。先把两个 softmax danger zones 分开：output 与 attention。output 用 z-loss 约束 log normalizer；attention 用 QK norm 控制 Q/K scale；soft-cap 是更强、更保守但可能掉 quality 的干预。 | 统一 stability table：作用位置、公式、控制变量、adoption、quality caveat；两个 softmax 在模型图上定位；各公式只做短 scrub。 | `failure cost -> two danger zones -> three interventions -> comparative caveat` 要保持连续，不能把每个 trick 变成孤立 prescription。 |
| [74:06-75:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4446s) | 57 | 明确最后一单元只谈 dense all-to-all attention 中常用的两类干预：GQA/MQA 降 inference cost；local/sliding attention 降 long-context cost。SSM 留下讲。 | 两行 roadmap，作为前一单元与 inference 单元的显式过渡。 | 当前 segment coverage 跳过 slide 57，导致读者突然进入 KV cache。 |
| [75:01-84:59](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4501s) | 58-63 | 从 deployment 问题而不是 head diagram 开始。先比较 training/prefill 与 token-by-token decode 的 arithmetic intensity；解释 KV cache 为什么省 compute 却造成反复 memory read；再引出 MQA 的单 K/V extreme，指出 expressiveness loss；GQA 才作为可调折中；最后看质量/latency evidence，并回答 hyperparameter-search 与训练时能否改变结构等问题。 | prefill/decode accounting table；可增长 KV cache；MHA -> MQA -> GQA head-sharing morph；exact bytes calculator；page 63 evidence plot readout。 | `systems problem -> MQA extreme -> quality cost -> GQA compromise -> evidence` 是本讲最强的一条论证。当前 10.5 秒 clip 直接从 heads 开始，省掉了问题和证据。 |
| [85:08-89:10](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5108s) | 64-67 | full causal attention -> local/sliding band -> periodic full layer 让远程信息传播；Cohere、LLaMA、Gemma、OLMo 的 hybrid pattern；Qwen 的 cheap layer 换成 state-space variant；结论是 long-context 仍是活跃设计区，最后回到“共识 + 可变轴”。 | user-controlled causal matrix；local/local/local/full layer stack；模型对比 table；最终返回 slide 7/9 architecture matrix。 | local band 只是前半。若动画只展示 78 -> 42 cells 而不展示 interleaving，就没有完成讲者的结论。 |

## 4. 原 lecture 的不可打断边界

以下段落应被当成单个 `lectureRun`。可在内部放 annotation，但不应在 DOM 中变成相互重复的完整 cards：

1. **Slides 10-12 / 07:41-13:04**：pre-norm claim、data、gradient explanation。
2. **Slides 14-19 / 14:15-20:14**：RMSNorm formula、runtime reason、evidence、no-bias generalization、recap。
3. **Slides 22-26 / 21:30-27:13**：GLU construction、parameter matching、two evidence slides、caveat。
4. **Slides 30-35 / 31:11-39:09**：position-family comparison、RoPE invariant、geometry、multi-frequency、code。
5. **Slides 37-41 / 43:40-49:52**：FF default、GLU correction、T5 exception、Kaplan basin、conclusion。
6. **Slides 48-51 / 59:21-65:09**：regularization intuition、observed practice、optimizer interaction、hyperparameter recap。
7. **Slides 52-56 / 65:10-74:05**：stability motivation、two softmax zones、z-loss、QK norm、soft-cap comparative caveat。
8. **Slides 58-63 / 75:01-84:59**：prefill/decode accounting、KV-cache bottleneck、MQA、GQA、quality evidence。
9. **Slides 64-67 / 85:08-89:10**：local band、interleaved hybrid、current examples、lecture recap。

最自然的 optional-detour 边界是：RoPE 后 Q&A（约 39:10）、hyperparameter recap 后（约 65:10）、stability pause 后（约 74:06）、GQA Q&A 后（约 85:08）。背景论文、额外推导和 comments 应在这些停顿处展开，默认收起。

## 5. 29 个现有 segments 应怎样归位

不要删除 stable IDs；把它们从“页面结构”降级成“连续 lecture 内的 anchors”。建议 10 个可见 runs：

| 可见 run | 原始范围 | 保留的 deep-link anchors | 当前问题与动作 |
| --- | --- | --- | --- |
| 1. Why these modern defaults? | Slides 1-9, 00:05-07:40 | `L03-MODERN-TRANSFORMER`, `L03-ARCH-MATRIX` | 补回 slides 1-2 的 survey method 和 slide 8 roadmap；matrix 是 evidence，不是导航菜单。 |
| 2. Keep the residual stream clean | Slides 10-13, 07:41-14:14 | `L03-PRE-POST-NORM`, `L03-DOUBLE-NORM` | 两个 ID 共用一个 topology compare；第二个只是原则的 extension。 |
| 3. Normalization as systems design | Slides 14-19, 14:15-20:14 | `L03-RMSNORM`, `L03-NORM-RUNTIME`, `L03-NO-BIAS` | 合成一条 argument；视频/notes 只出现一次。修正 no-bias 的过强 copy：deck 给的是 memory + optimization-stability rationale，不是已测出的“bias 几乎没有 modeling capacity”。 |
| 4. Gated FFNs, then block topology | Slides 20-29, 20:15-31:10 | `L03-ACTIVATION-ZOO`, `L03-GLU-GATE`, `L03-SERIAL-PARALLEL` | 补回 slides 24-26 evidence 和 slide 29 architecture recap。`L03-GLU-DIMENSION` 在这里只做 foreshadow，不成为独立卡。serial/parallel 是同 run 的单独 subsection。 |
| 5. Position to RoPE | Slides 30-35, 31:11-39:09 | `L03-POSITION-FAMILIES`, `L03-ROPE-RELATIVE`, `L03-ROPE-FREQUENCIES`, `L03-ROPE-CODE` | 四个 anchor 共用连续 compare + demo；不要重复四次 claim/goal/beats。 |
| 6. Which hyperparameters actually matter? | Slides 36-51, 43:40-65:09 | `L03-FF-RATIO`, `L03-GLU-DIMENSION`, `L03-FF-BASIN`, `L03-HEAD-RATIO`, `L03-ASPECT-RATIO`, `L03-VOCAB-SIZE`, `L03-REGULARIZATION` | 把 `GLU-DIMENSION` 放回 slides 37-41 的显式推导；补回 slide 36 问题清单与 slide 51 recap。七个 anchor 是长 run 内的 chapters，不是七张完整 cards。 |
| 7. Stability: two softmax danger zones | Slides 52-56, 65:10-74:05 | `L03-Z-LOSS`, `L03-QK-NORM`, `L03-SOFT-CAP` | 共享一张模型定位图和 comparison table；soft-cap caveat 与 intervention 同屏。 |
| 8. Why decoding changes attention | Slides 57-60, 74:06-79:45 | `L03-KV-CACHE` | 补回 slide 57 roadmap，并把 Lecture 2 arithmetic intensity 作为 inline cross-link；先建立 problem，后面才出现 heads。 |
| 9. MQA to GQA | Slides 61-63, 79:45-84:59 | `L03-MQA`, `L03-GQA` | MQA 是 extreme，GQA 是 knob，page 63 evidence 是结论的一部分；共用一个 asset。 |
| 10. Long-context hybrids | Slides 64-67, 85:08-89:10 | `L03-SLIDING-WINDOW`, `L03-INTERLEAVED-ATTN` | full -> local -> interleave -> current examples -> global recap；现有动画只做了 local band 前半。 |

### 被现有 segmentation 漏掉、但对教学逻辑最重要的 slides

- **24-26**：GLU 的 controlled evidence 与 activation recap。
- **29**：architecture recap，把 norm/activation/parallel choices 收回 matrix。
- **36**：hyperparameter question list，告诉学生接下来为什么突然讨论数字。
- **51**：hyperparameter recap。
- **57**：attention-head interventions roadmap。
- **67**：全讲 recap，并返回 commonalities vs variations。

这些不必新增 segment ID，但必须出现在 source rail 中。当前页面的最大缺陷不是知识点少，而是 transition 和 recap 被筛掉了。

## 6. 现有 L3 Manim assets 的判定

四个公开 MP4 都是 `854x480`, `15 fps`，长度只有 `8.73-10.53 s`：

| Asset | 长度 | 判定 | 原因 | 下一版 |
| --- | ---: | --- | --- | --- |
| `L03PrePostNorm.mp4` | 8.73 s | **保留概念，重做/放慢** | topology 对比方向正确，但不到 9 秒内同时出现公式、两张 block、gradient arrow、take-away；没有 slide 11-12 evidence。`normalization bottleneck` 也比课程原意更绝对。 | 25-35 秒；先 post，再 morph 到 pre，再 trace identity/gradient；每个 key state 停 3-5 秒。evidence 留在旁边的原 slide，不塞进视频。 |
| `L03RuntimeNorm.mp4` | 10.07 s | **从页面下线，拆分** | 10 秒跨越 slides 14-19 的三段论：formula、runtime/data movement、no-bias。信息量太大，小字很多，动画本质是快速切三张新 slide。单纯减速仍会像重复 lecture。 | 公式用静态 compare table；runtime 用 15-25 秒 read/reduce/write demo；no-bias 用一行 annotated equation。三者共享同一 run，不再有一个“综合 micro-lecture”。 |
| `L03KVSharing.mp4` | 10.53 s | **保留 morph idea，完整重建** | MHA/MQA/GQA head diagram 是好 candidate，但当前标题直接从 cache-size knob 开始，缺少 75:01 后最关键的 prefill/decode accounting、incremental traffic、page 63 quality evidence。 | 45-70 秒，或两个 linked demos：A. cache grows + arithmetic-intensity table；B. MHA -> MQA -> GQA + evidence plot。确保 MQA 先于 GQA。 |
| `L03SlidingWindow.mp4` | 9.27 s | **改为 interactive；原视频退役** | 12-token、width-4 的 `78 -> 42` worked example 正确，但 lecture 的目的不是只数格子，而是引出 periodic full attention。当前结尾文字替代了真正的 layered propagation。 | 用 scrub/slider 控制 window width；另加 local/local/local/full layer-stack demo。worked-example 标签明确说明数字是 augmentation，不是 deck 原数值。 |

### 为什么“动画太快”不是只调一个 speed 参数

现有 pipeline 把两类资产混在了一起：

- **Illustration loop**：只表达一个空间关系，可以是 8-12 秒，但画面应极少文字，可循环、可 scrub。
- **Micro-lecture**：包含定义、推导、evidence、caveat，至少需要 40-90 秒，或直接做成用户点击推进的 stepper。

`RuntimeNorm` 和 `KVSharing` 是 micro-lecture 的信息量，却用了 illustration-loop 的时长，所以慢放 2x 仍不够。先决定 asset type，再 storyboard。

### 下一轮 render gate

1. 动画脚本必须有 source pages、video interval、一个 claim、一个 evidence boundary、一个 caveat。
2. 一个 silent loop 最多一个公式或一个 topology change；超过就改成 stepper/micro-lecture。
3. 教学 clip 目标 `1080p, 30 fps`；最小正文字号按嵌入宽度验收，不依赖全屏观看。
4. 关键公式/diagram 的 dwell 至少 3-5 秒；transition 后先静止再出现下一 annotation。
5. 默认不 autoplay。提供 pause、scrub、`0.5x/1x/1.5x`，并在关键 state 放 chapter markers。
6. QA 不只看 poster：在 1x 速度从头看完；抽取开头/中点/结尾；确认没有读到一半就消失的文字。
7. 同一 asset 在页面只 mount 一次；多个 segment anchor 指向该 asset 的内部 chapter，而不是复制视频和说明。

## 7. 最值得做的 augmentation backlog

按“补原 lecture 的理解缺口”而不是“哪个概念看起来适合 Manim”排序：

### P0 - 先修 source flow

1. **Lecture 3 player + slide timeline**：准确单集视频、time anchors、slides 1-67 filmstrip；segment IDs 作为内部 anchors。
2. **恢复 transition/evidence/recap slides**：24-26、29、36、51、57、67。
3. **RMSNorm run 重组**：公式 compare、runtime data movement、evidence table、no-bias，全部只出现一次。
4. **KV run 重建**：先解释 decode 问题，再 MQA/GQA，再证据；直接 cross-link Lecture 2 arithmetic intensity。

### P1 - 真正需要 motion 的内容

1. **RoPE slow demo**：这是 L3 最适合 Manim 的核心段，目前反而没有 rendered asset。保持 slide 31-35 的推导顺序。
2. **GLU gate + parameter fairness**：先只动画 branch multiplication；到 hyperparameter run 再用 calculator 推出 8/3。
3. **Pre/post/double norm topology**：一支连续、可暂停的 topology demo。
4. **Interleaved attention propagation**：local receptive field 如何逐层扩大，full layer 如何一次全局混合。

### P2 - table/link 比 animation 更好

- model architecture matrix 的 filter/compare。
- FF ratio、head ratio、aspect ratio、vocab range 的 table annotations。
- z-loss/QK norm/soft-cap 的 formula-and-location comparison。
- paper links 与“课程说了什么 / paper 额外给了什么”区分。

推荐 primary background links：

- [Xiong et al. 2020 - pre-LN vs post-LN](https://arxiv.org/abs/2002.04745)
- [Zhang & Sennrich 2019 - RMSNorm](https://arxiv.org/abs/1910.07467)
- [Shazeer 2020 - GLU variants](https://arxiv.org/abs/2002.05202)
- [Su et al. 2021 - RoPE](https://arxiv.org/abs/2104.09864)
- [Kaplan et al. 2020 - scaling/hyperparameter sweeps](https://arxiv.org/abs/2001.08361)
- [Ainslie et al. 2023 - GQA](https://arxiv.org/abs/2305.13245)
- [QK normalization reference used by the course](https://arxiv.org/abs/2302.05442)
- [Longformer - local + global attention](https://arxiv.org/abs/2004.05150)

## 8. 对新站点 content model 的直接建议

每个 `lectureRun` 应有：

```text
title
startSeconds / endSeconds
sourcePages
sourceIntent                 # 讲者此处试图解决什么问题
anchors[]                    # 保留 L03-* stable IDs
atomicClaimChain[]           # 不应拆散的 claim/evidence/caveat
inserts[]                    # compare | formula | table | link | demo
safeDetourAfterSeconds       # extended reading/comments 的自然停顿
```

页面呈现原则：

- 左侧/主栏始终是 official video + 当前 slide/slide range。
- augmentation 紧挨它所解释的公式或图，不形成永久“右栏第二套课程”。
- 没有实质增益的 segment 只留 anchor，不显示空 storyboard。
- 一个 run 最多一个主要 augmentation；其余是短 annotation 或折叠 background links。
- comments 挂在 `lectureRun + anchor` 上。朋友的反馈模板应问“这个 augmentation 是否帮助理解原视频的哪一句/哪张 slide”，而不是泛泛评价一个脱离语境的动画。
- Q&A 和 recap 后是讨论入口的默认位置；不要在推导中间插评论框。

## 9. Acceptance criteria for Lecture 3 refactor

1. 从页面顶部按顺序阅读，能重建本文件第 3 节的 18 个时间段，不需要使用搜索/filters 才知道讲者下一步为什么换题。
2. 任何 augmentation 都能指出准确 video interval 和 slide pages。
3. Slides 24-26、29、36、51、57、67 不再丢失。
4. `L03-GLU-DIMENSION` 的完整推导位于 hyperparameter run；activation run 只做 foreshadow。
5. RMSNorm 的三个现有 anchors 不再各自重复同一视频、sequence map、notes。
6. KV cache -> MQA -> GQA 的顺序与官方 lecture 相同，并包含 page 60 accounting 和 page 63 evidence。
7. Sliding-window augmentation 展示 periodic full-attention 的传播作用，不只展示 cell count。
8. 页面没有 480p/15fps、10 秒内包含三个概念的“教学视频”。保留的短 loop 每支只解释一个空间关系。
9. 每一个 prescription 都和它的 evidence/caveat 同屏或相邻；异常模型不会被藏进折叠脚注。
10. L3 的结尾回到 architecture matrix/common-vs-variable，而不是停在最后一个 attention trick。

