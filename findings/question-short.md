# 短版消息（正文）

各位老师好，

我在做 8 月社区任务 aclblasCtbmv（complex64，A2/A3，arch22）。功能已完成，
1872/1872 用例通过（golden 用 cblas_ctbmv），但性能离标杆有 3.5~6 倍差距，
想在提交验收前先确认一个问题。

**在 Ascend910B3 / CANN 9.1.0 上实测（n=512，complex64，预热后采样 100 次）：**

| 调用路径 | 耗时 |
|---|---|
| `aclrtMemcpyAsync` D2D（不启动 kernel） | 2.8 us |
| `asdBlasCcopy`（SIP plan 模式） | 17.2 us |
| 我们的 `aclblasCtbmv` k=0 | 23.0 us |
| `aclblasCcopy`（ops-blas 现网） | 28.5 us |
| `aclblasCscal`（ops-blas 现网） | 157.0 us |

即：**一次 kernel 下发的固定开销约 14.4us**（17.2 − 2.8），这已是平台上
最优调用模型（SIP plan）的水平；而且我们的 k=0 路径在 n=1/2/4/8/512 都是
~23us，与数据规模无关，说明是纯固定开销。

**任务书 case 1（n=512, k=8）的标杆是 10us，低于这个 14.4us 的下发开销。**

想请教：**在 arch22 上，complex64 的 BLAS-2 算子应通过什么路径达到 10us？
Ctbmv 的标杆是在 arch22 实测的，还是参考了 arch35（950PR）的数据？**

我们注意到已合入的 `aclblasCtrmv`（PR #365/#369）在 n=512 报告 9.47us，
但那是 arch35 实现，依赖 SIMT 与 `asc_shfl_down`，arch22 上不可用。

附件里还有三个相关问题（ops-blas 现网算子每次调用的阻塞 H2D、arch22 是否
可用 MIX 模式），以及一个**与本任务无关但可复现的缺陷**：已合入的
`aclblasStbmv`(arch22) 带状矩阵按行主序索引，与 BLAS 列主序约定不符，
和 cblas_stbmv(CblasColMajor) 结果不一致（现网 UT 因为数据生成用了同样的
错误约定所以没有发现）。

完整数据和复现代码见附件。谢谢！
