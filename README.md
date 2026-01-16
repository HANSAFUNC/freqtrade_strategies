# freqtrade_strategies
Trading Strategies for Freqtrade,It might be an ML strategy, an LLM strategy or a technical analysis strategy

[中文](Chinese.md) | [English](English.md)

# Freqtrade 策略分享

欢迎来到 Freqtrade 策略分享社区！这里汇集了各种类型的交易策略，包括机器学习策略、大语言模型策略和技术分析策略。

## 📊 策略类型

### 🤖 机器学习策略 (ML Strategies)

机器学习策略利用先进的算法和模型来识别市场模式并做出交易决策。这些策略通常包括：

- **FreqAI 策略**: 使用 Freqtrade 的 FreqAI 框架构建的自适应机器学习模型
- **LSTM 策略**: 基于长短期记忆网络的时序预测策略
- **XGBoost 策略**: 使用梯度提升算法进行价格预测
- **强化学习策略**: 通过强化学习训练的交易代理



### 🧠 大语言模型策略 (LLM Strategies)

大语言模型策略利用自然语言处理和 AI 模型来分析市场情绪、新闻和社交媒体数据：

- **情感分析策略**: 基于市场情绪和新闻分析
- **多模态分析策略**: 结合文本、图表和技术指标
- **GPT 增强策略**: 使用 GPT 模型进行市场分析



### 📈 技术分析策略 (Technical Analysis Strategies)

技术分析策略基于传统的技术指标和图表模式：

- **指标组合策略**: 结合多个技术指标（RSI、MACD、布林带等）
- **波段交易策略**: 基于价格波动和支撑阻力位
- **趋势跟踪策略**: 识别并跟随市场趋势
- **套利策略**: 利用价格差异和套利机会



## 🎯 策略特点

### 优化工具集成

- **Optuna 优化**: 自动参数优化，寻找最佳策略参数
- **Hyperopt 优化**: 使用遗传算法和贝叶斯优化
- **回测分析**: 完整的历史数据回测和性能分析

### 高级功能

- **多时间框架分析**: 同时分析多个时间周期的市场数据
- **动态止损止盈**: 根据市场条件自动调整止损止盈水平
- **风险管理**: 内置仓位管理和风险控制机制
- **市场状态识别**: 自动识别趋势、震荡、反转等市场状态

## 📚 策略使用指南

### 1. 选择策略

根据您的交易风格和风险偏好选择合适的策略：
- **保守型**: 选择趋势跟踪或波段交易策略
- **激进型**: 选择套利或高频交易策略
- **平衡型**: 选择多指标组合策略

### 2. 回测验证

在使用策略进行实盘交易前，务必进行充分的回测：

```bash
# 下载历史数据
freqtrade download-data --exchange binance --timeframe 5m --days 30

# 运行回测
freqtrade backtesting --strategy YourStrategy --timeframe 5m
```

### 3. 参数优化

使用 Optuna 或 Hyperopt 优化策略参数：

```bash
# Optuna 优化
freqtrade hyperopt --strategy YourStrategy --hyperopt-loss SharpeHyperOptLoss --epochs 100

# 查看优化结果
freqtrade hyperopt-show
```

### 4. 模拟交易

在实盘前先进行模拟交易（Dry-Run）：

```bash
freqtrade trade --strategy YourStrategy --dry-run
```

## 🔧 策略开发

### 创建新策略

```bash
# 使用模板创建新策略
freqtrade new-strategy --strategy MyNewStrategy
```

### 策略结构

每个策略通常包含以下核心方法：

- `populate_indicators()`: 计算技术指标
- `populate_entry_trend()`: 定义入场条件
- `populate_exit_trend()`: 定义出场条件
- `custom_stoploss()`: 自定义止损逻辑
- `custom_exit()`: 自定义出场逻辑

## 📊 性能监控

### 查看交易统计

```bash
# 查看所有交易
freqtrade show-trades

# 查看性能报告
freqtrade backtesting-analysis
```

### WebUI 监控

启动 WebUI 进行可视化监控：

```bash
freqtrade webserver
```

访问 `http://localhost:8080` 查看实时交易状态和图表。

## 🤝 社区支持

### 加入我们的 Discord 社区

**🎉 [点击加入 Freqtrade 策略分享 Discord 社区](https://discord.gg/d5ce3xtAPb)**

在 Discord 社区中，您可以：

- 📢 **策略分享**: 分享您的策略和经验
- 💬 **技术讨论**: 与其他交易者讨论策略开发
- 🐛 **问题求助**: 获得社区成员的帮助和支持
- 📈 **实盘分享**: 分享实盘交易结果和心得
- 🎓 **学习资源**: 获取策略开发教程和最佳实践
- 🔄 **策略更新**: 第一时间获取策略更新和优化建议

### 社区规则

- 尊重他人，保持友好的讨论氛围
- 分享策略时请注明风险提示
- 禁止发布任何形式的投资建议
- 鼓励分享代码和思路，共同进步

## ⚠️ 风险提示

**重要声明**: 

- 所有策略仅供学习和研究使用
- 加密货币交易存在高风险，可能导致本金损失
- 使用任何策略前请充分理解其工作原理
- 建议先在模拟环境中测试策略
- 实盘交易前请做好风险管理
- 作者不对任何交易损失负责

## 📝 贡献指南

欢迎贡献您的策略和经验！

1. **分享策略**: 在 Discord 社区分享您的策略代码
2. **报告问题**: 发现策略问题请及时反馈
3. **改进建议**: 提出策略优化建议
4. **文档完善**: 帮助完善策略文档和说明

## 🔗 相关资源

- [Freqtrade 官方文档](https://www.freqtrade.io)
- [Freqtrade GitHub](https://github.com/freqtrade/freqtrade)
- [策略开发指南](https://www.freqtrade.io/en/stable/strategy-customization/)
- [FreqAI 文档](https://www.freqtrade.io/en/stable/freqai/)

## 📧 联系方式

如有任何问题或建议，欢迎通过以下方式联系：

- **Discord**: [加入我们的 Discord 社区](https://discord.gg/d5ce3xtAPb)
- **GitHub Issues**: 提交问题或功能请求

---

**Happy Trading! 🚀**

*最后更新: 2026年*

