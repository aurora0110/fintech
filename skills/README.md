# 技能模块目录

存放策略研究的标准技能定义，按功能模块化组织。

## 内容

```
skills/
├── alloc-equal/          # 等分资金分配
│   └── SKILL.md
├── alloc-fixed/          # 固定仓位分配
│   └── SKILL.md
├── alloc-score/          # 分数加权分配
│   └── SKILL.md
├── b1-entry/             # B1策略入口
│   └── SKILL.md
├── b2-entry/             # B2策略入口
│   └── SKILL.md
├── b3-entry/             # B3策略入口
│   └── SKILL.md
├── brick-entry/          # BRICK策略入口
│   └── SKILL.md
├── conventions/          # 研究规范约定
│   └── SKILL.md
├── exit-fixed/           # 固定止盈止损退出
│   └── SKILL.md
├── exit-minute/          # 分钟级退出
│   └── SKILL.md
├── exit-model-tp/        # 模型+止盈退出
│   └── SKILL.md
├── exit-partial/         # 部分止盈退出
│   └── SKILL.md
├── pin-entry/            # 单针策略入口
│   └── SKILL.md
└── tools/
    ├── sync.py           # 同步工具
    └── watch.py          # 监控工具
```

## 用途说明

| 模块 | 用途 |
|------|------|
| `alloc-*` | 资金分配方式定义 |
| `*-entry` | 各策略入口规则定义 |
| `exit-*` | 退出规则定义 |
| `conventions/` | 研究回测规范 |
| `tools/` | 辅助工具 |
