# 归档文件说明

此目录包含开发过程中产生的临时文件和文档，已从主项目目录移除以保持项目整洁。

## 目录结构

### `test_scripts/` (18个文件)
开发和调试过程中的临时测试脚本：
- `test_*.py` - 各种功能测试脚本
- 这些脚本在开发阶段用于验证功能，现已被正式的测试套件替代

### `analysis_docs/` (15个文件)
项目分析和设计文档：
- `ANALYSIS_INDEX.md` - 分析索引
- `COMPREHENSIVE_CODEBASE_ANALYSIS.md` - 全面的代码库分析
- `EXECUTIVE_SUMMARY.txt` - 执行摘要
- `DATABASE_*.md` - 数据库调优相关文档
- `BENCHBASE_INTEGRATION_*.md` - BenchBase集成文档
- 其他分析和集成文档

### `dev_scripts/` (9个文件)
开发和调试辅助脚本：
- `check_*.py` - 各种检查脚本
- `debug_import.py` - 导入调试脚本
- `fix_evaluate.py` - 评估修复脚本
- `botorch_functions.py` - 早期的BoTorch函数实现
- `consolidated_functions.py` - 旧的综合函数文件（根目录版本）

## 已删除的文件

以下文件已被永久删除（不可恢复）：

1. **备份文件**
   - `bomegabench/functions/consolidated_functions_old.py.bak`

2. **旧配置文件**
   - `requirements.txt` (已被 `pyproject.toml` 替代)

## 当前主项目文件

根目录现在只保留必要文件：
- `README.md` - 项目说明
- `REFACTORING_SUMMARY.md` - 重构总结
- `setup.py` - 安装配置
- `pyproject.toml` - 现代Python项目配置
- `.gitignore` - Git忽略规则
- `.pre-commit-config.yaml` - 代码质量检查配置

## 注意事项

- 这些归档文件可以安全删除，但保留它们可以作为开发历史参考
- 如果需要恢复某些功能，可以从这些归档文件中查找
- 建议定期清理归档目录，删除不再需要的文件

## 清理日期

2025-10-21
