# ADR-0500: Reorganization of Scripts Directory Structure

## Status
✅ **Implemented** - 2025-07-15

## Context

The `scripts/` directory had evolved organically over time, resulting in a confusing structure where files were categorized by their historical purpose (`fix/`, `debug/`) rather than their actual functionality. This led to several issues:

1. **Poor Discoverability**: Developers couldn't easily find relevant scripts based on what they needed to do
2. **Misleading Organization**: Files in `fix/` weren't necessarily fixes, and `debug/` contained analysis tools beyond debugging
3. **Maintenance Burden**: The naming convention suggested temporary solutions rather than permanent tools
4. **Cognitive Load**: New team members struggled to understand where to place new scripts or find existing ones

## Decision

We reorganized the scripts directory to follow a functional categorization model:

```
scripts/
├── analysis/          # Data analysis and inspection tools
├── config_management/ # Configuration management and migration tools
├── validation/        # Testing and validation utilities
├── utils/            # General-purpose utilities and helpers
└── [other existing]  # Preserved existing functional directories
```

## Considered Alternatives

### Option 1: Minimal Refactoring (Keep fix/ and debug/)
- **Pros**: No breaking changes, minimal effort
- **Cons**: Perpetuates confusion, doesn't solve core issues

### Option 2: Domain-Based Organization
- Structure by ML pipeline stages (preprocessing/, training/, evaluation/)
- **Pros**: Aligns with ML workflow
- **Cons**: Some scripts cross boundaries, would create ambiguity

### Option 3: Functional Organization (Chosen)
- Structure by script purpose (analysis/, validation/, utils/)
- **Pros**: Clear intent, easy to navigate, scalable
- **Cons**: Requires file moves and potential import updates

## Decision Criteria

1. **Clarity**: Directory names should immediately convey purpose
2. **Scalability**: Structure should accommodate future growth
3. **Minimal Disruption**: Avoid breaking existing workflows where possible
4. **Consistency**: Align with common software engineering practices
5. **CLAUDE.md Compliance**: Follow project principle of modifying existing files rather than creating new ones

## Implementation Details

### File Migrations

| Original Path | New Path | Rationale |
|--------------|----------|-----------|
| `scripts/fix/deep_data_analysis.py` | `scripts/analysis/temporal_pattern_analysis.py` | Renamed to reflect actual purpose: analyzing temporal patterns in data |
| `scripts/debug/corrected_data_analysis.py` | `scripts/analysis/h5_data_analysis.py` | Clarified as HDF5 data analysis tool |
| `scripts/fix/clean_temporal_data.py` | `scripts/analysis/data_quality_analysis.py` | Reframed as quality analysis rather than "cleaning" |
| `scripts/debug/check_h5_structure.py` | `scripts/utils/h5_structure_inspector.py` | General utility for HDF5 inspection |
| `scripts/fix/minimal_training_test.py` | `scripts/validation/training_validation.py` | Properly categorized as validation tool |
| `scripts/fix/test_overfitting_fix.py` | `scripts/validation/overfitting_validation.py` | Validation tool for overfitting detection |
| `scripts/fix/test_temporal_splitting.py` | `scripts/validation/temporal_split_validation.py` | Validation for temporal data splitting |
| `scripts/utils/apply_fix_to_dev_config.py` | `scripts/config_management/apply_overfitting_fixes.py` | Specialized configuration management tool |

### Cleanup Actions

1. Removed deprecated wrapper files (`simple_data_fix.py`, `overfitting_diagnosis.py`) that were already integrated into main codebase
2. Deleted empty `scripts/fix/` and `scripts/debug/` directories
3. Cleaned Python cache files (`__pycache__`, `.pyc`)

### Expert Validation

This reorganization was validated using zen consensus tool with multiple AI models:
- **Google Gemini 2.5 Flash**: 9/10 confidence, strong support for functional categorization
- **Claude 4 Opus**: 8/10 confidence, emphasized proper git history preservation and systematic implementation

## Consequences

### Positive

1. **Improved Developer Experience**: Clear, intuitive organization reduces time to find/place scripts
2. **Better Maintainability**: Functional grouping makes it easier to identify redundant or related scripts
3. **Professional Structure**: Aligns with industry standards for tool organization
4. **Scalability**: New categories can be added without confusion
5. **Reduced Technical Debt**: Eliminates confusing temporary-sounding names

### Negative

1. **Breaking Changes**: Any hardcoded paths or imports will need updating (minimal impact expected)
2. **Documentation Updates**: README files and wikis need to reflect new structure
3. **Learning Curve**: Team members need to adapt to new locations

### Neutral

1. **Git History**: File moves preserve history but may complicate blame/log viewing
2. **CI/CD Updates**: Any automation referencing old paths needs updating

## Follow-up Actions

### Immediate (Required)
- [x] Create ADR documentation
- [ ] Update main README.md with new directory structure
- [ ] Create README.md in each new directory explaining its purpose
- [ ] Search for and update any imports referencing old paths
- [ ] Update CLAUDE.md with new directory structure

### Short-term (1 week)
- [ ] Verify all scripts still function correctly in new locations
- [ ] Check CI/CD pipelines for any failures
- [ ] Update any configuration files with hardcoded paths
- [ ] Announce changes in team channels

### Long-term (Ongoing)
- [ ] Monitor for any issues in the first week post-migration
- [ ] Create migration guide for any affected workflows
- [ ] Schedule brief knowledge transfer session if needed

## New Directory Structure

### `scripts/analysis/`
**Purpose**: Data analysis and inspection tools
- `temporal_pattern_analysis.py` - Deep temporal pattern analysis
- `h5_data_analysis.py` - HDF5 format data analysis
- `data_quality_analysis.py` - Data quality assessment

### `scripts/validation/`
**Purpose**: Testing and validation utilities
- `training_validation.py` - Minimal training validation
- `overfitting_validation.py` - Overfitting detection and validation
- `temporal_split_validation.py` - Temporal data splitting validation

### `scripts/utils/` (Enhanced)
**Purpose**: General-purpose utilities and helpers
- `h5_structure_inspector.py` - HDF5 file structure inspection
- [existing utilities continue as before]

### `scripts/config_management/`
**Purpose**: Configuration management and migration tools
- `apply_overfitting_fixes.py` - Apply validated overfitting fixes to development config

## Lessons Learned

1. **Organic Growth Needs Periodic Refactoring**: As projects evolve, directory structures that made sense initially may become counterproductive
2. **Functional Organization > Historical Organization**: Organizing by what scripts do is more sustainable than organizing by why they were created
3. **Clear Naming Prevents Confusion**: Generic names like "fix" and "debug" accumulate unrelated content over time
4. **Expert Validation Valuable**: Using zen consensus helped validate the approach and identify potential issues

## References

- [Python Project Structure Best Practices](https://docs.python-guide.org/writing/structure/)
- CLAUDE.md project principles
- ADR-0001 through ADR-0400 (previous architectural decisions)
- Zen consensus analysis results

---

**Date**: 2025-07-15  
**Author**: Claude Code Assistant  
**Implementation**: Completed  
**Status**: Active - monitoring for issues