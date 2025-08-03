# Changelog

All notable changes to the Social-xLSTM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-08-01

### Deprecated

- **SocialTrafficModel**: Centralized architecture deprecated due to scalability limitations
  - **Reason**: The centralized approach fundamentally cannot scale to distributed social traffic scenarios
  - **Impact**: Creates bottleneck that prevents proper distributed xLSTM implementation
  - **Migration**: Use DistributedSocialXLSTMModel instead
  - **Documentation**: See `docs/legacy/explanation/social-pooling-implementation-guide.md`
  - **Historical Access**: `git checkout centralized-legacy-v0.2`

- **create_social_traffic_model()**: Factory function for deprecated centralized model
  - **Alternative**: Use DistributedSocialXLSTMModel factory functions

### Added

- Comprehensive deprecation warnings with clear migration paths
- Git tag `centralized-legacy-v0.2` for historical preservation
- Enhanced documentation explaining architectural limitations

### Context

This deprecation is part of Phase 0: 緊急架構理論修正 (Emergency Architecture Theory Correction) 
to prevent further development on incorrect centralized architecture and guide migration 
to the correct distributed Social-xLSTM implementation.

## [Unreleased]

### Architecture Migration Status
- ✅ Phase 0.1: Documentation corrections completed
- ✅ Phase 0.2: Error implementation marking and cleanup
- 🔄 Phase 1: Distributed Social-xLSTM implementation (in progress)

---

## Historical Versions

For versions prior to deprecation cleanup, see:
- `git checkout centralized-legacy-v0.2` - Last stable centralized implementation
- `docs/legacy/` - Archived documentation and design decisions