# Native INT8 EP2 batch-profile implementation plan

> For agentic workers: execute with superpowers:executing-plans; independent review at each device/publication boundary.

**Goal:** Reuse actual qualified ASM profiles at served decode/prefill shapes.
**Architecture:** Minimal exactM/fulltuple compatibility policy through current
public expert; Core loading/runner/shared/reduction remain unchanged.
**Tech Stack:** Python/Torch/vLLM0.33, UV isolatedwheels, HCU/AITER, data-only audits.
**Spec:** docs/superpowers/specs/2026-11-10-int8-ep2-native-batches.md

## Global Constraints

Frozen published/runtime inputs untouched; only evidenced shapes/plans accepted;
unknowns fallback; realgfx936 identity; no numerical/quality waiver; no CIoverride.

## Review Focus

M512/1024 change bothstageIDs/block; localprofilesoverrideglobalcompatibility;
unknownMnevertakesglobalplan; actualCore2rankworkspace4MiBlifetime; rawreload+
changedGraph; originalHE32 protocol; actualprefixgeometry/warmedcounterparity.

## Tasks

- [x] Independent stage a6c60be8 pass andownedGPU7release.
- [ ] New isolated branchfrom338 and clean CPUbaseline.
- [ ] Behavioral RED:10newqualifiedM, fulltuplemismatch rejection, unknownM
      rawfallback, localprofileprecedence. Keep fixture tied to actualstageaudit.
- [ ] Minimal policy change→GREEN; sourceCI/package andindependentreview.
- [ ] UVnewPluginwheel/separateenv with20source/wheel/installed identities.
- [ ] Installed regression+independent exact-source public2rank launchgate.
- [ ] Real2rankpublic component208metrics/182Graph andindependentterminalaudit.
- [ ] New realmodelHE32pair plus controlledwarmedc1/c4 andcacheinspection;
      independentintegrity/quality/performance/lifecycle audit.
- [ ] Fresh3repo/openMR review→normalpublication/requiredCI/merge whenallowed;
      updateprovidedskill/manifest/handoff without declaringwholegoalcomplete.
