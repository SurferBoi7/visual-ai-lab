---
name: Identity-First model flow
description: How new model creation and silent-save work in Tiny AI Foundry after the Phase 1 refactor.
---

## The rule
"New Base Model" → `IdentityModal` (name + emoji + description) → `handleIdentitySubmit` creates the IndexedDB record with the model's permanent id → navigates to Training Lab.

The Save button (sidebar + mobile header) calls `handleSilentSave`:
- If `currentModelIdRef.current` is set → `store.put(model)` with the **same id** — silent in-place update, no duplicate, no modal.
- If no id (legacy model loaded from JSON import) → falls back to opening `SaveModal`.

**Why:** The old flow opened the Save dialog after training, creating duplicate records with new IDs for the same logical model. The fix is to persist the identity first (before any training), then every subsequent save is just an update.

**How to apply:** Any function that needs the current model's name/emoji must look it up from `savedModels.find(m => m.id === currentModelIdRef.current)` — never regenerate it from the llmConfig.
