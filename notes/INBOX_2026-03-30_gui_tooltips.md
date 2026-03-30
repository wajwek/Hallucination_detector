# Inbox: GUI Tooltips + Extra Config Options (2026-03-30)

## What was added/changed
- Added longer, beginner-friendly tooltips (`help=`) for the key GUI controls in Dataset/Config/Run/History.
- Fixed the tooltip semantics for `cluster_threshold`: the cluster method flags when `cluster_score > threshold`, so a higher threshold is *less* strict.
- Exposed missing config options in the GUI:
  - Cluster method: `min_chunks`, `sentence_min_chars`
  - Score-based aggregation: per-method weights (`weight_sgi`, `weight_dc`, `weight_cai`, `weight_cluster`)
- Copy-settings from History now also copies the new cluster settings and aggregation weights.

## How to verify (click path)
1. Start GUI: `streamlit run hd\gui\app.py`
2. Dataset tab:
   - Hover the tooltip icon on the file uploader.
   - If you upload a raw HaluEval file, hover the tooltips for the conversion checkbox/button.
3. Config tab:
   - Hover tooltips for dataset column mapping, embedding options, methods, thresholds.
   - Enable Cluster and check you can set `k_clusters`, `min_chunks`, `sentence_min_chars`.
   - Set aggregation to `score_based` and adjust weights.
4. Run tab:
   - Hover tooltips for auto/manual run prefix and the Run button.
   - Run a detection; confirm it finishes and saves artifacts.
5. History tab:
   - Select a past run, click Apply settings, then confirm the Config tab reflects the copied parameters (including weights and cluster settings).

## Update: History delete + time column
- History ranking now includes a created_at column parsed from run_id (fallback: folder mtime).
- You can mark runs with the Delete checkbox and delete them with confirmation; this removes only rtifacts/runs/<run_id>/.


## Update: Permanent delete
- History tab now has two delete actions: Delete selected runs (runs only) and Permanent delete (runs + unused caches; optional unused uploads).
- Permanent delete cleans rtifacts/embeddings/<dataset_id>/<embed_config_id> and rtifacts/datasets/<dataset_id> only if no remaining runs reference them.

