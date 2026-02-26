#!/usr/bin/env bash
#
# EKAP Anomaly Detection – Local Demo
# End-to-end pipeline: parse → filter → featurize → train → infer → score → serve
#
set -euo pipefail

echo "========================================="
echo "  EKAP Anomaly Detection – Local Demo"
echo "========================================="

# 1. Install
echo "📦 Installing package..."
pip install -e ".[dev]"

# 2. Parse
echo ""
echo "📄 Step 1: Parsing IIS logs..."
ekap-anom parse --input examples/sample_logs/ --out data/parsed

# 3. Filter dynamic
echo ""
echo "🔍 Step 2: Filtering dynamic pages..."
ekap-anom filter-dynamic --in data/parsed --out data/dynamic_only

# 4. Featurize Layer 1
echo ""
echo "📊 Step 3: Computing Layer 1 features..."
ekap-anom featurize-layer1 --in data/dynamic_only --out data/features/layer1

# 5. Infer Layer 1
echo ""
echo "🎯 Step 4: Scoring Layer 1 windows..."
ekap-anom infer-layer1 --in data/features/layer1 --out data/features/layer1

# 6. Featurize Layer 2
echo ""
echo "📊 Step 5: Computing Layer 2 features..."
ekap-anom featurize-layer2 --in data/dynamic_only --layer1-dir data/features/layer1 --out data/features/layer2

# 7. Train Layer 2
echo ""
echo "🧠 Step 6: Training Layer 2 model..."
ekap-anom train-layer2 --in data/features/layer2 --mode iforest --model-out models/layer2_iforest.pkl

# 8. Score
echo ""
echo "⚡ Step 7: Generating alerts..."
ekap-anom score --date 2024-04-22 --out data/alerts

# 9. Serve (optional)
echo ""
echo "🚀 Step 8: Starting API server..."
echo "   Press Ctrl+C to stop"
ekap-anom serve --host 0.0.0.0 --port 8000
