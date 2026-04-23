#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# CONFIG
# -----------------------------

# First environment/project
PROJECT1_DIR="/home/ubuntu/events_feature_extractor"
VENV1_DIR="/home/ubuntu/events_feature_extractor/venv"
SETTINGS1="/home/ubuntu/events_feature_extractor/settings_traffic.yaml"
SCRIPT1="python train_feature_extractor.py --settings_file \"$SETTINGS1\""
SCRIPT2="python extract_features.py --settings_file \"$SETTINGS1\""
CHECKPOINT="/home/ubuntu/events_feature_extractor/checkpoint/simclr/eventSym/traffic_v2/simclr_traffic_v2_eventSym_scnn_seed_10_batch_32_epoch_300.tar"

# Folder operations
SOURCE_FOLDER1="/home/ubuntu/events_feature_extractor/data/eventSym/extracted_features/eventSym/traffic_v2/training"
SOURCE_FOLDER2="/home/ubuntu/events_feature_extractor/data/eventSym/extracted_features/eventSym/traffic_v2/testing"
COPY_DEST="/home/ubuntu/original_resolution/traffic_symbols/v2/"
MOVE_DEST="/home/ubuntu/events_lifelong_learning/store/datasets/traffic/eventSym/"
DATA_DIR ="/home/ubuntu/events_lifelong_learning/store/datasets/traffic/"

# Second environment/project
PROJECT2_DIR="/home/ubuntu/events_lifelong_learning"
VENV2_DIR="/home/ubuntu/events_lifelong_learning/venv"
SETTINGS2="settings_paper.yaml"
SCRIPT3="python run_ncaltech_paper.py --settings_file \"$SETTINGS2\" --data-dir \"$DATA_DIR\""

# Logging
LOGFILE="/home/ubuntu/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Optional: auto shutdown EC2 at end
AUTO_SHUTDOWN=false
# AUTO_SHUTDOWN=true

# -----------------------------
# LOGGING
# -----------------------------
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================"
echo "Pipeline started at $(date)"
echo "Log file: $LOGFILE"
echo "========================================"

# -----------------------------
# STEP 1: activate venv in first folder
# -----------------------------
echo "[1/8] Activating first virtual environment..."
cd "$PROJECT1_DIR"
source "$VENV1_DIR/bin/activate"

# -----------------------------
# STEP 2: run first python script
# -----------------------------
echo "[2/8] Running first script..."
eval "$SCRIPT1"

# -----------------------------
# STEP 3: edit a particular line in settings file
# Example: change RUN_STAGE = 1  -> RUN_STAGE = 2
# Adjust this sed command to match your exact setting line
# -----------------------------
echo "[3/8] Editing settings file..."

sed -i "s|^file: \"\"|file: \"$CHECKPOINT\"|" "$SETTINGS1"

echo "Updated settings file line:"
grep '^RUN_STAGE' "$SETTINGS1" || true

# -----------------------------
# STEP 4: run second python script
# -----------------------------
echo "[4/8] Running second script..."
eval "$SCRIPT2"

# -----------------------------
# STEP 5: copy a certain folder to a location
# -----------------------------
echo "[5/8] Copying folder..."
mkdir -p "$COPY_DEST"
cp -r "$SOURCE_FOLDER1" "$COPY_DEST/"
cp -r "$SOURCE_FOLDER2" "$COPY_DEST/"

# -----------------------------
# STEP 6: move that folder to another directory
# -----------------------------
echo "[6/8] Moving folder..."
mkdir -p "$MOVE_DEST"
mv "$SOURCE_FOLDER1" "$MOVE_DEST/"
mv "$SOURCE_FOLDER2" "$MOVE_DEST/"
# -----------------------------
# STEP 7: deactivate first venv and activate second venv
# -----------------------------
echo "[7/8] Switching virtual environments..."
deactivate
cd "$PROJECT2_DIR"
source "$VENV2_DIR/bin/activate"

# -----------------------------
# STEP 8: run python file in second directory with second settings file
# -----------------------------
echo "[8/8] Running third script..."
eval "$SCRIPT3"

# Clean exit from second venv
deactivate

echo "========================================"
echo "Pipeline completed successfully at $(date)"
echo "========================================"

# Optional shutdown
if [ "$AUTO_SHUTDOWN" = true ]; then
    echo "Shutting down instance..."
    sudo shutdown -h now
fi