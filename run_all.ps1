# Step 1: Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Step 2: Run the full pipeline
python run_all.py

Write-Host "`n✅ All steps completed successfully!" -ForegroundColor Green
