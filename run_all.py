import subprocess

def run_script(script_name):
    print(f"\n{'='*30}\n▶ Running: {script_name}\n{'='*30}")
    result = subprocess.run(["python", script_name])
    if result.returncode != 0:
        print(f"❌ {script_name} failed.")
        exit(1)
    print(f"✅ {script_name} completed.\n")

if __name__ == "__main__":
    scripts = [
        "data_preprocessing.py",
        "eda.py",
        "train_models.py",
        "evaluate_models.py"
    ]

    print("🚀 Starting the Credit Scoring Project Pipeline...\n")
    for script in scripts:
        run_script(script)

    print("\n🎉 All steps completed successfully!")
