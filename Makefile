.PHONY: all install run clean

# Install dependencies
install:
	pip install -U pip
	pip install -r requirements.txt

# Run the full pipeline
run:
	python run_all.py

# Remove generated files
clean:
	rm -rf __pycache__ processed_data models data_summary.txt
