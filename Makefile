deploy:
	rsync -vzr --size-only --no-owner --no-group --exclude=".venv" --exclude="*.pyc" --exclude="__pycache__" --exclude="*.git" --exclude="*.DS_Store" . runpod:/workspace/verl --dry-run
