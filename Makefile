deploy:
	rsync -vzr --size-only --no-owner --no-group --exclude="*.git" --exclude="*.DS_Store" . runpod:/workspace/verl
