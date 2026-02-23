release-%:
	hatch version $*
	git add pyproject.toml
	git commit -m "chore: release $$(hatch version)"
	git push
