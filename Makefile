.PHONY: install clean uninstall rebuild doc undoc

current_dir := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

install:
	python3 setup.py --path ${NBISDIR} -q build_ext install
	
dev:
	python3 setup.py --path ${NBISDIR} -q build_ext --inplace develop
	
clean:
	python3 setup.py clean --all
	rm -rf dist $(current_dir)/pynger_DottD.egg-info
	
uninstall:
	pip3 uninstall -y pynger-DottD
	
rebuild: clean uninstall install

doc:
	SPHINX_APIDOC_OPTIONS="members,undoc-members,show-inheritance,special-members,private-members" sphinx-apidoc -d3 -fPMF --implicit-namespaces -HPynger -A"Filippo Santarelli" -V0.0.1 -o doc pynger
	echo "\n\nhtml_theme = 'sphinx_rtd_theme'" >> doc/conf.py
	echo "\n\n# Add napoleon to the extensions list" >> doc/conf.py
	echo "napoleon_use_param = True" >> doc/conf.py
	echo "\nextensions += ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.todo', 'sphinx_autodoc_typehints']" >> doc/conf.py
	make -C doc html
	
undoc:
	rm -fr doc