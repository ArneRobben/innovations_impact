VIRTUALENV := build/virtualenv

$(VIRTUALENV)/.installed: requirements.txt
	if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	virtualenv -p python3 $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip install -r $<
	touch $@
