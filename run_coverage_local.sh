#!/bin/bash

coverage run -m pytest tests/ --tb=short -m "not nocov" -x --maxfail=3
coverage report -m
