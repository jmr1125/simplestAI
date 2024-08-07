all:
	ninja -C build -f build-Debug.ninja
	ninja -C build -f build-Release.ninja
