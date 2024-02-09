all:
	ninja -Cbuild -k 0
	$(MAKE) Release
#	$(MAKE) all1 -j3
config:
	cmake -S. -Bbuild -G"Ninja Multi-Config"
all1:
	$(MAKE) Release
	$(MAKE) Debug
# $(MAKE) RelWithDebInfo
Release:
	cd build&&ninja -f build-Release.ninja
Debug:
	cd build&&ninja -f build-Debug.ninja
RelWithDebInfo:
	cd build&&ninja -f build-RelWithDebInfo.ninja
