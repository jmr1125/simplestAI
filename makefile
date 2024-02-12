all:
	[ -e buildninja ]||echo no config ninja
	[ -e buildxcode ]||echo no config xcode
	-$(MAKE) all-ninja -j3 -k
	-$(MAKE) all-xcode -k
config-ninja:
	cmake -S. -Bbuildninja -G "Ninja Multi-Config" -D CMAKE_PREFIX_PATH="./OpenCL-Headers/install;OpenCL-ICD-LOADER/install"
config-xcode:
	cmake -S. -Bbuildxcode -G Xcode -D CMAKE_PREFIX_PATH="./OpenCL-Headers/install;OpenCL-ICD-LOADER/install"
Release:
	cd buildninja&&ninja -f build-Release.ninja
Debug:
	cd buildninja&&ninja -f build-Debug.ninja
RelWithDebInfo:
	cd buildninja&&ninja -f build-RelWithDebInfo.ninja
all-ninja:
	$(MAKE) Release
	$(MAKE) Debug
	$(MAKE) RelWithDebInfo
all-xcode:
	cd buildxcode&&xcodebuild -quiet
