# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/jiang/myAI

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/jiang/myAI/build

# Include any dependencies generated for this target.
include CMakeFiles/readdata.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/readdata.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/readdata.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/readdata.dir/flags.make

CMakeFiles/readdata.dir/readmnist.cpp.o: CMakeFiles/readdata.dir/flags.make
CMakeFiles/readdata.dir/readmnist.cpp.o: /Users/jiang/myAI/readmnist.cpp
CMakeFiles/readdata.dir/readmnist.cpp.o: CMakeFiles/readdata.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jiang/myAI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/readdata.dir/readmnist.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/readdata.dir/readmnist.cpp.o -MF CMakeFiles/readdata.dir/readmnist.cpp.o.d -o CMakeFiles/readdata.dir/readmnist.cpp.o -c /Users/jiang/myAI/readmnist.cpp

CMakeFiles/readdata.dir/readmnist.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/readdata.dir/readmnist.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jiang/myAI/readmnist.cpp > CMakeFiles/readdata.dir/readmnist.cpp.i

CMakeFiles/readdata.dir/readmnist.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/readdata.dir/readmnist.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jiang/myAI/readmnist.cpp -o CMakeFiles/readdata.dir/readmnist.cpp.s

# Object files for target readdata
readdata_OBJECTS = \
"CMakeFiles/readdata.dir/readmnist.cpp.o"

# External object files for target readdata
readdata_EXTERNAL_OBJECTS =

readdata: CMakeFiles/readdata.dir/readmnist.cpp.o
readdata: CMakeFiles/readdata.dir/build.make
readdata: CMakeFiles/readdata.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/jiang/myAI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable readdata"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/readdata.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/readdata.dir/build: readdata
.PHONY : CMakeFiles/readdata.dir/build

CMakeFiles/readdata.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/readdata.dir/cmake_clean.cmake
.PHONY : CMakeFiles/readdata.dir/clean

CMakeFiles/readdata.dir/depend:
	cd /Users/jiang/myAI/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jiang/myAI /Users/jiang/myAI /Users/jiang/myAI/build /Users/jiang/myAI/build /Users/jiang/myAI/build/CMakeFiles/readdata.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/readdata.dir/depend

