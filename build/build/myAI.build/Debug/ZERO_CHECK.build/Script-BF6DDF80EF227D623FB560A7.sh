#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/jiang/Desktop/myAI/build
  make -f /Users/jiang/Desktop/myAI/build/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/jiang/Desktop/myAI/build
  make -f /Users/jiang/Desktop/myAI/build/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/jiang/Desktop/myAI/build
  make -f /Users/jiang/Desktop/myAI/build/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/jiang/Desktop/myAI/build
  make -f /Users/jiang/Desktop/myAI/build/CMakeScripts/ReRunCMake.make
fi

