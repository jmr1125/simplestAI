#include "function.hpp"
#include "main.hpp"
#include "network.hpp"
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>
#ifdef USE_OMP
#include <omp.h>
#endif
using namespace std;

network net({}, NULL, NULL);

int w = 64, h = 64;
int getgen(int x, int y) {
  matrix input;
  input.setn(2);
  input.setm(1);
  input(0, 0) = (valT)x / h;
  input(1, 0) = (valT)y / w;
  auto v = net.feed_forward(input);
  return (int)(v(0, 0) * 256) | ((int)(v(1, 0) * 256) << 8) |
         ((int)(v(2, 0) * 256) << 16);
}
void train() {
  matrix input;
  input.setn(2);
  input.setm(1);
#ifdef USE_OMP
#pragma omp for
#endif
  for (int x = 0; x < h; ++x) {
    for (int y = 0; y < w; ++y) {
      input(0, 0) = (valT)x / h;
      input(1, 0) = (valT)y / w;
      VvalT expect;
      if (x % 4 && y % 4) {
        expect = {1, 1, 1};
      } else {
        expect = {0, 0, 0};
      }
      net.backpropagation(input, expect, 0.00001);
    }
  }
}
void initnet() {
  cout << "init" << endl;
  ifstream fin("picture.net");
  if (fin) {
    net.load(fin);
    for (layer &l : net.layers) {
      l.Func() = funcof(ReLU);
      l.deFunc() = defuncof(ReLU);
    }
    if (!fin)
      cerr << "not good file: picture.net" << endl;
  } else {
    net = network({2, 128, 3}, pair_ReLU);
    /*randomize net*/
#ifdef USE_OMP
#pragma omp for
#endif
    for (layer &l : net.layers) {
      for (auto &x : l.w.m) { // rand matrix
        for (auto &x : x) {
          x = (genvalT() - 0.5) * 2 * (1 / sqrt(l.w.getm()));
        }
      }
      for (auto &x : l.b.m) { // rand basis
        for (auto &x : x) {
          x = (genvalT() - 0.5) * 2 * (1 / sqrt(l.b.getn()));
        }
      }
    }
  }
}

int main() {
#ifdef USE_OMP
  omp_set_dynamic(1);
#endif
  Display *d = XOpenDisplay(0);
  if (!d) {
    cerr << "no display" << endl;
    return 1;
  }
  int root = XDefaultRootWindow(d);
  int defaultScreen = XDefaultScreen(d);

  int bitdepth = 24;
  XVisualInfo visinfo = {};
  if (!XMatchVisualInfo(d, defaultScreen, bitdepth, TrueColor, &visinfo)) {
    cerr << "no matching visual info" << endl;
    return 1;
  }
  XSetWindowAttributes winattr;
  winattr.backing_pixel = 0;
  winattr.colormap = XCreateColormap(d, root, visinfo.visual, AllocNone);
  auto attrmask = CWBackPixel | CWColormap | CWEventMask;
  Window window =
      XCreateWindow(d, root, 0, 0, w+20, h+20, 0, visinfo.depth, InputOutput,
                    visinfo.visual, attrmask, &winattr);
  if (!window) {
    cerr << "can't create window" << endl;
  }
  XSelectInput(d, window,
               ExposureMask | ButtonPressMask | ButtonReleaseMask |
                   EnterWindowMask | LeaveWindowMask | PointerMotionMask |
                   FocusChangeMask | KeyPressMask | KeyReleaseMask |
                   KeymapStateMask | SubstructureNotifyMask |
                   StructureNotifyMask | SubstructureRedirectMask);
  XStoreName(d, window, "picture");

  XMapWindow(d, window);
  int stop = false;

  GC pen;
  XGCValues values;
  Colormap cmap;
  cmap = DefaultColormap(d, defaultScreen);
  values.foreground = WhitePixel(d, defaultScreen);
  values.line_width = 1;
  values.line_style = LineSolid;
  pen = XCreateGC(d, window, GCForeground | GCLineWidth | GCLineStyle, &values);
  thread th([&]() {
    int pixelBits = 32;
    int pixelBytes = pixelBits / 8;
    int windowBufferSize = w * h * pixelBytes;
    char *mem = (char *)malloc(windowBufferSize);
    int t = 0;

    XImage *xWindowBuffer = XCreateImage(d, visinfo.visual, visinfo.depth,
                                         ZPixmap, 0, mem, w, h, pixelBits, 0);
    int pitch = w * pixelBytes;
    initnet();
    while (!stop) {
      cout << ++t << " copy ";
      cout.flush();
      for (int x = 0; x < h; ++x) {
        // cout << x << ": ";
        for (int y = 0; y < w; ++y) {
          unsigned int *px = (unsigned int *)(mem + x * pitch + y * pixelBytes);
          *px = getgen(x, y);
          // cout << *px << ' ';
        }
        // cout << endl;
      }
      cout << "train ";
      cout.flush();
      train();
      cout << "show";
      cout.flush();
      usleep(1);
      XPutImage(d, window, pen, xWindowBuffer, 0, 0, 0, 0, w, h);
      XFlush(d);
      cout << endl;
    }
  });
  while (!stop) {
    XEvent evt = {};
    while (XPending(d) > 0) {
      XNextEvent(d, &evt);
      switch (evt.type) {
      case KeymapNotify:
        XRefreshKeyboardMapping(&evt.xmapping);
        break;
      case KeyRelease: {
        char str[25];
        int len;
        KeySym keysym;
        len = XLookupString(&evt.xkey, str, 25, &keysym, NULL);
        cout << "len: " << len << endl << "press: " << str << endl;
        if (str[0] == 'q')
          stop = true;
        break;
      }
      case ConfigureNotify: {
        if (w != evt.xconfigure.width || h != evt.xconfigure.height) {
          w = evt.xconfigure.width, h = evt.xconfigure.height;
          cout << "resize h= " << h << " w= " << w << endl;
        }
      }
      }
    }
  }
  th.join();
  return 0;
}
