#include "function.hpp"
#include "main.hpp"
#include "network.hpp"
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>
#ifdef USE_OMP
#include <omp.h>
#endif
using namespace std;

network net({}, NULL, NULL);

int W = 128, w = W * 2, h = 128;
valT picture(int x, int y) {
  // return (((x + 8) / 16) % 2 && ((y + 8) / 16) % 2) ? 1.0 : 0.5;
  return (20 < x && x < 108 && 20 < y && y < 90) ? 1 : 0;
}
matrix getInput(int x, int y) {
  matrix input;
  input.setn(15);
  input.setm(1);
  int i = 0;
  input(i++, 0) = 1;
  input(i++, 0) = sin(x);
  input(i++, 0) = cos(x);
  input(i++, 0) = sin(2 * x);
  input(i++, 0) = cos(2 * x);
  input(i++, 0) = sin(3 * x);
  input(i++, 0) = cos(3 * x);
  input(i++, 0) = sin(y);
  input(i++, 0) = cos(y);
  input(i++, 0) = sin(2 * y);
  input(i++, 0) = cos(2 * y);
  input(i++, 0) = sin(3 * y);
  input(i++, 0) = cos(3 * y);
  input(i++, 0) = x;
  input(i++, 0) = y;
  return input;
}
valT getgen(int x, int y) {
  auto input = getInput(x, y);
  auto v = net.feed_forward(input);
  return v(0, 0);
}
void train() {
#ifdef USE_OMP
#pragma omp for
#endif
  for (int x = 0; x < h; ++x) {
    for (int y = 0; y < W; ++y) {
      auto input = getInput(x, y);
      VvalT expect{picture(x, y)};
      net.backpropagation(input, expect, 10e-5);
    }
  }
}
void initnet() {
  cout << "init" << endl;
  ifstream fin("picture.net");
  if (fin) {
    net.load(fin);
    for (layer &l : net.layers) {
      l.Func() = funcof(sigma);
      l.deFunc() = defuncof(sigma);
    }
    if (!fin)
      cerr << "not good file: picture.net" << endl;
  } else {
    net = network({15, 16, 64, 1}, pair_sigma);
    /*randomize net*/
#ifdef USE_OMP
#pragma omp for
#endif
    for (layer &l : net.layers) {
      for (auto &x : l.w.m) { // rand matrix
        for (auto &x : x) {
          x = 1 * (genvalT() - 0.5) * 2 * (1 / sqrt(l.w.getm()));
        }
      }
      for (auto &x : l.b.m) { // rand basis
        for (auto &x : x) {
          x = 1 * (genvalT() - 0.5) * 2 * (1 / sqrt(l.b.getn()));
        }
      }
    }
  }
}

double result[512][512];
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
      XCreateWindow(d, root, 0, 0, w, h, 0, visinfo.depth, InputOutput,
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

    XImage *xWindowBuffer =
        XCreateImage(d, visinfo.visual, visinfo.depth, ZPixmap, 0, mem, w, h,
                     pixelBits, w * 4);
    int pitch = w * pixelBytes;
    initnet();
    while (!stop) {
      cout.flush();
      for (int x = 0; x < h; ++x) {
        for (int y = 0; y < W; ++y) {
          result[x][y] = getgen(x, y);
        }
      }
      for (int x = 0; x < h; ++x) {
        for (int y = 0; y < W; ++y) {
          unsigned int *px = (unsigned int *)(mem + x * pitch + y * pixelBytes);
          int p = result[x][y] * 255;
          *px = ((p << 16) | (p << 8) | p);
        }
        for (int y = 0; y < W; ++y) {
          unsigned int *px =
              (unsigned int *)(mem + x * pitch + (y + W) * pixelBytes);
          int p = picture(x, y) * 255;
          *px = ((p << 16) | (p << 8) | p);
        }
      }
      train();
      valT delta = 0;
      for (int x = 0; x <= h; ++x) {
        for (int y = 0; y <= W; ++y) {
          delta +=
              (picture(x, y) - result[x][y]) * (picture(x, y) - result[x][y]);
        }
      }
      cout << ++t << " : delta: " << delta << "  \r";
      usleep(1);
      XPutImage(d, window, pen, xWindowBuffer, 0, 0, 0, 0, w, h);
      XFlush(d);
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
