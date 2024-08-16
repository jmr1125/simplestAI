#include "NN.hpp"
#include "average_layer.hpp"
#include "bias_layer.hpp"
#include "convolution_layer.hpp"
#include "func_layer.hpp"
#include "layers.hpp"
#include "matrix.hpp"
#include "matrix_layer.hpp"
#include "max_layer.hpp"
#include "ocl.hpp"
#include <curses.h>
#include <fstream>
#include <iostream>
#include <ncurses.h>
#include <vector>
using namespace std;
const int maxx = 28, maxy = 28;
std::vector<valT> pic(maxx *maxy);
string name[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B",
                 "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
                 "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                 "a", "b", "d", "e", "f", "g", "h", "n", "q", "R", "T"};
int main() {
#ifdef USE_OCL
  init();
#endif
  nnet net;
  {
    std::ifstream fin("emnist.net");
#define netin_load(type)                                                       \
  if (fin) {                                                                   \
    net.add_layer(make_shared<type>());                                        \
    net.last_layer()->load(fin);                                               \
  } else {                                                                     \
    cerr << "error loading emnist.net" << endl;                                \
    return 1;                                                                  \
  }
    netin_load(convolution_layer);
    netin_load(bias_layer);
    netin_load(func_layer);
    netin_load(max_layer);
    netin_load(convolution_layer);
    netin_load(bias_layer);
    netin_load(func_layer);
    netin_load(max_layer);
    netin_load(matrix_layer);
    netin_load(bias_layer);
    netin_load(func_layer);
    netin_load(matrix_layer);
    netin_load(bias_layer);
    netin_load(func_layer);
    netin_load(matrix_layer);
    netin_load(bias_layer);
    netin_load(func_layer);
  }
  initscr();
  cbreak();
  noecho();
  keypad(stdscr, TRUE);
  mousemask(ALL_MOUSE_EVENTS | REPORT_MOUSE_POSITION, NULL);
  printf("\033[?1003h\n");
  bool quit = false;
  bool color = 0;
  while (!quit) {
    int ch = wgetch(stdscr);
    mvaddch(1, 1, ch);
    if (ch == KEY_MOUSE) {
      MEVENT evt;
      if (getmouse(&evt) == OK) {
        mvprintw(2, 1, "0x%08x %d %d", evt.bstate, evt.x, evt.y);
        if (evt.bstate == 0x2) {
          color = 1;
        }
        if (evt.bstate == 0x1) {
          color = 0;
        }
        int x = (evt.x - 3) / 2, y = evt.y - 3;
        x = min(x, maxx - 1);
        x = max(0, x);
        y = min(y, maxy - 1);
        y = max(0, y);
        auto draw = [&](int x, int y) {
          if (x < 0)
            return;
          if (y < 0)
            return;
          if (x >= maxx)
            return;
          if (y >= maxy)
            return;
          pic[x + maxx * y] = color;
        };
        if (color) {
          draw(x - 1, y);
          draw(x, y - 1);
          draw(x, y);
          draw(x + 1, y);
          draw(x, y + 1);
        }
      }
    }
    if (ch == 'q')
      break;
    if (ch == 'c') {
      color = !color;
    }
    if (ch == 'C') {
      for (auto &x : pic)
        x = 0;
    }
    for (int i = 0; i < maxx * maxy; ++i) {
      mvaddch(i / maxx + 3, (i % maxx) * 2 + 3, pic[i] ? 'M' : '.');
    }
    {
      auto out = net.forward(pic);
      int maxid;
      valT max = -1;
      for (int x = 0; x < 5; ++x) {
        move(3 + x, 60);
        for (int i = 10 * x; i < min((size_t)10 * x + 10, out.size()); ++i) {
          printw("%.3f ", out[i]);
        }
      }
      for (int i = 0; i < out.size(); ++i) {
        if (out[i] > max) {
          max = out[i], maxid = i;
        }
      }
      mvprintw(2, 32, "it is %s", name[maxid].c_str());
    }
  }
#ifdef USE_OCL
  teardown();
#endif
  printf("\033[?1003l\n");
  endwin();
}
