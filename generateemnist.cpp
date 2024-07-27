#include "NN.hpp"
#include "bias_layer.hpp"
#include "convolution_layer.hpp"
#include "func_layer.hpp"
#include "layers.hpp"
#include "matrix_layer.hpp"
#include <curses.h>
#include <fstream>
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
  nnet net;
  {
    std::ifstream fin("emnist.net");
    net.add_layer(new convolution_layer);
    net.last_layer()->load(fin);
    net.add_layer(new bias_layer);
    net.last_layer()->load(fin);
    net.add_layer(new func_layer);
    net.last_layer()->load(fin);
    net.add_layer(new matrix_layer);
    net.last_layer()->load(fin);
    net.add_layer(new bias_layer);
    net.last_layer()->load(fin);
    net.add_layer(new func_layer);
    net.last_layer()->load(fin);
    net.add_layer(new matrix_layer);
    net.last_layer()->load(fin);
    net.add_layer(new bias_layer);
    net.last_layer()->load(fin);
    net.add_layer(new func_layer);
    net.last_layer()->load(fin);
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
      for (int i = 0; i < out.size(); ++i) {
        if (out[i] > max) {
          max = out[i], maxid = i;
        }
      }
      mvprintw(2, 32, "it is %s", name[maxid].c_str());
    }
  }
  printf("\033[?1003l\n");
  endwin();
}
