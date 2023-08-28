#include "function.h"
#include "network.h"
#include <algorithm>
#include <cmath>
#include <curses.h>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <ncurses.h>
using std::ifstream;
using std::max;
using std::min;
const int maxx = 28, maxy = 28;
char pic[maxx][maxy];
int main(int argc, char *argv[]) {
  network net({}, NULL, NULL);
  {
    ifstream in("handwriteemnist.net");
    if (!in) {
      std::cerr << "handwritemnist.net NOT found\nrun trainemnist first"
                << std::endl;
      return 1;
    }
    net.load(in);
    funcT func = funcof(ReLU);
    funcT defunc = defuncof(ReLU);
    for (layer &x : net.layers) {
      x.Funct = func;
      x.deFunct = defunc;
    }
  }
  initscr(), noecho();
  intrflush(stdscr, FALSE), keypad(stdscr, TRUE);
  mousemask(ALL_MOUSE_EVENTS | REPORT_MOUSE_POSITION, NULL);
  printf("\033[?1003h\n");
  mouseinterval(1);
  mvprintw(0, 0, "train %s", argv[1]);
  mouseinterval(0);
  bool quit = false;
  while (!quit) {
    bool mousestatus = false;
    bool mousecolor = true;
    while (true) {
      int ch = 0;
      mvprintw(0, 1, "ch=%d status=%d color=%d", ch, mousestatus, mousecolor);
      clrtoeol();
      mvprintw(1, 1, "press: 0x%08lx release: 0x%08lx", BUTTON1_PRESSED,
               BUTTON1_RELEASED);
      ch = getch();
      {
        matrix input;
        input.setn(maxx * maxy);
        input.setm(1);
        for (int i = 0; i < maxx; ++i) {
          for (int j = 0; j < maxy; ++j) {
            input(i * maxy + j, 0) = pic[i][j];
          }
        }
        auto o = net.feed_forward(input);
        move(3, 32);
        valT max = std::numeric_limits<valT>::min();
        int maxid = 0;
        for (int i = 0; i < 47; ++i) {
          printw("%f ", o(i, 0));
          if (o(i, 0) > max) {
            max = o(i, 0);
            maxid = i;
          }
        }
        // mvprintw(5, 32, "it is %d %f", maxid, max);
        mvprintw(5, 32, "it is ");
        attron(A_BOLD);
        printw("%d ", maxid);
        attroff(A_BOLD);
        printw("%f", max);
      }
      if (ch == 'c') {
        mousecolor = !mousecolor;
      } else if (ch == '\n') {
        break;
      } else if (ch == 'q') {
        quit = true;
        break;
      } else if (KEY_MOUSE == ch) {
        MEVENT evt;
        if (getmouse(&evt) == OK) {
          mvprintw(2, 1, "0x%08x %d %d", evt.bstate, evt.x, evt.y);
          clrtoeol();
          int x = min(max(0, evt.x - 3), maxx - 1),
              y = min(max(0, evt.y - 3), maxy - 1);
          if (evt.bstate == 0x080000) {
            mousestatus = true;
          }
          if (evt.bstate == 0x040000) {
            mousestatus = false;
          }
          if (evt.bstate == BUTTON3_PRESSED) {
            mvprintw(2, 20, "BUTTON3_PRESSED");
            clrtoeol();
          }
          if (evt.bstate == BUTTON3_RELEASED || evt.bstate == BUTTON3_CLICKED) {
            mvprintw(2, 20, "BUTTON3_RELEASED");
            clrtoeol();
            // mousestatus = false;
          }
          if (evt.bstate == BUTTON1_DOUBLE_CLICKED) {
            break;
          }
          if (mousestatus) {
            bool b[3][3] = {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}};
            for (int i = 0; i < 3; ++i) {
              for (int j = 0; j < 3; ++j) {
                int xx = x + i;
                int yy = y + j;
                if (b[i][j])
                  pic[min(max(0, yy), maxy - 1)][min(max(0, xx), maxx - 1)] =
                      mousecolor;
              }
            }
            // pic[y][x] = mousecolor;
          }
        }
      }
      for (int i = 0; i < maxx; ++i) {
        for (int j = 0; j < maxy; ++j) {
          if (pic[i][j]) {
            mvaddch(i + 3, j + 3, '#' | A_REVERSE);
          } else {
            mvaddch(i + 3, j + 3, '.' | A_NORMAL);
          }
        }
      }
    }

    bool empty = true;
    for (int i = 0; i < maxx; ++i) {
      for (int j = 0; j < maxy; ++j) {
        if (pic[i][j] == true) {
          empty = false;
          break;
        }
      }
      if (!empty) {
        break;
      }
    }
    if (!empty) {
    } else {
      mvprintw(2, 20, "pic is empty");
    }
    memset(pic, 0, sizeof pic);
  }
  printf("\033[?1003l");
  endwin();
}
