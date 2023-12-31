#include <algorithm>
#include <cmath>
#include <curses.h>
#include <fstream>
#include <ios>
#include <iostream>
#include <ncurses.h>
using std::max;
using std::min;
using std::ofstream;
char pic[16][16];
int main(int argc, char *argv[]) {
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
          int x = min(max(0, evt.x - 3), 15), y = min(max(0, evt.y - 3), 15);
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
            pic[y][x] = mousecolor;
          }
        }
      }
      for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
          if (pic[i][j]) {
            mvaddch(i + 3, j + 3, '#' | A_REVERSE);
          } else {
            mvaddch(i + 3, j + 3, '.' | A_NORMAL);
          }
        }
      }
    }

    bool empty = true;
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
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
      ofstream ost(argv[1], std::ios_base::app);
      for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
          ost << (bool)pic[i][j];
        }
        ost << std::endl;
      }
      ost.close();
    } else {
      mvprintw(2, 20, "pic is empty");
    }
    memset(pic, 0, sizeof pic);
  }
  printf("\033[?1003l");
  endwin();
}
