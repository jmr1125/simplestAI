#include "layer.h"
#include "main.h"
#include "matrix.h"
#include "network.h"
#include "train.h"
#include <algorithm>
#include <ctime>
#include <curses.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ncurses.h>
#include <string>
// #include <unistd.h>
#include <utility>
#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_OMP
#define clock() omp_get_wtime()
#else
#define clock() ((double)clock() / CLOCKS_PER_SEC)
#endif

#define getcharcond(ch, cond)                                                  \
  {                                                                            \
    char cccccccccc;                                                           \
    while ((cccccccccc = getch()) == ERR && cond)                              \
      ;                                                                        \
    ch = cccccccccc;                                                           \
  }
#define getchartime(ch, time)                                                  \
  {                                                                            \
    clock_t t = clock();                                                       \
    getcharcond(ch, (clock() - t) <= time);                                    \
  }
#define getchardelay(ch) getcharcond(ch, 1);

using namespace std;
void save(const network &net) {
  ofstream out("handwrite.net");
  net.save(out);
}
int main(int argc, char *argv[]) { // blr?blur?...k
#ifdef USE_OMP
  omp_set_max_active_levels(3);
#endif
  if (argc == 1) {
    cout << "usage: " << argv[0] << " <1> <2> ...<9>" << endl;
    cout << "1 2 .. 9 are files contain data" << endl;
    return 1;
  }
  network net({}, NULL, NULL);
  {
    ifstream in("handwrite.net");
    funcT func = [](valT v) { return max((valT)0, v); };
    funcT defunc = [](valT v) -> valT { return (v >= 0 ? 1 : 0); };
    if (in) {
      net.load(in);
      for (layer &x : net.layers) {
        x.Funct = func;
        x.deFunct = defunc;
      }
      cout << "read from file" << endl;
    } else {
      net = network({256, 16, 16, 10}, func, defunc);
      /*randomize net*/
      for (layer &x : net.layers) {
        for (auto &x : x.w.m) { // rand matrix
          for (auto &x : x) {
            x = (genvalT() - 0.5) * 5;
          }
        }
        for (auto &x : x.b.m) { // rand basis
          for (auto &x : x) {
            x = (genvalT() - 0.5) * 5;
          }
        }
      }
    }
  }
  vector<pair<VvalT, int>> data;
  for (int i = 1; i < argc; ++i) {
    ifstream ist(argv[i]);
    int count = 0;
    string pic;
    while (true) {
      string s;
      if (!(ist >> s)) {
        break;
      }
      pic += s;
      count = (count + 1) % 16;
      if (count == 0) {
        VvalT tmp;
        assert(pic.size() == 256);
        tmp.resize(256);
        for (int i = 0; i < 256; ++i) {
          tmp[i] = pic[i] - '0';
        }
        data.push_back(make_pair(tmp, stoi(argv[i])));
        pic = "";
      }
    }
  }
  auto start = clock();
  cout << "starting..." << start << endl;
  initscr();
  intrflush(stdscr, TRUE), keypad(stdscr, TRUE);
  start_color();
  init_pair(1, COLOR_RED, COLOR_BLACK);
  init_pair(2, COLOR_CYAN, COLOR_BLACK);
  init_pair(3, COLOR_GREEN, COLOR_BLACK);
  nodelay(stdscr, TRUE);
  int t = 0;
  valT scale = 100;
  int batch_size = 1;
  int batch_size1 = 1;
  for (int epoch = 0; [&t]() {
         string s{};
         mvprintw(0, 0, "again?(Y/n/r)");
         clrtoeol();
         if (t) {
           printw(" y");
           clrtoeol();
           t = (t + 1) % 10;
           mvprintw(4, 0, "  r:%d", t);
           clrtoeol();
           return true;
         }
         int c;
         getchartime(c, 5);
         move(4, 0);
         clrtoeol();
         s = " ";
         s[0] = c;
         if (s == "n") {
           return false;
         } else if (s == "r") {
           t = 1;
           return true;
         } else {
           return true;
         }
       }();
       ++epoch) {
    //====EPOCH.FOR.LOOP====
    mvprintw(0, 0, "epoch: %d\n", epoch);
    batch_size = batch_size1;
    clrtoeol();
    {
      random_device rd;
      shuffle(data.begin(), data.end(), rd);
    }
    for (size_t i = 0; i < data.size(); i += batch_size) {
      //====DATA====
      vector<VvalT> pics;
      vector<VvalT> vexpects;
      vexpects.reserve(batch_size);
      pics.reserve(batch_size);
      for (int off = 0; off < batch_size && batch_size - 1 + i < data.size();
           ++off) {
        int x = off * 4;
        auto [pic, expect] = data.at(i + off);
        VvalT vexpect(10, 1);
        vexpect[expect] = 2;
        vexpects.push_back(vexpect);
        pics.push_back(pic);
        valT delta = 0;
        net.setInput(pic);
        net.getV();
        assert(net.output.size() == vexpect.size() && vexpect.size() == 10);
        for (int i = 0; i < 10; ++i) {
          delta += (net.output[i] - vexpect[i]) * (net.output[i] - vexpect[i]);
        }
        mvprintw(x + 1, 0, "%d/%d : delta:", i + off, data.size());
        clrtoeol();
        attr_t att;
        if (delta > 1) {
          att = COLOR_PAIR(2);
        } else if (fabs(delta - 0.8) == 0.1) {
          att = COLOR_PAIR(1) | A_BOLD;
        } else {
          att = COLOR_PAIR(3);
        }
        attron(att);
        printw("%8f", delta);
        attroff(att);
        printw(" time:%9f\n", clock() - start);
        {
          const auto &output = net.output;
          mvprintw(x + 2, 0, "output: ");
          for (const auto &x : output) {
            printw("%10f ", x);
          }
          mvprintw(x + 3, 0, "expect: ");
          for (const auto &x : vexpect) {
            if (x == 2) {
              attron(A_BOLD);
            }
            printw("%10f ", x);
            if (x == 2) {
              attroff(A_BOLD);
            }
          }
          // for (int i = 0; i < 16; ++i) {
          //   for (int j = 0; j < 16; ++j) {
          //     mvaddch(i + 10, 80 + j, (pic[i * 16 + j] == 1) ? ('1') :
          //     ('0'));
          //   }
          // }
          wrefresh(stdscr);
          // sleep(1);
        }
      }
      clrtoeol();

      wrefresh(stdscr);
      //==SETTING==
      move(0, 20);
      clrtoeol();
      int c = getch();
      if (c == 's') {
        for (; c != 'q';) {
          mvprintw(0, 20, "Setting(q/s/m/S)");
          clrtoeol();
          move(0, 40);
          getchardelay(c);
          if (c == 's') {
            save(net);
          } else if (c == 'm') {
            for (; c != 'q';) {
              mvprintw(0, 20, "Setting minibatch(q/j/k) ");
              getchardelay(c);
              if (c == 'j') {
                ++batch_size1;
              } else if (c == 'k') {
                --batch_size1;
              }
              batch_size1 = max(1, batch_size1);
              mvprintw(0, 60, "batch size: %d", batch_size1);
              clrtoeol();
            }
            while (data.size() % batch_size1) {
              --batch_size1;
            }
            c = 0;
          } else if (c == 'S') {
            for (; c != 'q';) {
              mvprintw(0, 20, "Setting scale(q/j/J/k/K)");
              getchardelay(c);
              if (c == 'j') {
                scale += 1;
              } else if (c == 'J') {
                scale += 100;
              } else if (c == 'k') {
                scale -= 1;
              } else if (c == 'K') {
                scale -= 100;
              }
              scale = max((valT)1, scale);
              mvprintw(0, 60, "scale: 1/%f", scale);
              clrtoeol();
            }
            c = 0;
          }
        }
      }
      trainn(net, pics, vexpects, (valT)1 / scale);
    }
  }
  save(net);
  endwin();
}
