#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#define clock (clock() / CLOCKS_PER_SEC)
using namespace std;
int main() {
  long long i = 1;
  ios::sync_with_stdio(true);
  // cout<<"thread count:
  // "<<omp_get_num_threads()<<"/"<<omp_get_max_threads()<<endl;
  cout << "thread count: " << omp_get_num_threads() << "/"
       << omp_get_max_threads() << endl;
#pragma omp parallel
  { // cout << omp_get_thread_num() <<' '<< i << endl;
#pragma omp atomic
    i++;
    printf("thread count: %d / %d\n", omp_get_num_threads(),
           omp_get_max_threads());
    printf("%d -> %lld \n", omp_get_thread_num(), i);
  }
  cout << "done." << endl;
  cout << "sleep 1s" << endl;
  sleep(1);
  cout << "thread count: " << omp_get_num_threads() << "/"
       << omp_get_max_threads() << endl;
#pragma omp parallel for
  for (int x = 0; x <= 100; ++x) {
    printf("thread count: %d / %d\n", omp_get_num_threads(),
           omp_get_max_threads());
    printf("%d >> %d \n", omp_get_thread_num(), x);
  }
  cout << "done." << endl;
  cout << "sleep 1s and calculate Pi" << endl;
  // clock_t now=clock;
  // cout<<"now: "<<now<<endl;
  sleep(1);
  cout << "now: " << clock << endl;
  long steps = 10000000000;
  double x, pi, s;
  const double dx = (double)1.0 / steps;
#pragma omp parallel
  {
    double x;
#pragma omp for reduction(+ : s)
    for (i = 0; i <= steps; i++) {
      x = i * dx;
      s += (double)4.0 / (x * x + 1);
    }
  }
  pi = s * dx;
  cout << "pi: " << pi << endl
       << "now: " << clock << endl; //<<"time: "<<clock-now<<endl;
  // now = clock;
  pi = s = 0;
  {
    double x;
    for (i = 0; i <= steps; i++) {
      x = i * dx;
      s += (double)4.0 / (x * x + 1);
    }
  }
  pi = s * dx;
  cout << "pi: " << pi << endl
       << "now: " << clock << endl; // << "time: " << clock - now << endl;
}
