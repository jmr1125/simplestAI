#include <assert.h>
#include <cstdio>
#include <stdio.h>
#include <string>
#include <vector>
#define freadint_t(x, fp, f)                                                   \
  {                                                                            \
    fread(&x, 1, sizeof x, fp);                                                \
    x = f(x);                                                                  \
  }
#define freadint32_t(x, fp) freadint_t(x, fp, reverse32);
#define freadint8_t(x, fp) freadint_t(x, fp, reverse8);
using namespace std;
int32_t reverse32(int32_t x) {
  const int16_t x1 = (x & 0x000000FF) >> 0;
  const int16_t x2 = (x & 0x0000FF00) >> 8;
  const int16_t x3 = (x & 0x00FF0000) >> 16;
  const int16_t x4 = (x & 0xFF000000) >> 24;
  return x1 << 24 | x2 << 16 | x3 << 8 | x4;
}
int32_t reverse8(int8_t x) {
  const int16_t x1 = (x & 0x00FF) >> 0;
  const int16_t x2 = (x & 0xFF00) >> 8;
  return x1 << 8 | x2;
}
void process_label(FILE *fp, vector<int8_t> &v) {
  int32_t num;
  fread(&num, 1, sizeof num, fp);
  num = reverse32(num);
  printf("-- %d labels\n", num);
  v.reserve(num);
  for (int i = 0; i < num; ++i) {
    int8_t x;
    fread(&x, 1, sizeof x, fp);
    assert(0 <= x && x <= 9);
    v.push_back(x);
  }
}
void process_image(FILE *fp, vector<vector<vector<int8_t>>> &v) {
  int32_t num;
  int32_t rows, cols;
  freadint32_t(num, fp);
  printf("-- %d images\n", num);
  freadint32_t(rows, fp);
  freadint32_t(cols, fp);
  v = vector(num, vector(cols, vector(rows, (int8_t)0)));
  printf("-- %d x %d\n", rows, cols);
  for (int n = 0; n < num; ++n) {
    for (int i = 0; i < cols; ++i) {
      for (int j = 0; j < rows; ++j) {
        int8_t x;
        freadint8_t(x, fp);
        v[n][i][j] = x;
      }
    }
  }
}
void process_file(string filename, vector<vector<vector<int8_t>>> &pics,
                  vector<int8_t> &labels) {
  int32_t magicnumber;
  FILE *fp = fopen(filename.c_str(), "rb");
  fread(&magicnumber, 1, sizeof magicnumber, fp);
  magicnumber = reverse32(magicnumber);
  printf("magicnumber: %x\n", magicnumber);
  switch (magicnumber) {
  case 0x00000801:
    printf("-- labels\n");
    process_label(fp, labels);
    break;
  case 0x00000803:
    printf("-- image\n");
    process_image(fp, pics);
    break;
  }
  fclose(fp);
};
vector<vector<vector<int8_t>>> pics;
vector<int8_t> labels;
int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <file-image> <file-label>\n", argv[0]);
    return 1;
  }
  process_file(argv[1], pics, labels);
  process_file(argv[2], pics, labels);
  assert(pics.size() == labels.size());
  const int n = pics.size();
  {
    FILE *fp = fopen("traindata.txt", "w");
    for (int i = 0; i < n; ++i) {
      for (const auto X : pics[i]) {
        for (const auto x : X) {
          if (((int)x + 256) % 256 > 128) {
            fprintf(fp, "1");
          } else {
            fprintf(fp, "0");
          }
        }
        fprintf(fp, "\n");
      }
      fprintf(fp, "%d ", ((int)labels[i] + 256) % 256);
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
}
