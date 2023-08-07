在```olddata```中有16*16的数据，在```1234567890```中有数据，使用```train 0 1 2 3 4 5 6 7 8 9```训练\
此处不提供CMake编译
```clang++ generate.cpp -ogenerate ../layer.cpp ../network.cpp ../matrix.cpp ../function.cpp -I.. -lncurses``` 
