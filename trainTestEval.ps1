cd ./bin
try {
    ./main.exe train
    ./main.exe test
    ./main.exe eval
} finally {
    cd ..
}