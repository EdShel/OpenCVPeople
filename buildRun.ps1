./build.ps1
cd ./bin
try {
    ./main.exe $args[0]
}
finally {
    cd ..
}