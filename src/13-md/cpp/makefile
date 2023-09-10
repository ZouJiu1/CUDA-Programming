all: ljmd   # 全部需要编译的部分，ljmd，转到ljmd指令执行的

CC = nvcc   #编译器
CFLAGS = -O3 -arch=sm_70 #编译的选项

#  编译出来的目标档案ljmd 以及 对应的source档案 ***，没有转到对应的指定执行，产生*.o档案
ljmd: initialize.o integrate.o neighbor.o force.o memory.o main.O
    #  编译的方式
	$(CC) -o ljmd \
	initialize.o integrate.o neighbor.o force.o memory.o main.o

#  编译出来的目标档案 initialize.o 以及 对应的source档案
initialize.o: initialize.cu
	$(CC) $(CFLAGS) -c initialize.cu   # 编译的方式, -c是产生*.o档案，不做link
integrate.o: integrate.cu
	$(CC) $(CFLAGS) -c integrate.cu
memory.o: memory.cu
	$(CC) $(CFLAGS) -c memory.cu
neighbor.o: neighbor.cu
	$(CC) $(CFLAGS) -c neighbor.cu
force.o: force.cu
	$(CC) $(CFLAGS) -c force.cu
main.o: main.cu
	$(CC) $(CFLAGS) -c main.cu

#  clean时删除相应的*.o档案以及编译的ljmd
clean:
	rm *o ljmd

