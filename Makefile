# 'make'	build executable file
# 'make clean'	removes all *.o and executalbe file

# define the C compiler
NVCC	= gcc

# define any compile-time flags
CUFLAGS = -fopenmp -O3

# define openmp flags
OPENMP  = -fopenmp
CUOPENMP  = -Xcompiler -fopenmp

# define the direction containing header file
INCLUDES= -I/usr/include -I include -I /fred/oz016/opt-pipe/include 

# define the library path
LFLAGS	= -L/fred/oz016/opt-pipe/lib 

# define any libraries to link to executable
LIBS	= -lm -lgsl -lgslcblas -llal -llalsimulation -lfftw3

# define the C object files
OBJS	= coherent.o generate_signal.o filter.o statistic.o

#define the directory for object
OBJSDIR = object

# define the executable file
MAIN	= exe_170817

all: $(MAIN)

$(MAIN): $(OBJS)
	$(NVCC) $(CUFLAGS) -o $(MAIN) $(OBJS) $(LIBS) $(LFLAGS) $(INCLUDES) 

%.o: %.c
	$(NVCC) $(CUFLAGS) $(INCLUDES) -c $^


# clean the executable file and object files
clean:
	$(RM) $(OBJS) $(MAIN)
