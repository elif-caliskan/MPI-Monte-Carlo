/*
Student Name: Elif Çalışkan
Student Number: 2016400183
Compile Status: Compiling
Program Status: Working
Notes: -
*/

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <mpi.h>
#include <string>
using namespace std;
//it returns the maximum of a and b
int max(int a,int b){
    if(a>b)
        return a;
    else
        return b;
}
//it returns the minimum of a and b
int min(int a,int b){
    if(a<b)
        return a;
    else
        return b;
}
//it takes four arguments arg[1] is input's path, arg[2] is output's path, arg[3] is beta arg[4] is pi
int main(int argc, char** argv) {
    string inputFile=argv[1];
    string outputFile=argv[2];
    double beta = stod(argv[3]);
    double pi= stod(argv[4]);
    //MPI is initialized
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double gamma = 0.5*log((1-pi)/pi);
    //iteration count
    int T=500000;
    int N=200;
    srandom((unsigned)time(NULL));
    //z matrix is the noisy matrix and it is distributed among processors
    int z[N][N];
    //slaveNum is the number of processors that will work on the picture
    int slaveNum=world_size-1;
    //if it is master processor, the input file gets read and the array is distributed among the processors
    //then the data is received and written to output file
    if(world_rank==0){
        FILE *myFile;
        //I assumed the inputFile exists
        myFile = fopen(inputFile.c_str(), "r");
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                int k=0;
                fscanf(myFile, "%2d", &k);
                z[i][j]=k;
            }
        }
        //this for loop sends arrays to corresponding processors
        //since MPI_Send is blocking, it waits the data to be received
        for(int i = 1 ; i <=slaveNum ; i++){
            MPI_Send(z[(N/slaveNum)*(i-1)], (N/slaveNum)*N, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        //this for loop receives the updated arrays and changes z[N][N] accordingly
        for(int i = 1 ; i <= slaveNum ; i++){
            int* subarr = NULL;
            subarr = new int[(N/slaveNum)*N];
            MPI_Recv(subarr, (N/slaveNum)*N, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int k=0;k<(N/slaveNum)*N;k++){
                z[(N/slaveNum)*(i-1)+k/N][k%N]=subarr[k];
            }
        }
        //it writes z array into output file
        freopen(outputFile.c_str(), "w",stdout);
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                printf("%d ", z[i][j]);
            }
            printf("\n");
        }
        fclose(stdout);
    }
    else if(world_rank==1){
        //subZ is received from master and subX is copied from subZ
        //subX is the original array and it is used for finding sum in delta_E
        int* subZ = NULL;
        subZ = new int[(N/slaveNum)*N];
        int* subX = NULL;
        subX = new int[(N/slaveNum)*N];
        MPI_Recv(subZ, (N/slaveNum)*N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int k=0;k<(N/slaveNum)*N;k++) {
            subX[k] = subZ[k];
        }
        //in every iteration the first row of the next processor's array is received and the last row is sent
        //since it is the first processor there is no previous slave processor
        for(int t=0;t<T/slaveNum;t++){
            //newLineBottom is the first row of the next processor's array
            int* newLineBottom = NULL;
            newLineBottom = new int[N];
            MPI_Recv(newLineBottom, N, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //it sends its last row to second processor
            MPI_Send(&subZ[(N/slaveNum-1)*N], N, MPI_INT, 2, 0, MPI_COMM_WORLD);
            //a random pixel is picked
            int i=rand() % ((N/slaveNum)*N-1);
            int sum=0;
            //sum is calculated this also has its own value
            for(int a=max(i/N-1,0);a<=min(i/N+1,N/slaveNum-1);a++){
                for(int b=max(i%N-1,0);b<=min(i%N+1,N-1);b++){
                    sum+=subZ[a*N+b];
                }
            }
            //this is the communication part
            //if the picked pixel is on the last row of processor, it should add the corresponding values in newLineBottom
            if(i/N==N/slaveNum-1){
                for(int b=max(i%N-1,0);b<=min(i%N+1,N-1);b++){
                    sum+=newLineBottom[b];
                }
            }
            //its value is subtracted this only has its neighbors' values
            sum-=subZ[i];
            //delta_E is computed according to the formula
            double delta_E = -2*gamma*subX[i]*subZ[i] -2*beta*subZ[i]*sum;
            double random= rand() / (double)RAND_MAX ;
            //if the probability is less than acceptance probability, a flip occurs
            if(log(random)<delta_E){
                subZ[i] = -subZ[i];
            }
            //pointers are freed
            delete newLineBottom;
            newLineBottom=NULL;
        }
        //the updated subZ is sent to master
        MPI_Send(subZ, (N/slaveNum)*N, MPI_INT, 0, 0, MPI_COMM_WORLD);
        delete subZ;
        subZ=NULL;
        delete subX;
        subX=NULL;
    }
    //if this is the last processor there is no next processor
    else if(world_rank==world_size-1){
        int* subZ = NULL;
        subZ = new int[(N/slaveNum)*N];
        int* subX = NULL;
        subX = new int[(N/slaveNum)*N];
        MPI_Recv(subZ, (N/slaveNum)*N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int k=0;k<(N/slaveNum)*N;k++) {
            subX[k] = subZ[k];
        }
        for(int t=0;t<T/slaveNum;t++){
            //it sends its first row to the previous processor
            MPI_Send(&subZ[0], N, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD);
            int* newLineTop = NULL;
            newLineTop = new int[N];
            //it receives the last row of previous processor's array
            MPI_Recv(newLineTop, N, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int i=rand() % ((N/slaveNum)*N-1);
            int sum=0;
            for(int a=max(i/N-1,0);a<=min(i/N+1,N/slaveNum-1);a++){
                for(int b=max(i%N-1,0);b<=min(i%N+1,N-1);b++){
                    sum+=subZ[a*N+b];
                }
            }
            //if the pixel is at the first row, newLineTop is used for sum
            if(i/N==0){
                for(int b=max(i%N-1,0);b<=min(i%N+1,N-1);b++){
                    sum+=newLineTop[b];
                }
            }
            sum-=subZ[i];
            double delta_E = -2*gamma*subX[i]*subZ[i] -2*beta*subZ[i]*sum;
            double random= rand() / (double)RAND_MAX ;
            if(log(random)<delta_E){
                subZ[i] = -subZ[i];
            }
            delete newLineTop;
            newLineTop=NULL;

        }
        MPI_Send(subZ, (N/slaveNum)*N, MPI_INT, 0, 0, MPI_COMM_WORLD);
        delete subZ;
        subZ=NULL;
        delete subX;
        subX=NULL;
    }
    //if this process between 1 and world_size-1, it needs to have communication in two ways
    else{
        int* subZ = NULL;
        subZ = new int[(N/slaveNum)*N];
        int* subX = NULL;
        subX = new int[(N/slaveNum)*N];
        MPI_Recv(subZ, (N/slaveNum)*N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int k=0;k<(N/slaveNum)*N;k++) {
            subX[k] = subZ[k];
        }
        for(int t=0;t<T/slaveNum;t++){
            //first it sends its first row to previous one
            MPI_Send(&subZ[0], N, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD);
            int* newLineTop = NULL;
            newLineTop = new int[N];
            //then it gets the last row from previous process
            MPI_Recv(newLineTop, N, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int* newLineBottom = NULL;
            newLineBottom = new int[N];
            //it gets the first row of the next process
            MPI_Recv(newLineBottom, N, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //it sends the last row to the next process
            MPI_Send(&subZ[(N/slaveNum-1)*N], N, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD);
            int i=rand() % ((N/slaveNum)*N-1);
            int sum=0;

            for(int a=max(i/N-1,0);a<=min(i/N+1,N/slaveNum-1);a++){
                for(int b=max(i%N-1,0);b<=min(i%N+1,N-1);b++){
                    sum+=subZ[a*N+b];
                }
            }
            //if the pixel is at the last row, newLineBottom is used for sum
            if(i/N==N/slaveNum-1){
                for(int b=max(i%N-1,0);b<=min(i%N+1,N-1);b++){
                    sum+=newLineBottom[b];
                }
            }
            //if the pixel is at the first row, newLineTop is used for sum
            if(i/N==0){
                for(int b=max(i%N-1,0);b<=min(i%N+1,N-1);b++){
                    sum+=newLineTop[b];
                }
            }
            sum-=subZ[i];
            double delta_E = -2*gamma*subX[i]*subZ[i] -2*beta*subZ[i]*sum;
            double random= rand() / (double)RAND_MAX ;
            if(log(random)<delta_E){
                subZ[i] = -subZ[i];
            }
            delete newLineTop;
            newLineTop=NULL;
            delete newLineBottom;
            newLineBottom=NULL;

        }
        MPI_Send(subZ, (N/slaveNum)*N, MPI_INT, 0, 0, MPI_COMM_WORLD);
        delete subZ;
        subZ=NULL;
        delete subX;
        subX=NULL;
    }
    MPI_Finalize();
    return 0;
}
