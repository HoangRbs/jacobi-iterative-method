// chạy với 2 câu lệnh: 
// mpicxx Main.cpp
// mpirun -np 3 ./a.out

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
 #include <time.h>

// #define N 9         
// #define M 9         // M chia het cho so cpu (-np 3)

// dung de test hieu nang
#define N 60         
#define M 300         // M chia het cho so cpu (-np 3)

#define tolerance 0.0000001

#define c0 0.25     // periodic boudary
#define cL 0.25

float **InitMatrix2d(int n, int m);

// giải phóng bộ nhớ 2 chiều
void freeMatrix2d(float **array);

// hiển thị ma trận
void DisplayMatrix(float **a, int n, int m);

// gửi cạnh trái phải cho các tiến trình
void sendTlTr(float **c, float *tl, float *tr, int processId, int numberProcess, MPI_Status status, int ms);

float* InitArray(int n);

int main(int argc, char *argv[])
{
    
    MPI_Init(&argc, &argv);                         // must have argument so the command from terminal can pass arguments into program
    
    int num_process, processId;
    MPI_Status status;

    MPI_Comm_size(MPI_COMM_WORLD, &num_process);    // get OUT num_processes (-np 3)
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);      // get OUT id_process   

    // if (M % num_process != 0) {                     // M nen chia het cho so process
    //     printf("ko chia het cho so process!!");
    //     exit;
    // }

    // -------------------------------------------------------

    float **conMatx = InitMatrix2d(N, M);   // concentration matrix
    float *sendptr, *recvptr;

    clock_t start, end;  // calculate time
    double cpu_time_used; // calculate time

    // if only one process
    if (num_process == 1) {
        start = clock(); // calculate time

        float **conMatx = InitMatrix2d(N, M);

        float global_maxConDiff, prevCon;      // max concentration difference
        float west, east, south, north;
        float C0, CL = 0.25;

        do {

            global_maxConDiff = 0;
            prevCon = 0;

            // loop all grid points 
            for(int i = 0; i < N; i++) {

                for (int j = 0; j < M; j++) {

                    prevCon = conMatx[i][j];
                    // nếu Cij là source
                    if(i == N - 1) {
                        conMatx[i][j] = 1;
                    } 
                    // nếu Cij là sink
                    else if (i == 0) {
                        conMatx[i][j] = 0;
                    } 
                    else {
                        west = (j == 0) ? C0 : conMatx[i][j - 1];
                        east = (j == M - 1) ? CL : conMatx[i][j + 1];
                        north = conMatx[i - 1][j];
                        south = conMatx[i + 1][j];

                        conMatx[i][j] = 0.25 * (west + east + south + north);
                    }
                    
                    // when <= tolerance 0.000000001, it means maxConDiff still == 0 --> maxConDiff < tolerance
                    if(fabs(conMatx[i][j] - prevCon) > tolerance) global_maxConDiff = fabs(conMatx[i][j] - prevCon);
                }
            }
            
        } while (global_maxConDiff > tolerance);  

        end = clock(); // calculate time
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; // calculate time

        if (M > 100) {
            // test hieu nang M = 300
            printf("cpu = 1 ... hoan thanh in %f s \n", cpu_time_used);
        } else {
            DisplayMatrix(conMatx, N, M);
        }

        MPI_Finalize();
        return 0;
    }

    int ms = M / num_process;
    float **conMatx_cpu = InitMatrix2d(N, ms);

    // define column type : a unit is a columns contains N squares.
    MPI_Datatype latticecol, latticecoltype, mscol, mscoltype;

    if(processId == 0)
    {
        sendptr = &(conMatx[0][0]);

        MPI_Type_vector(N, 1, M, MPI_FLOAT, &latticecol);
        MPI_Type_commit(&latticecol);
        MPI_Type_create_resized(latticecol, 0, 1 * sizeof(float), &latticecoltype);
        MPI_Type_commit(&latticecoltype);
    }
    else
    {
        sendptr = NULL;
    }

    MPI_Type_vector(N, 1, ms, MPI_FLOAT, &mscol);
    MPI_Type_commit(&mscol);
    MPI_Type_create_resized(mscol, 0, 1 * sizeof(float), &mscoltype);
    MPI_Type_commit(&mscoltype);

    MPI_Scatter(sendptr, ms, latticecoltype, &conMatx_cpu[0][0], ms, mscoltype, 0, MPI_COMM_WORLD);

    float *tl, *tr, **result;
    float local_maxConDiff, global_maxConDiff, prevCon;      // max concentration difference
    float west, east, south, north;

    start = clock(); // calculate time
    
    do {
        local_maxConDiff = 0;
        global_maxConDiff = 0;
        prevCon = 0;

        tl = InitArray(N);
        tr = InitArray(N);

        // exchange boundary strips with neighboring processors
        // trao đổi miền cho mỗi tiến trình (tl, tr)
        sendTlTr(conMatx_cpu,tl,tr,processId, num_process, status, ms);

        // loop all grid points 
        for(int i = 0; i < N; i++) {

            for (int j = 0; j < ms; j++) {

                prevCon = conMatx_cpu[i][j];
                // nếu Cij là source
                if(i == N - 1) {
                    conMatx_cpu[i][j] = 1;
                } 
                // nếu Cij là sink
                else if (i == 0) {
                    conMatx_cpu[i][j] = 0;
                } 
                else {
                    west = (j == 0) ? tl[i] : conMatx_cpu[i][j - 1];
                    east = (j == ms - 1) ? tr[i] : conMatx_cpu[i][j + 1];
                    north = conMatx_cpu[i - 1][j];
                    south = conMatx_cpu[i + 1][j];

                    conMatx_cpu[i][j] = 0.25 * (west + east + south + north);
                }
                
                // when <= tolerance, then maxConDiff == 0 --> maxConDiff < tolerance
                if(fabs(conMatx_cpu[i][j] - prevCon) > tolerance) local_maxConDiff = fabs(conMatx_cpu[i][j] - prevCon);
            }
        }

        free(tl);
        free(tr);

        MPI_Allreduce(&local_maxConDiff, &global_maxConDiff, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        
    } while (global_maxConDiff > tolerance);    

    end = clock(); // calculate time
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; // calculate time

    if(processId == 0)  {
        //result = InitMatrix2d(N,M);         // mảng 2 chiều để tiện thao tác
        recvptr = &(conMatx[0][0]);            // mảng 1 chiều thì chỉ đến mảng 2 chiều để dễ làm việc với mpi
    } else {
        recvptr = NULL;
    }

    if (M > 100) {
        // test hieu nang M = 300
        printf("cpu = 3 ... hoan thanh in %f s \n", cpu_time_used);
    } else {
        DisplayMatrix(conMatx_cpu, N, ms);
    }

    // MPI_Gather(&conMatx_cpu[0][0], N * ms, MPI_FLOAT, recvptr, M, latticecoltype, 0, MPI_COMM_WORLD);
    // MPI_Gather( const void* sendbuf , int sendcount , MPI_Datatype sendtype , void* recvbuf , int recvcount , MPI_Datatype recvtype , int root , MPI_Comm comm);

    if(processId == 0) {
        // for (int i = 0; i < N; i++) {
        //     for (int j =0; j < M; j++) {
        //         printf("%f ", recvptr[N * i + j]);
        //     }

        //     printf("\n");
        // }


        // DisplayMatrix(conMatx, N, M);
        // printf("\n");

        freeMatrix2d(conMatx);
        freeMatrix2d(result);
    }

    freeMatrix2d(conMatx_cpu);

    // --------------------------------------------------------
    MPI_Finalize();

    return 0;
}   

void sendTlTr(float **conMatx_cpu, float *tl, float *tr, int processId, int numberProcess, MPI_Status status, int ms) {

    float* sendTl = InitArray(N);
    float* sendTr = InitArray(N);

    // tl
    if(processId == 0) {
        for (int i = 0; i < N; i++) {
            // periodic boudary: tl array of process 0 is [c0, c0, c0]
            tl[i] = c0; 
            // send tl to next process
            sendTl[i] = conMatx_cpu[i][ms - 1];
        }

        // send tl to next process
        // send and recv tag must be equal: cpu 0 send with tag 0 then cpu 1 recv also with tag 0.
        MPI_Send(sendTl, N, MPI_FLOAT, processId + 1, processId, MPI_COMM_WORLD);
    }
    else if(processId == numberProcess - 1) {
        // recv tl data from prev process
        MPI_Recv(tl, N, MPI_FLOAT, processId - 1, processId - 1, MPI_COMM_WORLD, &status);
    }
    else {
        for (int i = 0; i < N; i++) {
            // send tl to next process
            sendTl[i] = conMatx_cpu[i][ms - 1];
        }

        // send tl to next process
        MPI_Send(sendTl, N, MPI_FLOAT, processId + 1, processId, MPI_COMM_WORLD);

        // recv tl data from prev process
        MPI_Recv(tl, N, MPI_FLOAT, processId - 1, processId - 1, MPI_COMM_WORLD, &status);
    }

    // tr
    if(processId == numberProcess - 1) {
        for (int i = 0; i < N; i++) {
            // periodic boudary: tr array of process 0 is [cL, cL, cL]
            tr[i] = cL; 
            // send tr to prev process
            sendTr[i] = conMatx_cpu[i][0];
        }

        // send tl to prev process
        MPI_Send(sendTr, N, MPI_FLOAT, processId - 1, processId, MPI_COMM_WORLD);
    }
    else if(processId == 0) {
        // recv tr data from next process
        MPI_Recv(tr, N, MPI_FLOAT, processId + 1, processId + 1, MPI_COMM_WORLD, &status);
    }
    else {
        for (int i = 0; i < N; i++) {
            // send tr to prev process
            sendTr[i] = conMatx_cpu[i][0];
        }

        // send tr to prev process
        MPI_Send(sendTr, N, MPI_FLOAT, processId - 1, processId, MPI_COMM_WORLD);
        // recv tr data from next process
        MPI_Recv(tr, N, MPI_FLOAT, processId + 1, processId + 1, MPI_COMM_WORLD, &status);
    }
}

void DisplayMatrix(float **a, int n, int m) {
    int i, j;
    printf("\n Matrix: \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            printf("%.2f ", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

float* InitArray(int n) {
    float* arr = new float[n];

    for (int i = 0 ; i < n; i++) {
        arr[i] = 0;
    }

    return arr;
}

float **InitMatrix2d(int n, int m)
{
    float *array1d = (float *) calloc(n*m,sizeof(float));
    float **array2d = (float **) calloc(n, sizeof(float*));

    for(int i = 0; i < n; i++) {
        array2d[i] = &(array1d[i * m]);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            array2d[i][j] = 0;      // concentration of all grid  equal = 0
        }
    }

    return array2d;
}

void freeMatrix2d(float **array)
{
    free(array[0]);
    free(array);
}

