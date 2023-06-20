#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer *l_input = NULL;
static Layer *l_c1 = NULL;
static Layer *l_s1 = NULL;
static Layer *l_f = NULL;

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();
static int write_int(int fd, int src);
static int write_layer(int fd, Layer *l);
static int load_layer(int fd, Layer **l);
static int load_model(const char *file);
static int write_int(int fd, int src);
static int write_layer(int fd, Layer *l);
static int save_model(const char *file);

static inline void loaddata(int status)
{
	if (status == -1) {
	        mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		        &train_set, &train_cnt);
	}
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, const  char **argv)
{
	struct stat buf;
	int status;
	srand(time(NULL));
	CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
		return 1;
	}
	if (argc == 1 || (argc >= 2 && (status = stat(argv[1], &buf)) == -1)) {
		l_input = new Layer(0, 0, 28*28);
		l_c1 = new Layer(5*5, 6, 24*24*6);
		l_s1 = new Layer(4*4, 1, 6*6*6);
		l_f = new Layer(6*6*6, 10, 10);
		loaddata(-1);
	        learn();
	} else {
		if (load_model(argv[1]) != 0) {
			goto err;
		}
		if (argc == 3) {
			if (stat(argv[2], NULL) == -1) {
				int lines = atoi(argv[2]);
				char file[255];
				for (int i = 0;i < lines;i++) {
					scanf("%s", file);
					mnist_load(file, NULL, &test_set, &test_cnt);
					classify(test_set[0].data);
				}
			} else {
				mnist_load(argv[2], NULL, &test_set, &test_cnt);
				classify(test_set[0].data);
			}
			goto cleanup;
		}
		loaddata(status);
	}
	test();
	if (argc >= 2 && status == -1 && save_model(argv[1])) {
		return errno;
	}

cleanup:
	delete l_input;
	delete l_c1;
	delete l_s1;
	delete l_f;
	return 0;
err:
	fprintf(stderr, strerror(errno));
	delete l_input;
	delete l_c1;
	delete l_s1;
	delete l_f;
	return errno;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	l_input->clear();
	l_c1->clear();
	l_s1->clear();
	l_f->clear();

	clock_t start, end;
	start = clock();

	l_input->setOutput((float *)input);
	
	fp_preact_c1<<<64, 64>>>((float (*)[28])l_input->output, (float (*)[24][24])l_c1->preact, (float (*)[5][5])l_c1->weight);
	fp_bias_c1<<<64, 64>>>((float (*)[24][24])l_c1->preact, l_c1->bias);
	apply_step_function<<<64, 64>>>(l_c1->preact, l_c1->output, l_c1->O);

	fp_preact_s1<<<64, 64>>>((float (*)[24][24])l_c1->output, (float (*)[6][6])l_s1->preact, (float (*)[4][4])l_s1->weight);
	fp_bias_s1<<<64, 64>>>((float (*)[6][6])l_s1->preact, l_s1->bias);
	apply_step_function<<<64, 64>>>(l_s1->preact, l_s1->output, l_s1->O);

	fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s1->output, l_f->preact, (float (*)[6][6][6])l_f->weight);
	fp_bias_f<<<64, 64>>>(l_f->preact, l_f->bias);
	apply_step_function<<<64, 64>>>(l_f->preact, l_f->output, l_f->O);
	
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
	clock_t start, end;

	start = clock();

	bp_weight_f<<<64, 64>>>((float (*)[6][6][6])l_f->d_weight, l_f->d_preact, (float (*)[6][6])l_s1->output);
	bp_bias_f<<<64, 64>>>(l_f->bias, l_f->d_preact);

	bp_output_s1<<<64, 64>>>((float (*)[6][6])l_s1->d_output, (float (*)[6][6][6])l_f->weight, l_f->d_preact);
	bp_preact_s1<<<64, 64>>>((float (*)[6][6])l_s1->d_preact, (float (*)[6][6])l_s1->d_output, (float (*)[6][6])l_s1->preact);
	bp_weight_s1<<<64, 64>>>((float (*)[4][4])l_s1->d_weight, (float (*)[6][6])l_s1->d_preact, (float (*)[24][24])l_c1->output);
	bp_bias_s1<<<64, 64>>>(l_s1->bias, (float (*)[6][6])l_s1->d_preact);

	bp_output_c1<<<64, 64>>>((float (*)[24][24])l_c1->d_output, (float (*)[4][4])l_s1->weight, (float (*)[6][6])l_s1->d_preact);
	bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1->d_preact, (float (*)[24][24])l_c1->d_output, (float (*)[24][24])l_c1->preact);
	bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1->d_weight, (float (*)[24][24])l_c1->d_preact, (float (*)[28])l_input->output);
	bp_bias_c1<<<64, 64>>>(l_c1->bias, (float (*)[24][24])l_c1->d_preact);


	apply_grad<<<64, 64>>>(l_f->weight, l_f->d_weight, l_f->M * l_f->N);
	apply_grad<<<64, 64>>>(l_s1->weight, l_s1->d_weight, l_s1->M * l_s1->N);
	apply_grad<<<64, 64>>>(l_c1->weight, l_c1->d_weight, l_c1->M * l_c1->N);

	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Unfold the input layer
static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j) {
			int b = 0;
			for (int x = i; x < i + 2; ++x)
				for (int y = j; y < j+2; ++y)
					unfolded[a][b++] = input[x][y];
			a++;
		}
}

static void learn()
{
	static cublasHandle_t blas;
	cublasCreate(&blas);

	float err;
	int iter = 50;
	
	double time_taken = 0.0;

	fprintf(stdout ,"Learning\n");

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i) {
			float tmp_err;

			time_taken += forward_pass(train_set[i].data);

			l_f->bp_clear();
			l_s1->bp_clear();
			l_c1->bp_clear();

			// Euclid distance of train_set[i]
			makeError<<<10, 1>>>(l_f->d_preact, l_f->output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, l_f->d_preact, 1, &tmp_err);
			err += tmp_err;

			time_taken += back_pass();
		}

		err /= train_cnt;
		fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err, time_taken);

		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}

	}
	
	fprintf(stdout, "\n Time - %lf\n", time_taken);
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
	float res[10];

	forward_pass(data);

	unsigned int max = 0;

	cudaMemcpy(res, l_f->output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}

static int load_layer(int fd, Layer **l) {
	int M, N, O;
	if (read(fd, &M, sizeof(int)) == -1) {
		return errno;
	}
	if (read(fd, &N, sizeof(int)) == -1) {
		return errno;
	}
	if (read(fd, &O, sizeof(int)) == -1) {
		return errno;
	}
	float bias[N];
	float weight[N * M];
	if (read(fd, bias, sizeof(float) * N) == -1) {
		return errno;
	}
	if (read(fd, weight, sizeof(float) * N * M) == -1) {
		return errno;
	}
	*l = new Layer(M, N, O, bias, weight);
	return 0;
}

// Load NN model
static int load_model(const char *file) {
	int fd;
	if ((fd = open(file, O_RDONLY)) == -1) {
		goto err;
	}
	if (load_layer(fd, &l_input) == -1) {
		goto err;
	}
	if (load_layer(fd, &l_c1) == -1) {
		goto err;
	}
	if (load_layer(fd, &l_s1) == -1) {
		goto err;
	}
	if (load_layer(fd, &l_f) == -1) {
		goto err;
	}
	close(fd);
	return 0;
err:
	if (fd != -1) {
		close(fd);
	}
	return errno;
}

static int write_int(int fd, int src) {
	if (write(fd, &src, sizeof(int)) == -1) {
		return errno;
	}
	return 0;
}

// Write Layer
static int write_layer(int fd, Layer *l) {
	float bias[l->N];
	float weight[l->N][l->M];
	//write metadata
	if (write_int(fd, l->M) == -1) {
		return errno;
	}
	if (write_int(fd, l->N) == -1) {
		return errno;
	}
	if (write_int(fd, l->O) == -1) {
		return errno;
	}
	cudaMemcpy(bias, l->bias, sizeof(float) * l->N, cudaMemcpyDeviceToHost);
	if (write(fd, bias, sizeof(float) * l->N) == -1) {
		return errno;
	}
	cudaMemcpy(weight, l->weight, sizeof(float) * l->N * l->M, cudaMemcpyDeviceToHost);
	for (int i = 0;i < l->N;i++) {
		if (write(fd, weight[i], sizeof(float) * l->M) == -1) {
			return errno;
		}
	}
	return 0;
}

// Save NN model
static int save_model(const char *file) {
	int fd;
	if ((fd = open(file, O_WRONLY | O_CREAT | O_TRUNC, 0644)) == -1) {
		return errno;
	}
	if (write_layer(fd, l_input) == -1) {
		return errno;
	}
	if (write_layer(fd, l_c1) == -1) {
		return errno;
	}
	if (write_layer(fd, l_s1) == -1) {
		return errno;
	}
	if (write_layer(fd, l_f) == -1) {
		return errno;
	}
	close(fd);
	return 0;
}
