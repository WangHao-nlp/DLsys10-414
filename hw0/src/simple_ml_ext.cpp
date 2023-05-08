#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


//def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
//    n = X.shape[0]
//    step = n // batch
//    index = np.arange(batch)
//    for i in range(step + 1):
//        start = i * batch
//        end = min(start + batch, n)
//        if start == end:
//           break
//        x1 = X[start: end]
//        y1 = y[start: end]
//        z = softmax(np.matmul(x1, theta)) #过softmax 
//        z[index, y1] -= 1  #每行标签位置减去1
//        # 也可以写成
//        # I = np.zeros_like(z)
//        # I[np.arange(x1.shape[0]), y_1] = 1
//        # z = z-I
//        grad = np.matmul(x1.transpose(), z) / batch # X转置乘z再除以batch（loss要除以batch）
//        theta -= lr * grad # 更新 
//        # 我之前一直以为是theta *= （1-lr*grad）， grad就是要减去的部分， 不用再乘本身值了

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // 不考虑无法整除的情况
    for(size_t num=0; num<m/batch; num++){
        size_t base = num*batch*n;
        float *Z = new float[batch*k]; // 中间变量
        // exp(np.matmul(x1, theta))
        for(size_t i=0; i<batch; i++){
        	  for(size_t j=0; j<k; j++){
        	  	float sum = 0;
        	  	// Z[i][j] = sum(X[i][x]*theta[x][j])
        	  	for(size_t x=0; x<n; x++){
        	  		sum+= X[base+i*n+x]*theta[x*k+j];
        	  	}
        	  	Z[i*k+j] = exp(sum);
        	  }
        }
        //softmax
        float *Z_sum = new float[batch];
        for (size_t i=0; i<batch; i++){
        	  float sum = 0;
        	  for(size_t j=0; j<k; j++){
        	  	sum += Z[i*k+j];
        	  }
        	  Z_sum[i] = sum;
        }
        for(size_t i=0; i<batch; i++){
        	 for(size_t j = 0; j<k; j++){
        	 	Z[i*k+j]/=Z_sum[i];
        	 }
        }
        // Z-I
        for(size_t i=0; i<batch; i++){
        	 Z[i*k+y[num*batch+i]] -= 1.0;
        }
        // X.T@Z
        for(size_t i=0; i<n; i++){
        	  for(size_t j=0; j<k; j++){
        	  	 float sum = 0;
        	  	 // dtheta[i][j] = sum(X.T[i][x]*Z[x][j])
        	  	 // X.T[i][x] = X[x][i]
        	  	 for(size_t x=0; x<batch; x++){
        	  	 	sum += X[base+x*n+i]*Z[x*k+j];
        	  	 }
        	  	 theta[i*k+j] -= lr*sum/batch; 
        	  }
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
