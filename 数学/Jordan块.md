## Jordan块的逆
特征值为倒数，代数重数与几何重数都与$J(\lambda,k)$相同

$$
\left(J(\lambda,k)\right)^{-1}=\left(\begin{array}{cccccc}
\frac{1}{\lambda} & \frac{-1}{\lambda^2} & \frac{1}{\lambda^3} & \frac{-1}{\lambda^4} & \cdots & \frac{(-1)^{k-1}}{\lambda^k} \\
0 & \frac{1}{\lambda} & \frac{-1}{\lambda^2} & \frac{1}{\lambda^3} & \cdots & \frac{(-1)^{k-2}}{\lambda^{k-1}} \\
0 & 0 & \frac{1}{\lambda} & \frac{-1}{\lambda^2} & \cdots & \frac{(-1)^{k-3}}{\lambda^{k-2}} \\
0 & 0 & 0 & \frac{1}{\lambda} & \cdots & \frac{(-1)^{k-4}}{\lambda^{k-3}} \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \frac{1}{\lambda}
\end{array}\right)
$$

即
$$
J(\lambda,k)^{-1}=\frac{1}{\lambda} I-\frac{1}{\lambda^2} N_k+\frac{1}{\lambda^3} N_k^2-\ldots+(-1)^{k-1} \frac{1}{\lambda^k} N_k^{k-1}
$$
其中$N_k$为标准幂零矩阵
每上一个对角线，取反并次数加一